# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 19:25:22 2024

@author: jerem
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QThread, pyqtSignal

import paramiko_SSH_SCP_class
from pathlib import Path

import os
import numpy as np
import time
from scipy import integrate
import glob
from utils import Popup, encrypt

class ThreadDownloadSimJSON(QtCore.QThread):
    def __init__(self, file_path, local_folder, parent=None):
        super(QtCore.QThread, self).__init__()
        self.file_path = file_path
        self.local_folder = local_folder
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    def downloadSimJSON(self, file_path, local_folder):
        print("THREAD downloadSimJSON")
        host = "llrlsi-gw.in2p3.fr"
        user = "jeremy"
        with open(f"{os.environ['SMILEI_QT']}\\..\\..\\tornado_pwdfile.txt",'r') as f: pwd_crypt = f.read()
        pwd = encrypt(pwd_crypt,-2041000*2-1)
        remote_path = r"\sps3\jeremy\LULI\simulations_info.json"
        ssh_key_filepath = r"C:\Users\Jeremy\.ssh\id_rsa.pub"
        self.remote_client = paramiko_SSH_SCP_class.RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
        self.remote_client.execute_commands(["(source /usr/share/Modules/init/bash; unset MODULEPATH; module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7; module load python/3.7.0; python /sps3/jeremy/LULI/check_sim_state_py.py)&"])
        self.remote_client.download_file(file_path, local_folder)
        return
    def run(self):
        """Long-running task."""
        self.downloadSimJSON(self.file_path, self.local_folder)
        self.finished.emit()

class ThreadDownloadSimData(QtCore.QThread):
    def __init__(self, job_full_path, parent=None):
        super(QtCore.QThread, self).__init__()
        self.job_full_path = job_full_path
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    def downloadSimData(self, job_full_path):
        print("THREAD downloadSimData")
        general_folder_name = job_full_path[27:] #/grid_mnt/sps3/jeremy/LULI/xxxxxxxx
        local_folder = os.environ["SMILEI_CLUSTER"]
        local_cluster_folder = f"{local_folder}\\{general_folder_name}"
        files = glob.glob(f'{local_cluster_folder}\\*')

        for f in files:
            print("removed",f)
            os.remove(f)

        Path(local_cluster_folder).mkdir(parents=True, exist_ok=True)
        print(f"Downloading in {local_cluster_folder}")
        host = "llrlsi-gw.in2p3.fr"
        user = "jeremy"
        with open(f"{os.environ['SMILEI_QT']}\\..\\..\\tornado_pwdfile.txt",'r') as f: pwd_crypt = f.read()
        pwd = encrypt(pwd_crypt,-2041000*2-1)
        # print(pwd_crypt,pwd)
        remote_path = "/sps3/jeremy/LULI/"
        ssh_key_filepath = r"C:\Users\Jeremy\.ssh\id_rsa.pub"
        remote_client = paramiko_SSH_SCP_class.RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
        _, list_of_files_raw, _ = remote_client.connection.exec_command(f"ls {job_full_path}")

        list_of_files = [job_full_path+"/"+s.rstrip() for s in list_of_files_raw]
        print("-----------------------------")
        print("\n".join(list_of_files)[27:])
        print("-----------------------------")
        remote_client.bulk_download(list_of_files, local_cluster_folder)
        return
    def run(self):
        """Long-running task."""
        self.downloadSimData(self.job_full_path)
        self.finished.emit()

class ThreadGetFieldsProbeData(QtCore.QThread):
    def __init__(self,  boolList, fields_names, S, fields_t_range, fields_paxisX, fields_paxisY, fields_paxisZ, parent=None):
        super(QtCore.QThread, self).__init__()

        (self.boolList, self.fields_names, self.S, self.fields_t_range,
         self.fields_paxisX, self.fields_paxisY, self.fields_paxisZ) = (boolList, fields_names, S, fields_t_range,
                                                                        fields_paxisX, fields_paxisY, fields_paxisZ)
    finished = pyqtSignal(list)

    def getFieldsProbeData(self, boolList, fields_names, S, fields_t_range, fields_paxisX, fields_paxisY, fields_paxisZ):
        print("THREAD getFieldsProbeData")
        fields_data_list = []
        for i in range(len(fields_names)):
            if boolList[i]:
                # print(fields_name)
                if fields_names[i]=="Er":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Er = (Y*np.array(S.Probe("fields","Ey").getData()).astype(np.float32)
                              + Z*np.array(S.Probe("fields","Ez").getData()).astype(np.float32))/np.sqrt(Y**2+Z**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Er)
                    del T,X,Y,Z,Er
                elif fields_names[i]=="Eθ":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Etheta = (Y*np.array(S.Probe("fields","Ez").getData()).astype(np.float32)
                              - Z*np.array(S.Probe("fields","Ey").getData()).astype(np.float32))/np.sqrt(Y**2+Z**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Etheta)
                    del T,X,Y,Z,Etheta
                elif fields_names[i]=="Br":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Br = (Y*np.array(S.Probe("fields","By").getData()).astype(np.float32)
                              + Z*np.array(S.Probe("fields","Bz").getData()).astype(np.float32))/np.sqrt(Y**2+Z**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Br)
                    del T,X,Y,Z,Br
                elif fields_names[i]=="Bθ":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Btheta = (Y*np.array(S.Probe("fields","Bz").getData()).astype(np.float32)
                              - Z*np.array(S.Probe("fields","By").getData()).astype(np.float32))/np.sqrt(Y**2+Z**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Btheta)
                    del T,X,Y,Z,Btheta

                else: fields_data_list.append(np.array(S.Probe("fields",fields_names[i]).getData()).astype(np.float32))

        return fields_data_list
    def run(self):
        """Long-running task."""
        fields_data_list = self.getFieldsProbeData(self.boolList, self.fields_names, self.S, self.fields_t_range, self.fields_paxisX, self.fields_paxisY, self.fields_paxisZ)
        self.finished.emit(fields_data_list)

class ThreadGetPlasmaProbeData(QtCore.QThread):
    def __init__(self,  S, selected_plasma_names, parent=None):
        super(QtCore.QThread, self).__init__()
        self.S, self.selected_plasma_names = S, selected_plasma_names

    finished = pyqtSignal(list)

    def getPlasmaProbeData(self, S, selected_plasma_names):
        print("THREAD getPlasmaProbeData")
        plasma_data_list = []

        # t0 = time.perf_counter()
        ne = S.namelist.ne
        toTesla = 10709
        for i in range(len(selected_plasma_names)):
            # try:
                if selected_plasma_names[i] == "Bx":
                    Bx_long_diag = S.Probe("long","Bx")
                    plasma_data_list.append(np.array(Bx_long_diag.getData())*toTesla)
                elif selected_plasma_names[i] == "Bx_av":
                    Bx_av_long_diag = S.Field("Bx_av","Bx_m")
                    plasma_data_list.append(np.array(Bx_av_long_diag.getData())*toTesla)
                elif selected_plasma_names[i] == "Bx_trans":
                    Bx_trans_diag = S.Probe("trans","Bx")
                    plasma_data_list.append(np.array(Bx_trans_diag.getData())*toTesla)

                elif selected_plasma_names[i] == "ne":
                    Bweight_long = S.ParticleBinning("2D_weight")
                    plasma_data_list.append(np.mean(np.array(Bweight_long.getData())/ne,axis=-1))
                elif selected_plasma_names[i] == "ne_av":
                    Bweight_long = S.ParticleBinning("2D_weight_av")
                    plasma_data_list.append(np.mean(np.array(Bweight_long.getData())/ne,axis=-1))
                elif selected_plasma_names[i] == "ne_trans":
                    Bweight_trans = S.ParticleBinning("2D_weight_trans")
                    plasma_data_list.append(np.array(Bweight_trans.getData())/ne)
                elif selected_plasma_names[i] == "ni":
                    Bweight_ions_long = S.ParticleBinning("2D_weight_ions")
                    plasma_data_list.append(np.mean(np.array(Bweight_ions_long.getData())/ne,axis=-1))

                # elif selected_plasma_names[i] == "Lx_av":
                #     BLx_long = S.ParticleBinning("2D_Lx")
                #     plasma_data_list.append(np.mean(np.array(BLx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "Lx_trans":
                    BLx_trans = S.ParticleBinning("2D_Lx_trans")
                    plasma_data_list.append(np.array(BLx_trans.getData()))

                # elif selected_plasma_names[i] == "Rho":
                #     BRho_long_diag = S.Probe("long","Rho")
                #     plasma_data_list.append(np.array(BRho_long_diag.getData())*toTesla)
                # elif selected_plasma_names[i] == "Rho_trans":
                #     BRho_trans_diag = S.Probe("trans","Rho")
                #     plasma_data_list.append(np.array(BRho_trans_diag.getData())*toTesla)

                elif selected_plasma_names[i] == "jx_av":
                    Bvx_long = S.ParticleBinning("2D_jx")
                    plasma_data_list.append(np.mean(np.array(Bvx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "jx_trans":
                    Bvx_trans = S.ParticleBinning("2D_jx_trans")
                    plasma_data_list.append(np.array(Bvx_trans.getData()))

                elif selected_plasma_names[i] == "rho":
                    Bvx_long = S.ParticleBinning("2D_rho")
                    plasma_data_list.append(np.mean(np.array(Bvx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "rho_trans":
                    Bvx_trans = S.ParticleBinning("2D_rho_trans")
                    plasma_data_list.append(np.array(Bvx_trans.getData()))
                    
                elif selected_plasma_names[i] == "px":
                    Bptheta_long = S.ParticleBinning("2D_px")
                    plasma_data_list.append(np.mean(np.array(Bptheta_long.getData()),axis=-1))

                    
                elif selected_plasma_names[i] == "pθ_av":
                    Bptheta_long = S.ParticleBinning("2D_ptheta")
                    plasma_data_list.append(np.mean(np.array(Bptheta_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "pθ_trans":
                    Bptheta_trans = S.ParticleBinning("2D_ptheta_trans")
                    plasma_data_list.append(np.array(Bptheta_trans.getData()))

                elif selected_plasma_names[i] == "Ekin":
                    BEkin_long = S.ParticleBinning("2D_ekin")
                    plasma_data_list.append(np.mean(np.array(BEkin_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "Ekin_trans":
                    BEkin_trans = S.ParticleBinning("2D_ekin_trans")
                    plasma_data_list.append(np.array(BEkin_trans.getData()))

                elif selected_plasma_names[i] == "Jx":
                    Jx_long_diag = S.Probe("long","Jx")
                    plasma_data_list.append(np.array(Jx_long_diag.getData()))
                elif selected_plasma_names[i] == "Jx_trans":
                    Jx_long_diag = S.Probe("trans","Jx")
                    plasma_data_list.append(np.array(Jx_long_diag.getData()))

                elif selected_plasma_names[i] == "Jθ":
                    Jy_long_diag = S.Probe("long","Jy")
                    Jz_long_diag = S.Probe("long","Jz")
                    Jy_long = np.array(Jy_long_diag.getData())
                    Jz_long = np.array(Jz_long_diag.getData())
                    paxisY = Jy_long_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
                    paxisZ = Jz_long_diag.getAxis("axis2")[:,2] - S.namelist.Ltrans/2 # = 0 everywhere
                    Y,Z = np.meshgrid(paxisY,paxisZ)
                    R = np.sqrt(Y**2+Z**2)
                    Jtheta_long = (Y.T*Jz_long - Z.T*Jy_long)/R.T
                    # print(Jtheta_long.shape)
                    plasma_data_list.append(Jtheta_long)
                    # np.savez(f"{sim_path}/plasma_{selected_plasma_names[i]}.npz", t_range=Jy_long_diag.getTimes(), data=Jtheta_long)
                elif selected_plasma_names[i] == "Jθ_trans":
                    Jy_trans_diag = S.Probe("trans","Jy")
                    Jz_trans_diag = S.Probe("trans","Jz")
                    Jy_trans = np.array(Jy_trans_diag.getData())
                    Jz_trans = np.array(Jz_trans_diag.getData())
                    paxisX = Jy_trans_diag.getAxis("axis1")[:,0]
                    paxisY = Jy_trans_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
                    paxisZ = Jy_trans_diag.getAxis("axis3")[:,2] - S.namelist.Ltrans/2
                    X,Y,Z = np.meshgrid(paxisX,paxisY,paxisZ,indexing="ij")
                    R = np.sqrt(Y**2+Z**2)
                    Jtheta_trans = (Y*Jz_trans - Z*Jy_trans)/R
                    plasma_data_list.append(Jtheta_trans)
                else:
                    print(f"'/!\ {selected_plasma_names[i]}' does not exist ! /!\ ")
                    # Popup().showError(f"ThreadGetPlasmaProbeData: '{selected_plasma_names[i]}' diag not registered !")

        # t1 = time.perf_counter()
        # print(round(t1-t0,2),"s")
        return plasma_data_list, selected_plasma_names
    
    def getPlasmaProbeData_OLD(self, S, selected_plasma_names):
        print("THREAD getPlasmaProbeData")
        plasma_data_list = []

        # t0 = time.perf_counter()
        ne = S.namelist.ne
        toTesla = 10709
        for i in range(len(selected_plasma_names)):
            # try:
                if selected_plasma_names[i] == "Bx":
                    Bx_long_diag = S.Probe("long","Bx")
                    plasma_data_list.append(np.array(Bx_long_diag.getData())*toTesla)
                elif selected_plasma_names[i] == "Bx_av":
                    Bx_av_long_diag = S.Field("Bx_av","Bx_m")
                    plasma_data_list.append(np.array(Bx_av_long_diag.getData())*toTesla)
                elif selected_plasma_names[i] == "Bx_trans":
                    Bx_trans_diag = S.Probe("trans","Bx")
                    plasma_data_list.append(np.array(Bx_trans_diag.getData())*toTesla)

                elif selected_plasma_names[i] == "ne":
                    Bweight_long = S.ParticleBinning("weight")
                    plasma_data_list.append(np.mean(np.array(Bweight_long.getData())/ne,axis=-1))
                elif selected_plasma_names[i] == "ne_av":
                    Bweight_long = S.ParticleBinning("weight_av")
                    plasma_data_list.append(np.mean(np.array(Bweight_long.getData())/ne,axis=-1))
                elif selected_plasma_names[i] == "ne_trans":
                    Bweight_trans = S.ParticleBinning("weight_trans")
                    plasma_data_list.append(np.array(Bweight_trans.getData())/ne)
                elif selected_plasma_names[i] == "ni":
                    Bweight_ions_long = S.ParticleBinning("weight_ions")
                    plasma_data_list.append(np.mean(np.array(Bweight_ions_long.getData())/ne,axis=-1))

                elif selected_plasma_names[i] == "Lx_av":
                    BLx_long = S.ParticleBinning("Lx_W")
                    plasma_data_list.append(np.mean(np.array(BLx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "Lx_trans":
                    BLx_trans = S.ParticleBinning("Lx_W_trans")
                    plasma_data_list.append(np.array(BLx_trans.getData()))

                elif selected_plasma_names[i] == "Rho":
                    BRho_long_diag = S.Probe("long","Rho")
                    plasma_data_list.append(np.array(BRho_long_diag.getData())*toTesla)
                elif selected_plasma_names[i] == "Rho_trans":
                    BRho_trans_diag = S.Probe("trans","Rho")
                    plasma_data_list.append(np.array(BRho_trans_diag.getData())*toTesla)

                elif selected_plasma_names[i] == "jx_av":
                    Bvx_long = S.ParticleBinning("jx")
                    plasma_data_list.append(np.mean(np.array(Bvx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "jx_trans":
                    Bvx_trans = S.ParticleBinning("jx_trans")
                    plasma_data_list.append(np.array(Bvx_trans.getData()))

                elif selected_plasma_names[i] == "rho":
                    Bvx_long = S.ParticleBinning("rho")
                    plasma_data_list.append(np.mean(np.array(Bvx_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "rho_trans":
                    Bvx_trans = S.ParticleBinning("rho_trans")
                    plasma_data_list.append(np.array(Bvx_trans.getData()))
                    
                elif selected_plasma_names[i] == "px":
                    Bptheta_long = S.ParticleBinning("px_W")
                    plasma_data_list.append(np.mean(np.array(Bptheta_long.getData()),axis=-1))

                    
                elif selected_plasma_names[i] == "pθ_av":
                    Bptheta_long = S.ParticleBinning("ptheta_W")
                    plasma_data_list.append(np.mean(np.array(Bptheta_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "pθ_trans":
                    Bptheta_trans = S.ParticleBinning("ptheta_W_trans")
                    plasma_data_list.append(np.array(Bptheta_trans.getData()))

                elif selected_plasma_names[i] == "Ekin":
                    BEkin_long = S.ParticleBinning("ekin_W")
                    plasma_data_list.append(np.mean(np.array(BEkin_long.getData()),axis=-1))
                elif selected_plasma_names[i] == "Ekin_trans":
                    BEkin_trans = S.ParticleBinning("ekin_W_trans")
                    plasma_data_list.append(np.array(BEkin_trans.getData()))

                elif selected_plasma_names[i] == "Jx":
                    Jx_long_diag = S.Probe("long","Jx")
                    plasma_data_list.append(np.array(Jx_long_diag.getData()))
                elif selected_plasma_names[i] == "Jx_trans":
                    Jx_long_diag = S.Probe("trans","Jx")
                    plasma_data_list.append(np.array(Jx_long_diag.getData()))

                elif selected_plasma_names[i] == "Jθ":
                    Jy_long_diag = S.Probe("long","Jy")
                    Jz_long_diag = S.Probe("long","Jz")
                    Jy_long = np.array(Jy_long_diag.getData())
                    Jz_long = np.array(Jz_long_diag.getData())
                    paxisY = Jy_long_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
                    paxisZ = Jz_long_diag.getAxis("axis2")[:,2] - S.namelist.Ltrans/2 # = 0 everywhere
                    Y,Z = np.meshgrid(paxisY,paxisZ)
                    R = np.sqrt(Y**2+Z**2)
                    Jtheta_long = (Y.T*Jz_long - Z.T*Jy_long)/R.T
                    # print(Jtheta_long.shape)
                    plasma_data_list.append(Jtheta_long)
                    # np.savez(f"{sim_path}/plasma_{selected_plasma_names[i]}.npz", t_range=Jy_long_diag.getTimes(), data=Jtheta_long)
                elif selected_plasma_names[i] == "Jθ_trans":
                    Jy_trans_diag = S.Probe("trans","Jy")
                    Jz_trans_diag = S.Probe("trans","Jz")
                    Jy_trans = np.array(Jy_trans_diag.getData())
                    Jz_trans = np.array(Jz_trans_diag.getData())
                    paxisX = Jy_trans_diag.getAxis("axis1")[:,0]
                    paxisY = Jy_trans_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
                    paxisZ = Jy_trans_diag.getAxis("axis3")[:,2] - S.namelist.Ltrans/2
                    X,Y,Z = np.meshgrid(paxisX,paxisY,paxisZ,indexing="ij")
                    R = np.sqrt(Y**2+Z**2)
                    Jtheta_trans = (Y*Jz_trans - Z*Jy_trans)/R
                    plasma_data_list.append(Jtheta_trans)
                else:
                    print(f"'/!\ {selected_plasma_names[i]}' does not exist ! /!\ ")
                    # Popup().showError(f"ThreadGetPlasmaProbeData: '{selected_plasma_names[i]}' diag not registered !")
        return plasma_data_list, selected_plasma_names

    def run(self):
        # plasma_data_list,selected_plasma_names = self.getPlasmaProbeData_OLD(self.S, self.selected_plasma_names)

        try:
            plasma_data_list,selected_plasma_names = self.getPlasmaProbeData(self.S, self.selected_plasma_names)
        except:
            try:
                plasma_data_list,selected_plasma_names = self.getPlasmaProbeData_OLD(self.S, self.selected_plasma_names)
            except:
                self.finished.emit([self.selected_plasma_names])
                return

        self.finished.emit([plasma_data_list,selected_plasma_names])
        return

class ThreadGetAMIntegral(QtCore.QThread):
    def __init__(self,  S, parent=None):
        super(QtCore.QThread, self).__init__()
        self.S = S

    finished = pyqtSignal(np.ndarray)

    def getAMIntegral(self, S):
        print("THREAD getAMIntegral")
        # t0 = time.perf_counter()
        fields_paxisX = self.S.Probe("fields","Ex").getAxis("axis1")[:,0]
        fields_paxisY = self.S.Probe("fields","Ex").getAxis("axis2")[:,1]-self.S.namelist.Ltrans/2
        fields_paxisZ = self.S.Probe("fields","Ex").getAxis("axis3")[:,2]-self.S.namelist.Ltrans/2
        X,Y,Z = np.meshgrid(fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
        Ex = np.array(self.S.Probe("fields","Ex").getData()).astype(np.float32)
        Ey = np.array(self.S.Probe("fields","Ey").getData()).astype(np.float32)
        Ez = np.array(self.S.Probe("fields","Ez").getData()).astype(np.float32)
        Bx = np.array(self.S.Probe("fields","Bx").getData()).astype(np.float32)
        By = np.array(self.S.Probe("fields","By").getData()).astype(np.float32)
        Bz = np.array(self.S.Probe("fields","Bz").getData()).astype(np.float32)

        AM_data = Y*(Ex*By-Ey*Bx)-Z*(Ez*Bx-Ex*Bz)
        AM_trans_int = integrate.simpson(integrate.simpson(AM_data[:,:,:,:],x=fields_paxisZ,axis=-1),x=fields_paxisY,axis=-1)
        AM_full_int = integrate.simpson(AM_trans_int,x=fields_paxisX,axis=-1)

        # t1 = time.perf_counter()
        return AM_full_int

    def run(self):
        AM_full_int = self.getAMIntegral(self.S)
        self.finished.emit(AM_full_int)
        
        
        
class ThreadUpdateVerlet(QtCore.QThread):
    def __init__(self,  POS, OLD_POS, SIZES, MOUSE_POS, parent=None):
        super(QtCore.QThread, self).__init__()
        self.POS = POS
        self.OLD_POS = OLD_POS
        self.SIZES = SIZES
        self.MOUSE_POS = MOUSE_POS
        self.dt = 0.1
        self.Window = 200
        self.Size = 10
        self.gravity_acc = np.array([0,-60])

    finished = pyqtSignal(np.ndarray)
    

    def update(self, POS, OLD_POS):
        # t0 = time.perf_counter()
        POS,OLD_POS = self.updatePositions(POS, OLD_POS)
        POS,OLD_POS = self.applyConstraint(POS, OLD_POS)
        POS = self.applyCollision(POS)
        # t1 = time.perf_counter()
        # print((t1-t0)*1000,"ms")
        return POS,OLD_POS
    
    def updatePositions(self, POS,OLD_POS):
        
        for i in range(len(POS)):
            pos, old_pos = POS[i], OLD_POS[i]
            POS[i], OLD_POS[i] = self.updateSinglePos(pos, old_pos)
        
        return POS, OLD_POS
            
        
    def updateSinglePos(self, pos, old_pos):
        vel = pos - old_pos
        old_pos = pos
        
        total_acc = self.getAcc(pos)
    
        pos += vel + total_acc*self.dt**2
        return pos, old_pos
    
    def getAcc(self,pos):
        direction = pos - self.MOUSE_POS
        distance = np.linalg.norm(direction) + 0.1  # Avoid division by zero
        force = 10000 / (distance**(1.5))  # Inverse square law
        mouse_acc = np.array([0,0])
        if distance < self.Window / 3:
                mouse_acc = (direction / distance) * force  # Normalize and apply force
        
        # print(mouse_acc)
        return self.gravity_acc + mouse_acc
        
    
    def applyConstraint(self, POS, OLD_POS):
        # print(POS)
        for i in range(len(POS)):
            new_pos = POS[i]
            # print("Check bounce:",new_pos)
            if new_pos[1] - self.SIZES[i] < 0.2: #Bounce on ground
                # print("BOUNCE ON GROUND")
                new_pos = np.array([new_pos[0],self.SIZES[i]])
            
            if new_pos[0] - self.SIZES[i] < 0.2: #left wall
                new_pos = np.array([self.SIZES[i],new_pos[1]])
            if new_pos[0] + self.SIZES[i] > self.Window-0.2: #right wall
                new_pos = np.array([self.Window-self.SIZES[i],new_pos[1]])
            POS[i] = new_pos
        return POS, OLD_POS
    
    def applyCollision(self, POS):
        # print(POS)
        for i in range(len(POS)):
            pos1 = POS[i]
            for j in range(len(POS)):
                if i==j: continue
                pos2 = POS[j]
                axis = pos1-pos2
                dist = np.sqrt(axis[0]**2+axis[1]**2)
                if ((dist < (self.SIZES[i]+self.SIZES[j])) & (dist>0.001)):
                    u = axis/dist
                    delta = (self.SIZES[i]+self.SIZES[j]) - dist
                    coef = 1#0.5 #viscosity
                    POS[i] += 0.5*delta*u*coef
                    POS[j] -= 0.5*delta*u*coef
        return POS

    def run(self):
        POS,OLD_POS = self.update(self.POS,self.OLD_POS)
        self.finished.emit(np.array([POS,OLD_POS]))