# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:54:10 2024

@author: Jeremy La Porte
"""
import sys
from sys import getsizeof
import os
module_dir_happi = 'C:/Users/jerem/Smilei'
sys.path.insert(0, module_dir_happi)
import happi

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtCore import QThreadPool

import qdarktheme

import numpy as np
from numpy import sqrt, cos, sin, exp, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import time
import gc
import psutil
from scipy.interpolate import griddata
from scipy import integrate

import ctypes

from log_dialog import *
from py_dialog import *

from paramiko_SSH_SCP_class import *
import subprocess
import json
from pathlib import Path
from functools import partial
# from win11toast import toast
from pyqttoast import Toast, ToastPreset
import decimal

class ThreadDownloadSimJSON(QtCore.QThread):
    def __init__(self, file_path, local_folder, parent=None):
        super(QtCore.QThread, self).__init__()
        self.file_path = file_path
        self.local_folder = local_folder
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    def downloadSimJSON(self, file_path, local_folder):
        print("download JSON TRND")
        host = "llrlsi-gw.in2p3.fr"
        user = "jeremy"
        with open('tornado_pwdfile.txt', 'r') as f: password = f.read()
        remote_path = r"\sps3\jeremy\LULI\simulation_info.json"
        ssh_key_filepath = r"C:\Users\jerem\.ssh\id_rsa.pub"
        self.remote_client = RemoteClient(host,user,password,ssh_key_filepath,remote_path)
        self.remote_client.execute_commands(["python3 /sps3/jeremy/LULI/check_sim_state_py.py"])
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
        print("-----------------------------")
        general_folder_name = job_full_path[18:]
        local_folder = os.environ["SMILEI_CLUSTER"]
        local_cluster_folder = f"{local_folder}\\{general_folder_name}"
        Path(local_cluster_folder).mkdir(parents=True, exist_ok=True)
        print(f"Downloading in {local_cluster_folder}")
        host = "llrlsi-gw.in2p3.fr"
        user = "jeremy"
        with open('tornado_pwdfile.txt', 'r') as f: password = f.read()
        remote_path = "/sps3/jeremy/LULI/"
        ssh_key_filepath = r"C:\Users\jerem\.ssh\id_rsa.pub"
        remote_client = RemoteClient(host,user,password,ssh_key_filepath,remote_path)
        _, list_of_files_raw, _ = remote_client.connection.exec_command(f"ls {job_full_path}")

        list_of_files = [job_full_path+"/"+s.rstrip() for s in list_of_files_raw]
        print("-----------------------------")
        print("\n".join(list_of_files)[18:])
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
        fields_data_list = []
        for i in range(len(fields_names)):
            if boolList[i]:
                # print(fields_name)
                if fields_names[i]=="Er":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Er = (X*np.array(S.Probe(0,"Ex").getData()).astype(np.float32)
                              + Y*np.array(S.Probe(0,"Ey").getData()).astype(np.float32))/np.sqrt(X**2+Y**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Er)
                    del T,X,Y,Z,Er
                elif fields_names[i]=="Eθ":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Etheta = (X*np.array(S.Probe(0,"Ey").getData()).astype(np.float32)
                              - Y*np.array(S.Probe(0,"Ex").getData()).astype(np.float32))/np.sqrt(X**2+Y**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Etheta)
                    del T,X,Y,Z,Etheta
                elif fields_names[i]=="Br":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Br = (X*np.array(S.Probe(0,"Bx").getData()).astype(np.float32)
                              + Y*np.array(S.Probe(0,"By").getData()).astype(np.float32))/np.sqrt(X**2+Y**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Br)
                    del T,X,Y,Z,Br
                elif fields_names[i]=="Bθ":
                    T,X,Y,Z = np.meshgrid(fields_t_range, fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")
                    Btheta = (X*np.array(S.Probe(0,"By").getData()).astype(np.float32)
                              - Y*np.array(S.Probe(0,"Bx").getData()).astype(np.float32))/np.sqrt(X**2+Y**2) #TO VERIFY IF NOT USE A TRANSPOSE
                    fields_data_list.append(Btheta)
                    del T,X,Y,Z,Btheta

                else: fields_data_list.append(np.array(S.Probe(0,fields_names[i]).getData()).astype(np.float32))

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
        plasma_data_list = []

        t0 = time.perf_counter()
        ne = S.namelist.ne
        toTesla = 10709

        for i in range(len(selected_plasma_names)):
            if selected_plasma_names[i] == "Bz":
                Bz_long_diag = S.Probe("long","Bz")
                plasma_data_list.append(np.array(Bz_long_diag.getData())*toTesla)
            elif selected_plasma_names[i] == "Bz_trans":
                Bz_trans_diag = S.Probe("trans","Bz")
                plasma_data_list.append(np.array(Bz_trans_diag.getData())*toTesla)

            elif selected_plasma_names[i] == "ne":
                Bweight_long = S.ParticleBinning("weight")
                plasma_data_list.append(np.array(Bweight_long.getData())/ne)
            elif selected_plasma_names[i] == "ne_trans":
                Bweight_trans = S.ParticleBinning("weight_trans")
                plasma_data_list.append(np.array(Bweight_trans.getData())/ne)

            elif selected_plasma_names[i] == "Lz":
                BLz_long = S.ParticleBinning("Lz_W")
                plasma_data_list.append(np.array(BLz_long.getData()))
            elif selected_plasma_names[i] == "Lz_trans":
                BLz_trans = S.ParticleBinning("Lz_W_trans")
                plasma_data_list.append(np.array(BLz_trans.getData()))

            elif selected_plasma_names[i] == "ptheta":
                Bptheta_long = S.ParticleBinning("ptheta_W")
                plasma_data_list.append(np.array(Bptheta_long.getData()))
            elif selected_plasma_names[i] == "ptheta_trans":
                Bptheta_trans = S.ParticleBinning("ptheta_W_trans")
                plasma_data_list.append(np.array(Bptheta_trans.getData()))

            elif selected_plasma_names[i] == "Jtheta":
                Jx_long_diag = S.Probe("long","Jx")
                Jy_long_diag = S.Probe("long","Jy")
                Jx_long = np.array(Jx_long_diag.getData())
                Jy_long = np.array(Jy_long_diag.getData())
                paxisX = Jx_long_diag.getAxis("axis1")[:,0] - S.namelist.Ltrans/2
                paxisY = Jx_long_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2

                X,Y = np.meshgrid(paxisX,paxisY)

                R = np.sqrt(X**2+Y**2)
                Jtheta_long = (X.T*Jy_long - Y.T*Jx_long)/R.T
                print(Jtheta_long.shape)
                plasma_data_list.append(Jtheta_long)

            elif selected_plasma_names[i] == "Jtheta_trans":
                Jx_trans_diag = S.Probe("trans","Jy")
                Jy_trans_diag = S.Probe("trans","Jy")
                Jx_trans = np.array(Jx_trans_diag.getData())
                Jy_trans = np.array(Jy_trans_diag.getData())
                paxisX = Jy_trans_diag.getAxis("axis1")[:,0] - S.namelist.Ltrans/2
                paxisY = Jy_trans_diag.getAxis("axis2")[:,1] - S.namelist.Ltrans/2
                paxisZ = Jy_trans_diag.getAxis("axis3")[:,2]
                X,Y,Z = np.meshgrid(paxisX,paxisY,paxisZ,indexing="ij")

                R = np.sqrt(X**2+Y**2)
                Jtheta_trans = (X*Jy_trans - Y*Jx_trans)/R
                print(Jtheta_trans.shape)
                plasma_data_list.append(Jtheta_trans)

            else:
                raise(f"{selected_plasma_names[i]} does not exist !")

        t1 = time.perf_counter()
        print(round(t1-t0,2),"s")
        return plasma_data_list, selected_plasma_names

    def run(self):
        plasma_data_list,selected_plasma_names = self.getPlasmaProbeData(self.S, self.selected_plasma_names)
        self.finished.emit([plasma_data_list,selected_plasma_names])



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # super().__init__()
        print("... INIT ...")
        screen = app.primaryScreen()
        size = screen.size()

        print("SCREEN SIZE:",size.width(),"x",size.height())
        is_screen_2K = size.width() > 1920

        window_height = int(size.height()/1.5)
        window_width = int(size.width()/1.5)

        self.thread = QThreadPool()


        groupBox_bg_color_light = "#f0f0f0"
        groupBox_border_color_light = "#FF17365D"

        font_color_light = "black"
        font_color_dark = "white"

        self.qss_button = """
        QPushButton {
            border-width: 2px;
            border-color: gray;
        }
        """
        self.light_qss_groupBox = """
        QGroupBox  {
        border: 1px solid gray;
        border-color: #FF17365D;
        margin-top: 7px;
        font-size: 10.5pt;
        border-radius: 15px;
        background-color:#f0f0f0;
        font-size:10pt;
        }
        """
        self.dark_qss_groupBox = """
        QGroupBox  {
        border: 1px solid gray;
        border-color: #0017365D;
        margin-top: 7px;
        font-size: 10.5pt;
        border-radius: 15px;
        background-color:#101010;
        font-size: 13;
        }
        """
        self.light_qss_tab = """
        QTabBar::tab {
        border: 1px solid lightgray;
        color: black;
        }
        """
        self.dark_qss_tab = """
        QTabBar::tab {
        border: 1px solid lightgray;
        color: white;
        }
        """
        self.light_qss_label = """
        QLabel {
        color : black;
        }
        """
        self.dark_qss_label = """
        QLabel {
        color : white;
        }
        """

        self.qss_progressBar = """
        QProgressBar::chunk {
            background-color: rgb(0, 179, 0);
            background-color: qlineargradient(x1: 0, y1: 0.5, x2: 1, y2: 0.5,
                            stop: 0 green,
                            stop: 1 lightgreen);
        }
        QProgressBar {
            border: 2px solid #acacac;
            border-radius: 0px;
            text-align: center;
            color: black;
            background: #dfdfdf;
            }
        """
        self.qss_progressBar_COMPLETED = """
        QProgressBar::chunk {
            background-color: rgb(0, 179, 0);
            background-color: qlineargradient(x1: 0, y1: 0.5, x2: 1, y2: 0.5,
                            stop: 0 #c30010,
                            stop: 1 #ee6b6e);
        }
        QProgressBar {
            border: 2px solid #acacac;
            border-radius: 0px;
            text-align: center;
            color: black;
            background: #dfdfdf;
            }
        """
        self.qss_progressBar_DOWNLOADED = """
        QProgressBar::chunk {
            background-color: rgb(0, 179, 0);
            background-color: qlineargradient(x1: 0, y1: 0.5, x2: 1, y2: 0.5,
                            stop: 0 #083d58,
                            stop: 1 #0e6e9e);
        }
        QProgressBar {
            border: 2px solid #acacac;
            border-radius: 0px;
            text-align: center;
            color: black;
            background: #dfdfdf;
            }
        """

        self.theme = "light"
        if self.theme == "dark":
            self.qss = self.dark_qss_groupBox + self.dark_qss_tab + self.dark_qss_label + self.qss_button + self.qss_progressBar
        else:
            self.qss = self.light_qss_groupBox + self.light_qss_tab + self.light_qss_label + self.qss_button + self.qss_progressBar

        qdarktheme.setup_theme(self.theme,additional_qss=self.qss)
        self.setGeometry(175,125,window_width,window_height)

        #==============================
        # FONTS
        #==============================
        self.bold_FONT=QtGui.QFont("Arial", 10,QFont.Bold)
        self.small_bold_FONT=QtGui.QFont("Arial", 8,QFont.Bold)
        self.medium_bold_FONT=QtGui.QFont("Arial", 12,QFont.Bold)

        self.medium_FONT=QtGui.QFont("Arial", 11)

        self.float_validator = QtGui.QDoubleValidator(0.00, 999.99, 2)
        self.float_validator.setLocale(QtCore.QLocale("en_US"))

        self.int_validator = QtGui.QIntValidator(1, 999)
        self.int_validator.setLocale(QtCore.QLocale("en_US"))

        self.MEMORY = psutil.virtual_memory
        self.SCRIPT_VERSION ='0.5.3 "Scalar + plasma + labels"'
        self.COPY_RIGHT = "Jeremy LA PORTE"
        self.spyder_default_stdout = sys.stdout

        #==============================
        # MENU BAR
        #==============================
        layoutMenuBar = QtWidgets.QVBoxLayout()


        menu = self.menuBar()

        self.menuBar = self.menuBar()
        # self.menuBar.setGeometry(QtCore.QRect(0, 0, window_width, 21))
        self.fileMenu = self.menuBar.addMenu("&File")
        self.editMenu = self.menuBar.addMenu("&Edit")

        #Actions

        self.actionOpenSim = QtWidgets.QAction("Open Simulation",self)
        self.actionOpenLogs = QtWidgets.QAction("Open Logs",self)
        self.actionOpenIPython = QtWidgets.QAction("Open IPython",self)

        self.actionDiagScalar = QtWidgets.QAction("Scalar",self)
        self.actionDiagFields = QtWidgets.QAction("Fields",self)
        self.actionDiagTrack = QtWidgets.QAction("Track",self)
        self.actionDiagPlasma = QtWidgets.QAction("Plasma",self)
        self.actionTornado = QtWidgets.QAction("Tornado",self)

        self.actionDiagScalar.setCheckable(True)
        self.actionDiagFields.setCheckable(True)
        self.actionDiagTrack.setCheckable(True)
        self.actionDiagPlasma.setCheckable(True)

        self.actionTornado.setCheckable(True)

        self.fileMenu.addAction(self.actionOpenSim)
        self.fileMenu.addAction(self.actionOpenLogs)
        self.fileMenu.addAction(self.actionOpenIPython)
        self.menuBar.addAction(self.fileMenu.menuAction())

        self.editMenu.addAction(self.actionDiagScalar)
        self.editMenu.addAction(self.actionDiagFields)
        self.editMenu.addAction(self.actionDiagTrack)
        self.editMenu.addAction(self.actionDiagPlasma)
        self.editMenu.addAction(self.actionTornado)
        self.menuBar.addAction(self.editMenu.menuAction())


        #==============================
        # SETTINGS
        #==============================
        self.settings_groupBox = QtWidgets.QGroupBox("Open Simulation")
        self.settings_groupBox.setMinimumWidth(300)
        self.settings_groupBox.setMaximumWidth(300)
        boxLayout_settings = QtWidgets.QVBoxLayout()

        # self.cluster_base_path_LEDIT = QtWidgets.QLineEdit("D:/JLP/CMI/_MASTER 2_/_STAGE_LULI_/CLUSTER")
        # self.simulation_directory_LEDIT = QtWidgets.QLineEdit("LGConv_dx64")
        self.sim_directory_name_LABEL = QtWidgets.QLabel("")
        self.sim_directory_name_LABEL.setFont(self.small_bold_FONT)
        self.sim_directory_name_LABEL.setWordWrap(True)

        # layoutSimulationBasePath = self.creatPara("Cluster directory :",self.sim_directory_name_LABEL)
        # layoutSimulationName = self.creatPara("Simulation :",self.sim_directory_name_LABEL,fontsize=9)


        self.load_sim_BUTTON = QtWidgets.QPushButton('Open')
        self.load_status_LABEL = QtWidgets.QLabel("")
        self.load_status_LABEL.setStyleSheet("color: black")
        self.load_status_LABEL.setAlignment(QtCore.Qt.AlignCenter)

        self.load_status_LABEL.setFont(self.medium_bold_FONT)
        layoutLoadSim =  QtWidgets.QHBoxLayout()
        layoutLoadSim.addWidget(self.load_sim_BUTTON)
        layoutLoadSim.addWidget(self.load_status_LABEL)

        # self.sim_geometry_BOX = QtWidgets.QComboBox()
        # self.sim_geometry_BOX.addItem("3D")
        # self.sim_geometry_BOX.addItem("AMC")
        # self.sim_geometry_BOX.addItem("2D")
        # self.sim_geometry_BOX.addItem("1D")
        # layoutSimGeometry = self.creatPara("Geometry :",self.sim_geometry_BOX)


        # boxLayout_settings.addLayout(layoutSimulationBasePath)
        boxLayout_settings.addLayout(layoutLoadSim)
        boxLayout_settings.addWidget(self.sim_directory_name_LABEL)

        # boxLayout_settings.addLayout(layoutSimGeometry)
        self.settings_groupBox.setLayout(boxLayout_settings)

        #==============================
        # SIM INFO
        #==============================
        self.sim_info_groupBox = QtWidgets.QGroupBox("Simulation Parameters")
        self.sim_info_groupBox.setMinimumWidth(300)
        self.sim_info_groupBox.setMaximumWidth(300)
        boxLayout_sim_info = QtWidgets.QVBoxLayout()

        self.geometry_LABEL = QtWidgets.QLabel("")
        self.geometry_LABEL.setFont(self.medium_FONT)
        layoutGeometry = self.creatPara("Geometry :", self.geometry_LABEL)
        boxLayout_sim_info.addLayout(layoutGeometry)
        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))
        self.w0_LABEL = QtWidgets.QLabel("")
        self.w0_LABEL.setFont(self.medium_FONT)
        layoutW0 = self.creatPara("w0 :", self.w0_LABEL)
        self.a0_LABEL = QtWidgets.QLabel("")
        self.a0_LABEL.setFont(self.medium_FONT)
        layoutA0 = self.creatPara("a0 :", self.a0_LABEL)
        self.Tp_LABEL = QtWidgets.QLabel("")
        self.Tp_LABEL.setFont(self.medium_FONT)
        layoutTp = self.creatPara("Tp :", self.Tp_LABEL)

        self.Pola_LABEL = QtWidgets.QLabel("")
        self.Pola_LABEL.setFont(self.medium_FONT)
        layoutPola = self.creatPara("σ, l :", self.Pola_LABEL)


        boxLayout_sim_info.addLayout(layoutA0)
        boxLayout_sim_info.addLayout(layoutW0)
        boxLayout_sim_info.addLayout(layoutTp)
        boxLayout_sim_info.addLayout(layoutPola)

        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))

        self.Ltrans_LABEL = QtWidgets.QLabel("")
        self.Ltrans_LABEL.setFont(self.medium_FONT)
        layoutLtrans = self.creatPara("Ltrans :", self.Ltrans_LABEL)
        self.Llong_LABEL = QtWidgets.QLabel("")
        self.Llong_LABEL.setFont(self.medium_FONT)
        layoutLlong = self.creatPara("Llong :", self.Llong_LABEL)
        self.tsim_LABEL = QtWidgets.QLabel("")
        self.tsim_LABEL.setFont(self.medium_FONT)
        layoutTp = self.creatPara("tsim :", self.tsim_LABEL)

        boxLayout_sim_info.addLayout(layoutLtrans)
        boxLayout_sim_info.addLayout(layoutLlong)
        boxLayout_sim_info.addLayout(layoutTp)
        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))
        self.dx_LABEL = QtWidgets.QLabel("")
        self.dx_LABEL.setFont(self.medium_FONT)
        layoutDx = self.creatPara("dx :", self.dx_LABEL)
        boxLayout_sim_info.addLayout(layoutDx)

        self.mesh_LABEL = QtWidgets.QLabel("")
        self.mesh_LABEL.setFont(self.medium_FONT)
        layoutMesh = self.creatPara("Mesh :", self.mesh_LABEL)
        boxLayout_sim_info.addLayout(layoutMesh)

        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))

        plasma_param_LABEL= QtWidgets.QLabel("PLASMA PARAMETERS")
        plasma_param_LABEL.setFont(self.small_bold_FONT)
        plasma_param_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        boxLayout_sim_info.addWidget(plasma_param_LABEL)
        self.density_LABEL= QtWidgets.QLabel("")
        self.density_LABEL.setFont(self.medium_FONT)
        layoutDensity = self.creatPara("ne0 :", self.density_LABEL)
        self.nppc_LABEL = QtWidgets.QLabel("")
        self.nppc_LABEL.setFont(self.medium_FONT)
        layoutNPPC = self.creatPara("nppc :", self.nppc_LABEL)
        boxLayout_sim_info.addLayout(layoutDensity)
        boxLayout_sim_info.addLayout(layoutNPPC)

        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))

        SI_assume_LABEL= QtWidgets.QLabel("SI UNITS (𝝀 = 1 µm)")
        SI_assume_LABEL.setFont(self.small_bold_FONT)
        SI_assume_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        boxLayout_sim_info.addWidget(SI_assume_LABEL)

        self.Tp_SI_LABEL= QtWidgets.QLabel("")
        self.Tp_SI_LABEL.setFont(self.medium_FONT)
        layoutTp_SI = self.creatPara("Tp :", self.Tp_SI_LABEL)
        self.energy_SI_LABEL = QtWidgets.QLabel("")
        self.energy_SI_LABEL.setFont(self.medium_FONT)
        layoutEnergy_SI = self.creatPara("Energy :", self.energy_SI_LABEL)
        self.power_SI_LABEL = QtWidgets.QLabel("")
        self.power_SI_LABEL.setFont(self.medium_FONT)
        layoutPower_SI = self.creatPara("Power :", self.power_SI_LABEL)
        self.intensity_SI_LABEL = QtWidgets.QLabel("")
        self.intensity_SI_LABEL.setFont(self.medium_FONT)
        layoutIntensity_SI = self.creatPara("Intensity :", self.intensity_SI_LABEL)
        boxLayout_sim_info.addLayout(layoutTp_SI)
        boxLayout_sim_info.addLayout(layoutEnergy_SI)
        boxLayout_sim_info.addLayout(layoutPower_SI)
        boxLayout_sim_info.addLayout(layoutIntensity_SI)


        verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        boxLayout_sim_info.addItem(verticalSpacer)
        self.sim_info_groupBox.setLayout(boxLayout_sim_info)

        boxLayoutLEFT = QtWidgets.QVBoxLayout()
        boxLayoutLEFT.addWidget(self.settings_groupBox)
        boxLayoutLEFT.addWidget(self.sim_info_groupBox)
        boxLayoutLEFT.setContentsMargins(2,0,0,0)



        #==============================
        # TAB WIDGET
        #==============================
        #--------------------
        # TAB 0
        #--------------------
        self.figure_0 = Figure()
        self.canvas_0 = FigureCanvas(self.figure_0)
        self.plt_toolbar_0 = NavigationToolbar(self.canvas_0)
        self.plt_toolbar_0.setFixedHeight(35)

        fontsize = 12
        self.Utot_CHECK = QtWidgets.QCheckBox()
        layoutUtot = self.creatPara("Utot ", self.Utot_CHECK,adjust_label=True,fontsize=fontsize)
        self.Uelm_CHECK = QtWidgets.QCheckBox()
        layoutUelm = self.creatPara("Uelm ", self.Uelm_CHECK,adjust_label=True,fontsize=fontsize)
        self.Ukin_CHECK = QtWidgets.QCheckBox()
        layoutUkin = self.creatPara("Ukin ", self.Ukin_CHECK,adjust_label=True,fontsize=fontsize)
        self.AM_CHECK = QtWidgets.QCheckBox()
        layoutAM = self.creatPara("AM ", self.AM_CHECK,adjust_label=True,fontsize=fontsize)

        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        layoutTabSettingsCheck.addLayout(layoutUtot)
        layoutTabSettingsCheck.addLayout(layoutUelm)
        layoutTabSettingsCheck.addLayout(layoutUkin)
        layoutTabSettingsCheck.addLayout(layoutAM)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addWidget(self.plt_toolbar_0)


        self.scalar_groupBox = QtWidgets.QGroupBox("Scalar Diagnostics")
        self.scalar_groupBox.setMinimumHeight(210)
        self.scalar_groupBox.setMaximumHeight(210)
        self.scalar_groupBox.setLayout(layoutTabSettings)

        self.layoutScalar = QtWidgets.QVBoxLayout()
        self.layoutScalar.addWidget(self.scalar_groupBox)
        self.layoutScalar.addWidget(self.canvas_0)

        self.scalar_Widget = QtWidgets.QWidget()
        self.scalar_Widget.setLayout(self.layoutScalar)

        #--------------------
        # TAB 1
        #--------------------
        self.figure_1 = Figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.plt_toolbar_1 = NavigationToolbar(self.canvas_1, self)
        self.plt_toolbar_1.setFixedHeight(35)

        fontsize = 12
        self.Ex_CHECK = QtWidgets.QCheckBox()
        layoutEx = self.creatPara("Ex ", self.Ex_CHECK,adjust_label=True,fontsize=fontsize)
        self.Ey_CHECK = QtWidgets.QCheckBox()
        layoutEy = self.creatPara("Ey ", self.Ey_CHECK,adjust_label=True,fontsize=fontsize)
        self.Ez_CHECK = QtWidgets.QCheckBox()
        layoutEz = self.creatPara("Ez ", self.Ez_CHECK,adjust_label=True,fontsize=fontsize)
        self.Bx_CHECK = QtWidgets.QCheckBox()
        layoutBx = self.creatPara("Bx ", self.Bx_CHECK,adjust_label=True,fontsize=fontsize)
        self.By_CHECK = QtWidgets.QCheckBox()
        layoutBy = self.creatPara("By ", self.By_CHECK,adjust_label=True,fontsize=fontsize)
        self.Bz_CHECK = QtWidgets.QCheckBox()
        layoutBz = self.creatPara("Bz ", self.Bz_CHECK,adjust_label=True,fontsize=fontsize)

        self.Er_CHECK = QtWidgets.QCheckBox()
        layoutEr = self.creatPara("Er ", self.Er_CHECK,adjust_label=True,fontsize=fontsize)
        self.Etheta_CHECK = QtWidgets.QCheckBox()
        layoutEtheta = self.creatPara("Eθ ", self.Etheta_CHECK,adjust_label=True,fontsize=fontsize)
        self.Br_CHECK = QtWidgets.QCheckBox()
        layoutBr = self.creatPara("Br ", self.Br_CHECK,adjust_label=True,fontsize=fontsize)
        self.Btheta_CHECK = QtWidgets.QCheckBox()
        layoutBtheta = self.creatPara("Bθ ", self.Btheta_CHECK,adjust_label=True,fontsize=fontsize)

        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.VLine)
        separator1.setLineWidth(1)

        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.VLine)
        separator2.setLineWidth(1)

        layoutEx.setSpacing(0)
        layoutEy.setSpacing(0)
        layoutEz.setSpacing(0)
        layoutBy.setSpacing(0)
        layoutBx.setSpacing(0)
        layoutBz.setSpacing(0)
        layoutEr.setSpacing(0)
        layoutEtheta.setSpacing(0)
        layoutBr.setSpacing(0)
        layoutBtheta.setSpacing(0)
        self.Ex_CHECK.setChecked(True)
        # self.Ey_CHECK.setChecked(True)
        self.Ez_CHECK.setChecked(True)

        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        layoutTabSettingsCheck.addLayout(layoutEx)
        layoutTabSettingsCheck.addLayout(layoutEy)
        layoutTabSettingsCheck.addLayout(layoutEz)
        layoutTabSettingsCheck.addWidget(separator1)
        layoutTabSettingsCheck.addLayout(layoutBx)
        layoutTabSettingsCheck.addLayout(layoutBy)
        layoutTabSettingsCheck.addLayout(layoutBz)
        layoutTabSettingsCheck.addWidget(separator2)
        layoutTabSettingsCheck.addLayout(layoutEr)
        layoutTabSettingsCheck.addLayout(layoutEtheta)
        layoutTabSettingsCheck.addLayout(layoutBr)
        layoutTabSettingsCheck.addLayout(layoutBtheta)

        layoutTabSettingsCheck.setSpacing(20)
        # layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)

        self.sim_cut_direction_BOX = QtWidgets.QComboBox()
        self.sim_cut_direction_BOX.addItem("Longitudinal cut")
        self.sim_cut_direction_BOX.addItem("Transverse cut")
        layoutTabSettingsCutDirection = QtWidgets.QHBoxLayout()
        layoutTabSettingsCutDirection.addWidget(self.sim_cut_direction_BOX)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addLayout(layoutTabSettingsCutDirection)


        self.fields_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fields_time_SLIDER.setMaximum(1)
        self.fields_time_SLIDER.setMinimum(0)
        self.fields_zcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fields_zcut_SLIDER.setMaximum(1)
        self.fields_zcut_SLIDER.setMinimum(0)

        self.fields_time_EDIT = QtWidgets.QLineEdit("0")

        self.fields_time_EDIT.setValidator(self.float_validator)
        self.fields_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.fields_zcut_EDIT = QtWidgets.QLineEdit("0")
        self.fields_zcut_EDIT.setValidator(self.float_validator)
        self.fields_zcut_EDIT.setMaximumWidth(70)

        self.fields_play_time_BUTTON = QtWidgets.QPushButton("Play")
        self.fields_play_time_BUTTON.setMinimumHeight(15)
        layoutTimeSlider = self.creatPara("t/t0=", self.fields_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.fields_time_SLIDER)
        layoutTimeSlider.addWidget(self.fields_play_time_BUTTON)


        layoutZcutSlider = self.creatPara("z/𝝀=", self.fields_zcut_EDIT)
        layoutZcutSlider.addWidget(self.fields_zcut_SLIDER)

        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addLayout(layoutZcutSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_1)

        self.fields_groupBox = QtWidgets.QGroupBox("Fields Diagnostics")
        self.fields_groupBox.setMinimumHeight(210)
        self.fields_groupBox.setMaximumHeight(210)
        self.fields_groupBox.setLayout(layoutTabSettings)

        self.layoutFields = QtWidgets.QVBoxLayout()
        self.layoutFields.addWidget(self.fields_groupBox)
        self.layoutFields.addWidget(self.canvas_1)

        self.fields_Widget = QtWidgets.QWidget()
        self.fields_Widget.setLayout(self.layoutFields)

        #--------------------
        # TAB 2 TRACK
        #--------------------
        self.figure_2 = Figure()
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.plt_toolbar_2 = NavigationToolbar(self.canvas_2, self)
        self.plt_toolbar_2.setFixedHeight(35)

        self.track_file_BOX = QtWidgets.QComboBox()
        self.track_file_BOX.addItem("track_eon")
        self.track_file_BOX.addItem("track_eon_full")

        layoutTabSettingsTrackFile = QtWidgets.QHBoxLayout()
        self.track_Npart_EDIT = QtWidgets.QLineEdit("10")
        self.track_Npart_EDIT.setValidator(self.int_validator)
        self.track_Npart_EDIT.setMaximumWidth(35)
        layoutNpart = self.creatPara("Npart=", self.track_Npart_EDIT,adjust_label=True)

        layoutTabSettingsTrackFile.addLayout(layoutNpart)
        layoutTabSettingsTrackFile.addWidget(self.track_file_BOX)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsTrackFile)

        self.track_time_EDIT = QtWidgets.QLineEdit("0")
        self.track_time_EDIT.setValidator(self.float_validator)
        self.track_time_EDIT.setMaximumWidth(70)
        self.track_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.track_time_SLIDER.setMaximum(1)
        self.track_time_SLIDER.setMinimum(0)
        self.track_play_time_BUTTON = QtWidgets.QPushButton("Play")

        layoutTimeSlider = self.creatPara("t/t0=", self.track_time_EDIT,adjust_label=True)
        layoutTimeSlider.addWidget(self.track_time_SLIDER)
        layoutTimeSlider.addWidget(self.track_play_time_BUTTON)
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_2)

        self.track_groupBox = QtWidgets.QGroupBox("Track Particles Diagnostic")
        self.track_groupBox.setMinimumHeight(120)
        self.track_groupBox.setMaximumHeight(150)
        self.track_groupBox.setLayout(layoutTabSettings)

        self.layoutTrack = QtWidgets.QVBoxLayout()
        self.layoutTrack.addWidget(self.track_groupBox)
        self.layoutTrack.addWidget(self.canvas_2)

        self.track_Widget = QtWidgets.QWidget()
        self.track_Widget.setLayout(self.layoutTrack)

        #--------------------
        # TAB 3 PLASMA
        #--------------------
        self.figure_3 = Figure()
        self.canvas_3 = FigureCanvas(self.figure_3)
        self.plt_toolbar_3 = NavigationToolbar(self.canvas_3, self)
        self.plt_toolbar_3.setFixedHeight(35)

        self.plasma_Bz_CHECK = QtWidgets.QCheckBox()
        layoutBz = self.creatPara("Bz ", self.plasma_Bz_CHECK,adjust_label=True,fontsize=12)
        self.plasma_Bz_trans_CHECK = QtWidgets.QCheckBox()
        layoutBzTrans = self.creatPara("Bz_trans ", self.plasma_Bz_trans_CHECK,adjust_label=True,fontsize=12)
        self.plasma_ne_CHECK = QtWidgets.QCheckBox()
        layoutNe = self.creatPara("ne ", self.plasma_ne_CHECK,adjust_label=True,fontsize=12)
        self.plasma_ne_trans_CHECK = QtWidgets.QCheckBox()
        layoutNeTrans = self.creatPara("ne_trans ", self.plasma_ne_trans_CHECK,adjust_label=True,fontsize=12)

        self.plasma_Lz_CHECK = QtWidgets.QCheckBox()
        layoutLz = self.creatPara("Lz ", self.plasma_Lz_CHECK,adjust_label=True,fontsize=12)
        self.plasma_Lz_trans_CHECK = QtWidgets.QCheckBox()
        layoutLzTrans = self.creatPara("Lz_trans ", self.plasma_Lz_trans_CHECK,adjust_label=True,fontsize=12)

        self.plasma_ptheta_CHECK = QtWidgets.QCheckBox()
        layoutPtheta = self.creatPara("ptheta ", self.plasma_ptheta_CHECK,adjust_label=True,fontsize=12)
        self.plasma_ptheta_trans_CHECK = QtWidgets.QCheckBox()
        layoutPthetaTrans = self.creatPara("ptheta_trans ", self.plasma_ptheta_trans_CHECK,adjust_label=True,fontsize=12)

        self.plasma_Jtheta_CHECK = QtWidgets.QCheckBox()
        layoutJtheta = self.creatPara("Jtheta ", self.plasma_Jtheta_CHECK,adjust_label=True,fontsize=12)
        self.plasma_Jtheta_trans_CHECK = QtWidgets.QCheckBox()
        layoutJthetaTrans = self.creatPara("Jtheta_trans ", self.plasma_Jtheta_trans_CHECK,adjust_label=True,fontsize=12)

        separator1 = QtWidgets.QFrame()
        separator1.setFrameShape(QtWidgets.QFrame.VLine)
        separator1.setLineWidth(1)

        separator2 = QtWidgets.QFrame()
        separator2.setFrameShape(QtWidgets.QFrame.VLine)
        separator2.setLineWidth(1)

        separator3 = QtWidgets.QFrame()
        separator3.setFrameShape(QtWidgets.QFrame.VLine)
        separator3.setLineWidth(1)

        separator4 = QtWidgets.QFrame()
        separator4.setFrameShape(QtWidgets.QFrame.VLine)
        separator4.setLineWidth(1)

        layoutBz.setSpacing(0)
        layoutBzTrans.setSpacing(0)
        layoutNe.setSpacing(0)
        layoutNeTrans.setSpacing(0)

        self.plasma_Bz_CHECK.setChecked(True)
        # self.Ey_CHECK.setChecked(True)
        self.plasma_Bz_trans_CHECK.setChecked(True)

        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        layoutTabSettingsCheck.addLayout(layoutBz)
        layoutTabSettingsCheck.addLayout(layoutBzTrans)
        layoutTabSettingsCheck.addWidget(separator1)
        layoutTabSettingsCheck.addLayout(layoutNe)
        layoutTabSettingsCheck.addLayout(layoutNeTrans)
        layoutTabSettingsCheck.addWidget(separator2)
        layoutTabSettingsCheck.addLayout(layoutLz)
        layoutTabSettingsCheck.addLayout(layoutLzTrans)
        layoutTabSettingsCheck.addWidget(separator3)
        layoutTabSettingsCheck.addLayout(layoutPtheta)
        layoutTabSettingsCheck.addLayout(layoutPthetaTrans)
        layoutTabSettingsCheck.addWidget(separator4)
        layoutTabSettingsCheck.addLayout(layoutJtheta)
        layoutTabSettingsCheck.addLayout(layoutJthetaTrans)
        layoutTabSettingsCheck.setSpacing(4)

        self.plasma_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plasma_time_SLIDER.setMaximum(1)
        self.plasma_time_SLIDER.setMinimum(0)
        self.plasma_zcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plasma_zcut_SLIDER.setMaximum(1)
        self.plasma_zcut_SLIDER.setMinimum(0)

        self.plasma_time_EDIT = QtWidgets.QLineEdit("0")

        self.plasma_time_EDIT.setValidator(self.float_validator)
        self.plasma_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.plasma_zcut_EDIT = QtWidgets.QLineEdit("0")
        self.plasma_zcut_EDIT.setValidator(self.float_validator)
        self.plasma_zcut_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.plasma_play_time_BUTTON = QtWidgets.QPushButton("Play")
        layoutTimeSlider = self.creatPara("t/t0=", self.plasma_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.plasma_time_SLIDER)
        layoutTimeSlider.addWidget(self.plasma_play_time_BUTTON)

        layoutZcutSlider = self.creatPara("z/𝝀=", self.plasma_zcut_EDIT)
        layoutZcutSlider.addWidget(self.plasma_zcut_SLIDER)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addLayout(layoutZcutSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_3)

        self.plasma_groupBox = QtWidgets.QGroupBox("Fields Diagnostics")
        self.plasma_groupBox.setMinimumHeight(210)
        self.plasma_groupBox.setMaximumHeight(210)
        self.plasma_groupBox.setLayout(layoutTabSettings)

        self.layoutPlasma = QtWidgets.QVBoxLayout()
        self.layoutPlasma.addWidget(self.plasma_groupBox)
        self.layoutPlasma.addWidget(self.canvas_3)

        self.plasma_Widget = QtWidgets.QWidget()
        self.plasma_Widget.setLayout(self.layoutPlasma)

        #--------------------
        # TAB 4 TORNADO
        #--------------------
        self.tornado_Widget = QtWidgets.QWidget()

        self.layoutTornado = QtWidgets.QVBoxLayout()
        self.tornado_groupBox = QtWidgets.QGroupBox("Infos")
        self.tornado_groupBox.setMinimumHeight(75)
        self.tornado_groupBox.setMaximumHeight(80)


        tornado_group_box_layout = QtWidgets.QHBoxLayout()


        self.tornado_last_update_LABEL = QtWidgets.QLabel("LOADING...")
        self.tornado_last_update_LABEL.setFont(QFont('Arial', 12))
        tornado_group_box_layout.addWidget(self.tornado_last_update_LABEL)


        self.tornado_refresh_BUTTON = QtWidgets.QPushButton("Refresh")
        tornado_group_box_layout.addWidget(self.tornado_refresh_BUTTON)


        self.tornado_groupBox.setLayout(tornado_group_box_layout)
        self.layoutTornado.addWidget(self.tornado_groupBox)

        self.tornado_Widget.setLayout(self.layoutTornado)


        #--------------------
        # ADD TABS TO QTabWidget
        #--------------------
        self.programm_TABS = QtWidgets.QTabWidget()

        self.programm_TABS.setTabsClosable(True)
        # self.track_Widget.setTabsClosable(True)

        # self.programm_TABS.addTab(self.fields_Widget,"FIELDS")
        # self.programm_TABS.addTab(self.track_Widget,"TRACK")
        # self.programm_TABS.addTab(self.plasma_Widget,"PLASMA")
        # self.programm_TABS.addTab(self.tornado_Widget,"TORNADO")



        layoutTabsAndLeft = QtWidgets.QHBoxLayout()
        layoutTabsAndLeft.addLayout(boxLayoutLEFT)
        layoutTabsAndLeft.addWidget(self.programm_TABS)

        layoutBottom = QtWidgets.QHBoxLayout()
        layoutBottom.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)


        self.general_info_LABEL = QtWidgets.QLabel(f"Version: {self.SCRIPT_VERSION}  |  Memory: {self.MEMORY().used*100/self.MEMORY().total:.0f}% | {self.COPY_RIGHT}")
        # self.general_info_LABEL.setStyleSheet('border: 1px solid black')
        # self.general_info_LABEL.adjustSize()
        # self.general_info_LABEL.setFixedSize(QtCore.QSize(250, 15))
        # self.general_info_LABEL.setMaximumSize(QtCore.QSize(250, 13))

        layoutMAIN = QtWidgets.QVBoxLayout()
        layoutMAIN.addLayout(layoutTabsAndLeft)
        layoutMAIN.addWidget(self.general_info_LABEL,alignment=QtCore.Qt.AlignRight)

        layoutMAIN.setContentsMargins(0, 0, 0, 0)

        widget = QtWidgets.QWidget()
        widget.setLayout(layoutMAIN)
        self.setCentralWidget(widget)

        #====================
        # CONNECTS
        #====================

        # self.canvas.mpl_connect('scroll_event', self.onScroll)
        # self.iterMax.valueChanged.connect(self.plot)
        # self.coeflog.valueChanged.connect(self.plot)
        # self.load_sim_BUTTON.clicked.connect(self.onLoadSim)
        self.load_sim_BUTTON.clicked.connect(self.onOpenSim)

        self.Utot_CHECK.clicked.connect(lambda: self.onUpdateTabScalar(0))
        self.Uelm_CHECK.clicked.connect(lambda: self.onUpdateTabScalar(1))
        self.Ukin_CHECK.clicked.connect(lambda: self.onUpdateTabScalar(2))
        self.AM_CHECK.clicked.connect(lambda: self.onUpdateTabScalar(3))

        self.Ex_CHECK.clicked.connect(lambda: self.onUpdateTabFields(0))
        self.Ey_CHECK.clicked.connect(lambda: self.onUpdateTabFields(1))
        self.Ez_CHECK.clicked.connect(lambda: self.onUpdateTabFields(2))
        self.Bx_CHECK.clicked.connect(lambda: self.onUpdateTabFields(3))
        self.By_CHECK.clicked.connect(lambda: self.onUpdateTabFields(4))
        self.Bz_CHECK.clicked.connect(lambda: self.onUpdateTabFields(5))
        self.Er_CHECK.clicked.connect(lambda: self.onUpdateTabFields(6))
        self.Etheta_CHECK.clicked.connect(lambda: self.onUpdateTabFields(7))
        self.Br_CHECK.clicked.connect(lambda: self.onUpdateTabFields(8))
        self.Btheta_CHECK.clicked.connect(lambda: self.onUpdateTabFields(9))
        self.fields_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabFields(100))
        self.fields_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabFields(100))
        self.fields_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabFields(101))
        self.fields_zcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabFields(200))
        self.fields_zcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabFields(200))
        self.fields_zcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabFields(201))

        self.fields_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabFields(1000))
        self.sim_cut_direction_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabFields(-100))

        self.track_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabTrack(100))
        self.track_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabTrack(100))
        self.track_time_EDIT.returnPressed.connect(lambda:  self.onUpdateTabTrack(101))
        self.track_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabTrack(1000))
        self.track_file_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabTrack(-1))
        self.track_Npart_EDIT.returnPressed.connect(lambda:  self.onUpdateTabTrack(-1))


        self.plasma_Bz_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(0))
        self.plasma_Bz_trans_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(1))
        self.plasma_ne_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(2))
        self.plasma_ne_trans_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(3))
        self.plasma_Lz_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(4))
        self.plasma_Lz_trans_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(5))
        self.plasma_ptheta_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(6))
        self.plasma_ptheta_trans_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(7))
        self.plasma_Jtheta_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(8))
        self.plasma_Jtheta_trans_CHECK.clicked.connect(lambda: self.onUpdateTabPlasma(9))


        self.plasma_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabPlasma(100))
        self.plasma_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabPlasma(100))
        self.plasma_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabPlasma(101))
        self.plasma_zcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabPlasma(200))
        self.plasma_zcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabPlasma(200))
        self.plasma_zcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabPlasma(201))
        self.plasma_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabPlasma(1000))

        self.tornado_refresh_BUTTON.clicked.connect(self.call_ThreadDownloadSimJSON)

        #Open and Close Tabs
        self.actionDiagScalar.toggled.connect(lambda: self.onMenuTabs("SCALAR"))
        self.actionDiagFields.toggled.connect(lambda: self.onMenuTabs("FIELDS"))
        self.actionDiagTrack.toggled.connect(lambda: self.onMenuTabs("TRACK"))
        self.actionDiagPlasma.toggled.connect(lambda: self.onMenuTabs("PLASMA"))
        self.actionTornado.toggled.connect(lambda: self.onMenuTabs("TORNADO"))
        self.programm_TABS.tabCloseRequested.connect(self.onCloseTab)


        self.actionOpenSim.triggered.connect(self.onOpenSim)
        self.actionOpenLogs.triggered.connect(self.onOpenLogs)
        self.actionOpenIPython.triggered.connect(self.onOpenIPython)


        self.memory_update_TIMER = QtCore.QTimer()
        self.memory_update_TIMER.setInterval(5000) #in ms
        self.memory_update_TIMER.timeout.connect(self.updateInfoLabelMem)
        self.memory_update_TIMER.start()



        # self.reset.clicked.connect(self.onReset)
        # self.interpolation.currentIndexChanged.connect(self.plot)


        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle("Smilei IFE GUI")


        # C:\_DOSSIERS_PC\_STAGE_LULI_\SMILEI_QT_GUI
        self.setWindowIcon(QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\smileiIcon.png"))


        #============================
        # GENERAL VARIABLES
        #============================
        self.INIT_tabScalar = None
        self.INIT_tabFields = None
        self.INIT_tabTrack = None
        self.INIT_tabPlasma = None
        self.INIT_tabTornado = True

        self.loop_in_process = False
        self.is_sim_loaded = False
        self.logs_history_STR = None

        self.l0 = 2*pi
        self.t0 = 2*pi

        self.timer = time.perf_counter()
        #====================
        # CALL INIT METHODS
        #====================

    def deleteItemsOfLayout(self, layout):
        """Remove all sub widgets and layout from main layout"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                else:
                    deleteItemsOfLayout(item.layout())
        return

    def deleteLayout(self, main_layout, layout_ID_to_del):
        """Delate layout and all its components"""
        layout_item = main_layout.itemAt(layout_ID_to_del)
        #delete sub widgets from layout before deleting layout itself
        self.deleteItemsOfLayout(layout_item.layout())
        main_layout.removeItem(layout_item)
        return



    def closeEvent(self):
        sys.exit(0)
    def __del__(self):
        sys.exit(0)

    def onCloseLogs(self):
        try:
            self.logs_history_STR = self.logs_DIALOG.saveHistory()
            self.logs_DIALOG.deleteLater
            self.logs_DIALOG = None
        except:
            print("no logs to delete")
            return

    def onOpenLogs(self):
        self.onCloseLogs()
        sys.stdout = self.spyder_default_stdout
        self.logs_DIALOG = LogsDialog(self.spyder_default_stdout,app)
        if self.logs_history_STR is not None: self.logs_DIALOG.initHistory(self.logs_history_STR)
        self.logs_DIALOG.show()

    def onOpenIPython(self):
        self.ipython_DIALOG = PyDialog(self)
        self.ipython_DIALOG.show()


    def updateInfoLabelMem(self):
        # print(f"update mem: {self.MEMORY().used*100/self.MEMORY().total:.0f}%")

        mem_prc = self.MEMORY().used*100/self.MEMORY().total
        if mem_prc > 85:
            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | <font color='red'>Memory: {mem_prc:.0f}%</font> | {self.COPY_RIGHT}")
        else:
            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | Memory: {mem_prc:.0f}% | {self.COPY_RIGHT}")

    def onOpenSim(self):
        sim_file_DIALOG= QtWidgets.QFileDialog()
        sim_file_DIALOG.setDirectory(os.environ["SMILEI_CLUSTER"])

        file = str(sim_file_DIALOG.getExistingDirectory(self, "Select Directory"))

        print(f"file:{file};")
        if file !="":
            self.sim_directory_path = file
            self.onLoadSim()

    def onLoadSim(self):
        self.load_sim_status = "verifying..."
        self.load_status_LABEL.setStyleSheet("color: black")
        self.load_status_LABEL.setText(self.load_sim_status)
        app.processEvents()
        path = self.sim_directory_path
        path_list = path.split("/")
        self.sim_directory_name = path_list[-1]
        self.sim_directory_parent = path_list[-2]
        self.sim_directory_name_LABEL.setText(self.sim_directory_parent+"/"+ self.sim_directory_name)
        self.S = happi.Open(path)

        if self.S.valid:
            self.load_sim_status = "Loaded"
            self.load_status_LABEL.setStyleSheet("color: green")
        else:
            self.load_sim_status = "Invalid"
            self.load_status_LABEL.setStyleSheet("color: red")
        self.load_status_LABEL.setText(self.load_sim_status)

        l0 = 2*pi
        self.w0 = self.S.namelist.w0
        self.a0 = self.S.namelist.a0
        self.Tp = self.S.namelist.Tp
        self.dx = self.S.namelist.dz
        self.Ltrans = self.S.namelist.Ltrans
        self.Llong = self.S.namelist.Lz
        self.tsim = self.S.namelist.tsim
        self.l1 = self.S.namelist.l1
        self.eps = self.S.namelist.eps
        self.sim_geometry = self.S.namelist.Main.geometry

        # print(self.dx)
        self.geometry_LABEL.setText(f"{self.sim_geometry}")
        self.w0_LABEL.setText(f"{self.w0/l0:.1f}𝝀")
        self.a0_LABEL.setText(f"{self.a0:.2f}")
        self.Tp_LABEL.setText(f"{self.Tp/l0:.1f}𝝀")
        self.Pola_LABEL.setText(f"{self.eps}, {self.l1}")
        self.dx_LABEL.setText(f"𝝀/{l0/self.dx:.0f}")
        mesh_trans = int(self.Ltrans/self.dx)
        mesh_long = int(self.Llong/self.dx)
        self.mesh_LABEL.setText(f"{mesh_trans} x {mesh_trans} x {mesh_long}")

        self.Ltrans_LABEL.setText(f"{self.Ltrans/l0:.1f}𝝀")
        self.Llong_LABEL.setText(f"{self.Llong/l0:.1f}𝝀")
        self.tsim_LABEL.setText(f"{self.tsim/l0:.1f}𝝀")


        self.intensity_SI = (self.a0/0.85)**2 *10**18 #W/cm^2

        self.power_SI = self.intensity_SI * pi*(self.w0/l0*10**-4)**2/2

        me = 9.1093837*10**-31
        e = 1.60217663*10**-19
        c = 299792458
        eps0 = 8.854*10**-12
        wr = 2*pi*c/1e-6
        K = me*c**2
        N = eps0*me*wr**2/e**2
        L = c/wr
        KNL3 = K*N*L**3
        self.energy_SI = np.max(self.S.Scalar("Uelm").getData())*1000*KNL3
        self.Tp_SI = self.Tp/wr*10**15

        self.intensity_SI_LABEL.setText(f"{'%.1E' % decimal.Decimal(str(self.intensity_SI))} W/cm²")
        self.power_SI_LABEL.setText(f"{self.printSI(self.power_SI,'W',ndeci=2):}")
        self.energy_SI_LABEL.setText(f"{self.energy_SI:.2f} mJ")
        self.Tp_SI_LABEL.setText(f"{self.Tp_SI:.0f} fs")


        self.nppc_LABEL.setText(f"{self.S.namelist.nppc_plasma}")
        self.density_LABEL.setText(f"{self.S.namelist.ne} nc")

        self.is_sim_loaded = True
        self.INIT_tabScalar = True
        self.INIT_tabFields = True
        self.INIT_tabTrack = True
        self.INIT_tabPlasma = True

        self.updateInfoLabelMem()
        if self.actionDiagScalar.isChecked(): self.onUpdateTabScalar(0)
        if self.actionDiagFields.isChecked(): self.onUpdateTabFields(0)
        if self.actionDiagTrack.isChecked(): self.onUpdateTabTrack(0)
        if self.actionDiagPlasma.isChecked(): self.onUpdateTabPlasma(0)

    def displayLoadingLabel(self, widget_to_cover):
        self.loading_LABEL = QtWidgets.QLabel("LOADING...",widget_to_cover)
        self.loading_LABEL.setScaledContents(True)
        self.loading_LABEL.setStyleSheet("background-color:rgba(0, 0, 0, 50); border-radius: 15px;")
        self.loading_LABEL.setAlignment(QtCore.Qt.AlignCenter)

        self.loading_LABEL.setFont(self.bold_FONT)
        # widget_pos = self.fields_groupBox.mapToGlobal(QtCore.QPoint(0, 0))
        # print(widget_pos)
        tab_size = widget_to_cover.frameGeometry().width(),widget_to_cover.frameGeometry().height()
        self.loading_LABEL.resize(tab_size[0],tab_size[1]-45)
        self.loading_LABEL.move(QtCore.QPoint(0,45))
        self.loading_LABEL.raise_()
        self.loading_LABEL.show()
        app.processEvents()

    def onCloseTab(self, currentIndex):
          tab_name = self.programm_TABS.tabText(currentIndex)
          self.programm_TABS.removeTab(currentIndex)
          if tab_name=="SCALAR":
              self.actionDiagScalar.setChecked(False)
              self.onRemoveScalar()
          if tab_name=="FIELDS":
              self.actionDiagFields.setChecked(False)
              self.onRemoveFields()
          if tab_name=="TRACK":
              self.actionDiagTrack.setChecked(False)
              self.onRemoveTrack()
          if tab_name=="PLASMA":
              self.actionDiagPlasma.setChecked(False)
              self.onRemovePlasma()
          if tab_name=="TORNADO":
              self.actionTornado.setChecked(False)
              self.onRemoveTornado()

    def onMenuTabs(self, tab_name):
        self.tab2 = self.track_Widget

        if tab_name == "SCALAR":
            if not self.actionDiagScalar.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "SCALAR":
                        self.programm_TABS.removeTab(currentIndex)
                        self.onRemoveScalar()
            else:
                self.programm_TABS.addTab(self.scalar_Widget,"SCALAR")
                self.INIT_tabScalar = True
                app.processEvents()
                self.onUpdateTabScalar(0)

        if tab_name == "FIELDS":
            if not self.actionDiagFields.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "FIELDS":
                        self.programm_TABS.removeTab(currentIndex)
                        self.onRemoveFields()
            else:
                self.programm_TABS.addTab(self.fields_Widget,"FIELDS")
                self.INIT_tabFields = True
                app.processEvents()
                self.onUpdateTabFields(0)

        if tab_name == "TRACK":
            if not self.actionDiagTrack.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TRACK":
                        self.programm_TABS.removeTab(currentIndex)
                        self.onRemoveTrack()
            else:
                self.programm_TABS.addTab(self.track_Widget,"TRACK")
                if self.INIT_tabTrack != None: self.INIT_tabTrack = True
                self.INIT_tabTrack = True
                app.processEvents()
                self.onUpdateTabTrack(0)

        if tab_name == "PLASMA":
            if not self.actionDiagPlasma.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "PLASMA":
                        self.programm_TABS.removeTab(currentIndex)
                        self.onRemovePlasma()
            else:
                self.programm_TABS.addTab(self.plasma_Widget,"PLASMA")
                if self.INIT_tabPlasma != None: self.INIT_tabPlasma = True
                self.INIT_tabPlasma = True
                app.processEvents()
                self.onUpdateTabPlasma(0)

        if tab_name == "TORNADO":
            if not self.actionTornado.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TORNADO":
                        self.programm_TABS.removeTab(currentIndex)
                        self.onRemoveTornado()
            else:
                self.programm_TABS.addTab(self.tornado_Widget,"TORNADO")
                app.processEvents()
                self.onInitTabTornado()
                # self.loadthread = ThreadDownloadSimJSON("/sps3/jeremy/LULI/simulation_info.json", os.environ["SMILEI_QT"])
                # self.loadthread.finished.connect(self.onInitTabTornado)
                # self.loadthread.start()
                # self.onInitTabTornado()
        self.updateInfoLabelMem()
        return

    def onUpdateTabScalar(self, check_id):
        print(check_id)
        if self.INIT_tabScalar == None or self.is_sim_loaded == False: return
        self.INIT_tabScalar = False

        l0 = 2*pi
        check_list = [self.Utot_CHECK, self.Uelm_CHECK,self.Ukin_CHECK,self.AM_CHECK]
        boolList = [check.isChecked() for check in check_list]
        print(boolList)
        self.scalar_names = ["Utot","Uelm","Ukin","AM"]
        self.scalar_t_range = self.S.Scalar("Uelm").getTimes()

        if len(self.figure_0.axes) !=0:
            for ax in self.figure_0.axes: ax.remove()

        ax = self.figure_0.add_subplot(1,1,1)

        Utot_tot = integrate.simpson(self.S.Scalar("Utot").getData(), x = self.scalar_t_range)
        Uelm_tot = integrate.simpson(self.S.Scalar("Uelm").getData(), x = self.scalar_t_range)
        Ukin_tot = integrate.simpson(self.S.Scalar("Ukin").getData(), x = self.scalar_t_range)

        Utot_tot = np.max(self.S.Scalar("Utot").getData())
        Uelm_tot = np.max(self.S.Scalar("Uelm").getData())
        Ukin_tot = np.max(self.S.Scalar("Ukin").getData())

        AM_tot = np.NaN

        for i in range(len(self.scalar_names)):
            if boolList[i]:
                if self.scalar_names[i] != "AM":
                    im = ax.plot(self.scalar_t_range/l0, self.S.Scalar(self.scalar_names[i]).getData(),
                                            label=self.scalar_names[i])
                else:
                    fields_t_range = self.S.Probe(0,"Ex").getTimes()
                    fields_paxisX = self.S.Probe(0,"Ex").getAxis("axis1")[:,0]-self.Ltrans/2
                    fields_paxisY = self.S.Probe(0,"Ex").getAxis("axis2")[:,1]-self.Ltrans/2

                    fields_paxisZ = self.S.Probe(0,"Ex").getAxis("axis3")[:,2]
                    X,Y,Z = np.meshgrid(fields_paxisX,fields_paxisY,fields_paxisZ,indexing="ij")

                    Ex = np.array(self.S.Probe(0,"Ex").getData()).astype(np.float32)
                    Ey = np.array(self.S.Probe(0,"Ey").getData()).astype(np.float32)
                    Ez = np.array(self.S.Probe(0,"Ez").getData()).astype(np.float32)
                    Bx = np.array(self.S.Probe(0,"Bx").getData()).astype(np.float32)
                    By = np.array(self.S.Probe(0,"By").getData()).astype(np.float32)
                    Bz = np.array(self.S.Probe(0,"Bz").getData()).astype(np.float32)

                    print(Ex.shape)
                    AM = X*(Ez*Bx-Ex*Bz)-Y*(Ey*Bz-Ez*By)
                    print(AM.shape)

                    AM_trans_int = integrate.simpson(integrate.simpson(AM[:,:,:,:],x=fields_paxisX,axis=1),x=fields_paxisY,axis=1)
                    print(AM_trans_int.shape)
                    AM_full_int = integrate.simpson(AM_trans_int,x=fields_paxisZ,axis=1)
                    AM_tot = np.max(AM_full_int)

                    print(AM_full_int.shape)

                    im = ax.plot(fields_t_range/l0, AM_full_int,label=self.scalar_names[i])

        ax.grid()
        ax.legend(fontsize=14)
        ax.set_xlabel("t/t0",fontsize=14)

        me = 9.1093837*10**-31
        e = 1.60217663*10**-19
        c = 299792458
        eps0 = 8.854*10**-12
        wr = 2*pi*c/1e-6
        K = me*c**2
        N = eps0*me*wr**2/e**2
        L = c/wr
        KNL3 = K*N*L**3

        # print("LASER ENERGY:", Uelm*KNL3*1000,"mJ")

        self.figure_0.suptitle(f"Scalar time plot \nUtot={Utot_tot*KNL3*1000:.2f} mJ; Uelm={Uelm_tot*KNL3*1000:.2f} mJ; Ukin={Ukin_tot*KNL3*1000:.2f} mJ; AM/U={AM_tot/Uelm_tot:.2f}",fontsize=14)
        self.figure_0.tight_layout()
        self.canvas_0.draw()


        return
    def onUpdateTabFieldsFigure(self, fields_data_list):

        #=====================================
        # REMOVE ALL FIGURES --> NOT OPTIMAL
        #=====================================
        if len(self.figure_1.axes) !=0:
            for ax in self.figure_1.axes: ax.remove()

        self.fields_data_list = fields_data_list
        self.fields_image_list = []

        check_list = [self.Ex_CHECK,self.Ey_CHECK,self.Ez_CHECK,self.Bx_CHECK,self.By_CHECK,self.Bz_CHECK,
                      self.Er_CHECK,self.Etheta_CHECK,self.Br_CHECK,self.Btheta_CHECK]
        boolList = [check.isChecked() for check in check_list]
        combo_box_index = self.sim_cut_direction_BOX.currentIndex()

        Naxis = min(sum(boolList),3)

        if Naxis != len(self.fields_data_list):
            # self.loading_LABEL.deleteLater()
            return

        time_idx = self.fields_time_SLIDER.sliderPosition()
        z_idx = self.fields_zcut_SLIDER.sliderPosition()


        t1 = time.perf_counter()
        k=0
        print("--------------")
        for i in range(len(self.fields_names)):
            if boolList[i]:
                if combo_box_index==0:
                    print(len(self.fields_data_list))
                    ax = self.figure_1.add_subplot(Naxis,1,k+1)
                    im = ax.imshow(self.fields_data_list[k][time_idx,:,self.fields_trans_mid_idx,:],cmap="RdBu", aspect="auto",
                                   extent=self.extentZX,origin='lower', interpolation="spline16")
                    ax.set_title(self.fields_names[i],rotation='vertical',x=-0.1,y=0.48)
                else:
                    ax = self.figure_1.add_subplot(1,Naxis,k+1)
                    im = ax.imshow(fields_data_list[k][time_idx,:,:,z_idx],cmap="RdBu", aspect="auto",
                                   extent=self.extentXY,origin='lower', interpolation="spline16")
                    ax.set_title(self.fields_names[i])
                im.autoscale()
                self.figure_1.colorbar(im, ax=ax,pad=0.01)
                self.fields_image_list.append(im)
                k+=1
        t2 = time.perf_counter()
        print("plot field",(t2-t1)*1000,"ms")

        byte_size_track = getsizeof(self.fields_data_list)+getsizeof(self.fields_image_list)
        print("Memory from FIELDS:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/psutil.virtual_memory().total,1),"%)")
        if combo_box_index==0:
            self.figure_1.suptitle(f"t={self.fields_t_range[time_idx]/self.l0:.2f}$~t_0$")
        else:
            self.figure_1.suptitle(f"$t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $z={self.fields_paxisZ[z_idx]/self.l0:.2f}~\lambda$")
        self.figure_1.tight_layout()
        self.figure_1.tight_layout()
        self.canvas_1.draw()

        # self.loading_LABEL.deleteLater()


    def onUpdateTabFields(self,check_id):
        if self.INIT_tabFields == None or self.is_sim_loaded == False: return
        if self.INIT_tabFields:
            print("===== INIT FIELDS TAB =====")
            # self.displayLoadingLabel(self.fields_groupBox)
            Ex_diag = self.S.Probe(0,"Ex")
            l0 = 2*pi
            # fields_shape = np.array(self.S.Probe(0,"Ex").getData()).astype(np.float32).shape

            self.fields_paxisX = Ex_diag.getAxis("axis1")[:,0]-self.Ltrans/2
            self.fields_paxisY = Ex_diag.getAxis("axis2")[:,1]-self.Ltrans/2
            self.fields_paxisZ = Ex_diag.getAxis("axis3")[:,2]
            self.extentXY = [self.fields_paxisX[0]/self.l0,self.fields_paxisX[-1]/self.l0,self.fields_paxisY[0]/self.l0,self.fields_paxisY[-1]/self.l0]
            self.extentZX = [self.fields_paxisZ[0]/self.l0,self.fields_paxisZ[-1]/self.l0,self.fields_paxisX[0]/self.l0,self.fields_paxisX[-1]/self.l0]
            self.fields_t_range = Ex_diag.getTimes()

            del Ex_diag

            self.fields_trans_mid_idx = len(self.fields_paxisY)//2
            self.fields_long_mid_idx = len(self.fields_paxisZ)//2
            self.fields_time_SLIDER.setMaximum(len(self.fields_t_range)-1)
            self.fields_zcut_SLIDER.setMaximum(len(self.fields_paxisZ)-1)

            self.fields_image_list = []
            self.fields_data_list = []
            self.fields_names =["Ex","Ey","Ez","Bx","By","Bz","Er","Eθ","Br","Bθ"]


            self.fields_time_SLIDER.setValue(len(self.fields_t_range))
            self.fields_previous_zcut_SLIDER_value = self.fields_zcut_SLIDER.sliderPosition()

            self.fields_time_EDIT.setText(str(round(self.fields_t_range[-1]/l0,2)))
            self.fields_zcut_EDIT.setText(str(round(self.fields_paxisZ[-1]/l0,2)))


            byte_size_track = getsizeof(self.fields_paxisX)+getsizeof(self.fields_paxisY)+getsizeof(self.fields_paxisZ)
            print("Memory from FIELDS:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/self.MEMORY().total,1),"%)")

            # self.loading_LABEL.deleteLater()
            self.INIT_tabFields = False
            app.processEvents()
            self.updateInfoLabelMem()

        l0 = 2*pi
        if check_id < 10: #CHECK_BOX UPDATE
            # self.displayLoadingLabel(self.fields_groupBox)
            # print("BOX UPDATE < 10")

            check_list = [self.Ex_CHECK,self.Ey_CHECK,self.Ez_CHECK,self.Bx_CHECK,self.By_CHECK,self.Bz_CHECK,
                          self.Er_CHECK,self.Etheta_CHECK,self.Br_CHECK,self.Btheta_CHECK]
            boolList = [check.isChecked() for check in check_list]

            if sum(boolList)>3:
                check_list[check_id].setChecked(False)
                boolList[check_id] = False
                # self.loading_LABEL.deleteLater()
                return

            combo_box_index = self.sim_cut_direction_BOX.currentIndex()

            self.fields_image_list = []
            self.fields_data_list = []
            # print(len(self.fields_image_list))

            self.loadthread = ThreadGetFieldsProbeData(boolList, self.fields_names, self.S, self.fields_t_range, self.fields_paxisX, self.fields_paxisY, self.fields_paxisZ)
            self.loadthread.finished.connect(self.onUpdateTabFieldsFigure)
            self.loadthread.start()


        elif check_id==200 and self.sim_cut_direction_BOX.currentIndex()==0:
            self.fields_zcut_SLIDER.setValue(self.fields_previous_zcut_SLIDER_value) #cannot change z slider if not in Transverse mode
            return

        elif check_id <= 110 or ((check_id==200 or check_id==201) and self.sim_cut_direction_BOX.currentIndex()==1): #SLIDER UPDATE
            print((time.perf_counter()-self.timer)*1000,"ms")
            self.timer = time.perf_counter()
            if check_id == 101:
                time_edit_value = float(self.fields_time_EDIT.text())
                time_idx = np.where(abs(self.fields_t_range/l0-time_edit_value)==np.min(abs(self.fields_t_range/l0-time_edit_value)))[0][0]
                self.fields_time_SLIDER.setValue(time_idx)
                self.fields_time_EDIT.setText(str(round(self.fields_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.fields_time_SLIDER.sliderPosition()
                self.fields_time_EDIT.setText(str(round(self.fields_t_range[time_idx]/l0,2)))

            if check_id == 201:
                zcut_edit_value = float(self.fields_zcut_EDIT.text())
                zcut_idx = np.where(abs(self.fields_paxisZ/l0-zcut_edit_value)==np.min(abs(self.fields_paxisZ/l0-zcut_edit_value)))[0][0]
                self.fields_zcut_SLIDER.setValue(zcut_idx)
            else:
                zcut_idx = self.fields_zcut_SLIDER.sliderPosition()
                self.fields_zcut_EDIT.setText(str(round(self.fields_paxisZ[zcut_idx]/l0,2)))

            self.fields_previous_zcut_SLIDER_value = self.fields_zcut_SLIDER.sliderPosition()
            combo_box_index = self.sim_cut_direction_BOX.currentIndex()

            if combo_box_index==0:
                for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,:,self.fields_trans_mid_idx,:])
                        self.figure_1.suptitle(f"t={self.fields_t_range[time_idx]/self.l0:.2f}$~t_0$")
                        im.autoscale()
            else:
                for i,im in enumerate(self.fields_image_list):
                    im.set_data(self.fields_data_list[i][time_idx,:,:,zcut_idx])
                    im.autoscale()
                    self.figure_1.suptitle(f"$t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $z={self.fields_paxisZ[zcut_idx]/l0:.2f}~\lambda$")
            self.canvas_1.draw()

        elif check_id == 1000:

            if self.loop_in_process: return

            self.loop_in_process = True

            combo_box_index = self.sim_cut_direction_BOX.currentIndex()
            zcut_idx = self.fields_zcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.fields_t_range)):
                self.fields_time_SLIDER.setValue(time_idx)
                if combo_box_index==0:
                    for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,:,self.fields_trans_mid_idx,:])
                        self.figure_1.suptitle(f"t={self.fields_t_range[time_idx]/self.l0:.2f}$~t_0$")
                else:
                    for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,:,:,zcut_idx])
                        im.autoscale()
                self.figure_1.suptitle(f"$t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $z={self.fields_paxisZ[zcut_idx]/l0:.2f}~\lambda$")
                self.canvas_1.draw()
                time.sleep(0.05)
                app.processEvents()

            self.loop_in_process = False
        self.updateInfoLabelMem()

    def onRemoveScalar(self):
        if not self.INIT_tabScalar: #IF TAB OPEN AND SIM LOADED
            del self.fields_image_list, self.fields_data_list,self.fields_paxisX,self.fields_paxisY,self.fields_paxisZ,
            self.extentXY,self.extentZX,self.fields_t_range,self.fields_trans_mid_idx,self.fields_long_mid_idx
            gc.collect()
        self.updateInfoLabelMem()

    def onRemoveFields(self):
        if not self.INIT_tabFields: #IF TAB OPEN AND SIM LOADED
            del self.fields_image_list, self.fields_data_list,self.fields_paxisX,self.fields_paxisY,self.fields_paxisZ,
            self.extentXY,self.extentZX,self.fields_t_range,self.fields_trans_mid_idx,self.fields_long_mid_idx
            gc.collect()
        self.updateInfoLabelMem()

    def onRemoveTrack(self):
        if not self.INIT_tabTrack:
            del self.track_N, self.track_t_range,self.track_traj,self.x,self.y,self.z,self.px,self.py,self.pz,self.r,self.Lz_track
            gc.collect()
        self.updateInfoLabelMem()

    def onRemovePlasma(self):
        if not self.INIT_tabPlasma: #IF TAB OPEN AND SIM LOADED
            del self.plasma_paxisX_long, self.plasma_paxisZ_long, self.plasma_t_range, self.plasma_paxisX, self.plasma_paxisY,
            self.plasma_paxisZ_Bz, self.plasma_paxisZ_Weight, self.extentXZ_long, self.extentXY, self.plasma_image_list, self.plasma_data_list
            gc.collect()
        self.updateInfoLabelMem()

    def onRemoveTornado(self):
        gc.collect()
        self.updateInfoLabelMem()

    def onUpdateTabTrack(self, check_id):

        if self.INIT_tabFields == None or self.is_sim_loaded == False: return
        l0 = 2*pi
        if self.INIT_tabTrack or check_id==-1: #if change of name reinit
            print("===== INIT TRACK TAB =====")

            track_name = self.track_file_BOX.currentText()
            # self.displayLoadingLabel()
            app.processEvents()
            try:
                T0 = self.S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
            except Exception:
                self.error_msg = QtWidgets.QMessageBox()
                self.error_msg.setIcon(QtWidgets.QMessageBox.Critical)
                self.error_msg.setWindowTitle("Error")
                self.error_msg.setText("No TrackParticles diagnostic found")
                self.error_msg.exec_()
                return

            self.track_N_tot = T0.nParticles
            self.track_t_range = T0.getTimes()
            self.track_traj = T0.getData()
            self.track_time_SLIDER.setMaximum(len(self.track_t_range)-1)
            # self.extentXY = [-2*self.w0,2*self.w0,-2*self.w0,2*self.w0]

            del T0
            N_part = int(self.track_Npart_EDIT.text())
            self.x = self.track_traj["x"][:,::N_part]-self.Ltrans/2
            self.track_N = self.x.shape[1]

            self.y = self.track_traj["y"][:,::N_part]-self.Ltrans/2
            self.z = self.track_traj["z"][:,::N_part]
            self.py = self.track_traj["py"][:,::N_part]
            self.pz = self.track_traj["pz"][:,::N_part]
            self.px = self.track_traj["px"][:,::N_part]
            self.r = np.sqrt(self.x**2 + self.y**2)
            self.Lz_track =  self.x*self.py - self.y*self.px

            byte_size_track = getsizeof(self.Lz_track) + getsizeof(self.r)
            + getsizeof(self.x)+getsizeof(self.y) + getsizeof(self.z)
            + getsizeof(self.px)+getsizeof(self.py) + getsizeof(self.pz)
            del self.track_traj
            gc.collect()

            print("Memory from TRACK:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/psutil.virtual_memory().total,1),"%)")

            self.INIT_tabTrack = False
            # self.loading_LABEL.deleteLater()
            app.processEvents()
            self.updateInfoLabelMem()

        if check_id <= 0:
            if len(self.figure_2.axes) !=0:
                for ax in self.figure_2.axes: ax.remove()

            ax1,ax2 = self.figure_2.subplots(1,2)
            time0 = time.perf_counter()
            mean_coef = 5
            ax1.scatter(self.r[0]/l0,self.Lz_track[-1],s=1,label="$L_z$")

            a_range_r,MLz = self.averageAM(self.r[0], self.Lz_track[-1], 0.5)
            ax1.plot(a_range_r/l0, MLz*mean_coef,"r",label="5<$L_z$>")
            ax1.grid()
            ax1.legend()
            # im = ax2.imshow(Lz_interp,extent=extent_interp,cmap="RdYlBu")
            self.track_trans_distrib_im = ax2.scatter(self.x[0]/l0,self.y[0]/l0,c=self.Lz_track[-1],s=1,cmap="RdYlBu")
            self.figure_2.colorbar(self.track_trans_distrib_im,ax=ax2,pad=0.01)
            self.figure_2.suptitle(f"t={self.track_t_range[-1]/self.l0:.2f}$~t_0$ (N={self.track_N/1000:.2f}k)")
            self.figure_2.tight_layout()
            self.canvas_2.draw()
            time1 = time.perf_counter()
            print("draw:",(time1-time0)*1000,"ms")

        if check_id == 100 or check_id==101:#SLIDER UPDATE

            if check_id == 101:
                time_edit = float(self.track_time_EDIT.text())
                time_idx = np.where(abs(self.track_t_range/l0-time_edit)==np.min(abs(self.track_t_range/l0-time_edit)))[0][0]
                self.track_time_SLIDER.setValue(time_idx)
                self.track_time_EDIT.setText(str(round(self.track_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.track_time_SLIDER.sliderPosition()
                self.track_time_EDIT.setText(str(round(self.track_t_range[time_idx]/l0,2)))
            self.track_trans_distrib_im.set_array(self.Lz_track[time_idx])
            self.figure_2.suptitle(f"t={self.track_t_range[time_idx]/self.l0:.2f}$~t_0$ (N={self.track_N/1000:.2f}k)")
            self.canvas_2.draw()

        elif check_id == 1000: #PLAY ANIMATION
            combo_box_index = self.track_file_BOX.currentIndex()
            if combo_box_index==0:
                anim_speed = 0.001 #high time resolution = higher refresh rate
                every_frame = 10
            if self.loop_in_process: return
            self.loop_in_process = True
            for time_idx in range(0,len(self.track_t_range),every_frame):
                self.track_time_SLIDER.setValue(time_idx)
                # im.set_offsets(np.c_[x[time_idx]/l0,y[time_idx]/l0])
                self.track_trans_distrib_im.set_array(self.Lz_track[time_idx])
                self.figure_2.suptitle(f"t={self.track_t_range[time_idx]/self.l0:.2f}$~t_0$ (N={self.track_N/1000:.2f}k)")
                self.canvas_2.draw()
                # print('drawn')
                time.sleep(anim_speed)
                app.processEvents()
            self.loop_in_process = False
        self.updateInfoLabelMem()


    def averageAM(self, X,Y,dr_av):
        M = []
        da = 0.04
        t0 = time.perf_counter()
        print("Computing average...",da)
        a_range = np.arange(0,np.max(X)*1.0+da,da)
        M = np.empty(a_range.shape)
        for i,a in enumerate(a_range):
            mask = (X > a-dr_av/2) & (X < a+dr_av/2)
            M[i] = np.nanmean(Y[mask])
        t1 = time.perf_counter()
        print(f"...{(t1-t0):.0f} s")
        return a_range,M

    def onUpdateTabPlasmaFigure(self, plasma_data_list_used_selected_plasma_names):
        plasma_data_list, used_selected_plasma_names = plasma_data_list_used_selected_plasma_names
        l0=2*pi
        boolList = [check.isChecked() for check in self.plasma_check_list]
        selected_plasma_names = np.array(self.plasma_names)[boolList]

        Naxis = sum(boolList)

        print(selected_plasma_names,used_selected_plasma_names)
        if not np.array_equal(selected_plasma_names, used_selected_plasma_names):
            print("plasma get data list DISCARDED")
            return
        self.plasma_data_list = plasma_data_list

        only_trans = sum(["trans" in name for name in selected_plasma_names]) == Naxis
        only_long = sum(["trans" in name for name in selected_plasma_names]) == 0

        ne = self.S.namelist.ne
        VMAX_Bz = 0.001*self.toTesla*self.a0*ne/0.01 #1 = 10709T
        vmax_ptheta = 0.005


        #=====================================
        # REMOVE ALL FIGURES --> NOT OPTIMAL !
        #=====================================
        if len(self.figure_3.axes) !=0:
            for ax in self.figure_3.axes: ax.remove()

        time_idx = self.plasma_time_SLIDER.sliderPosition()
        z_idx = self.plasma_zcut_SLIDER.sliderPosition()
        k=0
        for i in range(len(self.plasma_names)):
            if boolList[i]:
                print(i,k,self.plasma_names[i])
                if only_trans:
                    ax = self.figure_3.add_subplot(1,Naxis,k+1)
                elif only_long:
                    ax = self.figure_3.add_subplot(Naxis,1,k+1)
                elif Naxis <= 2:
                    ax = self.figure_3.add_subplot(1,Naxis,k+1)
                else:
                    ax = self.figure_3.add_subplot(2,2,k+1)

                if "Bz" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin = -VMAX_Bz
                    vmax =  VMAX_Bz
                elif "ptheta" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin = -vmax_ptheta
                    vmax =  vmax_ptheta
                elif "ne" in self.plasma_names[i]:
                    cmap = "jet"
                    vmin = 0
                    vmax = 2
                else:
                    cmap = "RdYlBu"
                    vmin = -0.1*np.max(np.abs(self.plasma_data_list[k][time_idx]))
                    vmax =  0.1*np.max(np.abs(self.plasma_data_list[k][time_idx]))

                if "trans" in self.plasma_names[i]:
                    extent = self.extentXY
                    data = self.plasma_data_list[k][time_idx,:,:,z_idx]
                else:
                    extent = self.extentXZ_long
                    data = self.plasma_data_list[k][time_idx]

                im = ax.imshow(data, aspect="auto",
                                origin="lower", cmap = cmap,extent=extent, vmin=vmin, vmax=vmax) #bwr, RdYlBu
                ax.set_title(self.plasma_names[i])


                self.figure_3.colorbar(im, ax=ax,pad=0.01)
                self.plasma_image_list.append(im)
                k+=1
        self.figure_3.suptitle(f"t={self.plasma_t_range[time_idx]/l0:.2f} t0")
        for w in range(10):
            self.figure_3.tight_layout()
            self.figure_3.tight_layout()
        self.canvas_3.draw()
        # self.loading_LABEL.deleteLater()

    def onUpdateTabPlasma(self, check_id):
        if self.INIT_tabPlasma == None or self.is_sim_loaded == False: return
        if self.INIT_tabPlasma:
            l0 = 2*pi

            plasma_species_exist = "ion" in [s.name for s in self.S.namelist.Species]
            if not plasma_species_exist:
                self.error_msg = QtWidgets.QMessageBox()
                self.error_msg.setIcon(QtWidgets.QMessageBox.Critical)
                self.error_msg.setWindowTitle("Error")
                self.error_msg.setText("No Plasma (Ions) Species found")
                self.error_msg.exec_()
                return

            print("===== INIT FIELDS PLASMA =====")
            # self.displayLoadingLabel(self.plasma_groupBox)

            Bz_long_diag = self.S.Probe(2,"Bz")
            self.plasma_paxisX_long = Bz_long_diag.getAxis("axis1")[:,0]
            self.plasma_paxisZ_long = Bz_long_diag.getAxis("axis2")[:,2]
            self.plasma_t_range = Bz_long_diag.getTimes()

            Bz_trans_diag = self.S.Probe(1,"Bz")
            self.plasma_paxisX = Bz_trans_diag.getAxis("axis1")[:,0]
            self.plasma_paxisY = Bz_trans_diag.getAxis("axis2")[:,1]
            self.plasma_paxisZ_Bz = Bz_trans_diag.getAxis("axis3")[:,2]

            Bweight_XY = self.S.ParticleBinning("weight_trans")
            self.plasma_paxisZ_Weight = Bweight_XY.getAxis("z")
            self.toTesla = 10709


            self.extentXZ_long = [self.plasma_paxisZ_long[0]/l0,self.plasma_paxisZ_long[-1]/l0,
                                  self.plasma_paxisX_long[0]/l0-self.Ltrans/l0/2,self.plasma_paxisX_long[-1]/l0-self.Ltrans/l0/2]
            self.extentXY = [self.plasma_paxisX[0]/l0-self.Ltrans/l0/2,self.plasma_paxisX[-1]/l0-self.Ltrans/l0/2,
                             self.plasma_paxisY[0]/l0-self.Ltrans/l0/2,self.plasma_paxisY[-1]/l0-self.Ltrans/l0/2]

            self.plasma_time_SLIDER.setMaximum(len(self.plasma_t_range)-1)
            self.plasma_zcut_SLIDER.setMaximum(len(self.plasma_paxisZ_Bz)-1)
            self.plasma_time_SLIDER.setValue(len(self.plasma_t_range)-1)
            self.plasma_zcut_SLIDER.setValue(len(self.plasma_paxisZ_Bz)-3)

            self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[-1]/l0,2)))
            self.plasma_zcut_EDIT.setText(str(round(self.plasma_paxisZ_Bz[-3]/l0,2)))

            self.plasma_names =["Bz","Bz_trans","ne","ne_trans","Lz","Lz_trans","ptheta","ptheta_trans", "Jtheta", "Jtheta_trans"]
            self.plasma_check_list = [self.plasma_Bz_CHECK,self.plasma_Bz_trans_CHECK,
                                      self.plasma_ne_CHECK,self.plasma_ne_trans_CHECK,
                                      self.plasma_Lz_CHECK, self.plasma_Lz_trans_CHECK,
                                      self.plasma_ptheta_CHECK, self.plasma_ptheta_trans_CHECK,
                                      self.plasma_Jtheta_CHECK, self.plasma_Jtheta_trans_CHECK]


            self.plasma_image_list = []
            self.plasma_data_list = []

            # self.loading_LABEL.deleteLater()
            self.INIT_tabPlasma = False
            app.processEvents()
            self.updateInfoLabelMem()


        ne = self.S.namelist.ne
        l0 = 2*pi
        if check_id < 10: #CHECK_BOX UPDATE

            boolList = [check.isChecked() for check in self.plasma_check_list]

            if sum(boolList)>4:
                self.plasma_check_list[check_id].setChecked(False)
                boolList[check_id] = False
                # self.loading_LABEL.deleteLater()
                return

            #=====================================
            # REMOVE ALL FIGURES --> NOT OPTIMAL
            #=====================================
            if len(self.figure_3.axes) !=0:
                for ax in self.figure_3.axes: ax.remove()

            selected_plasma_names = np.array(self.plasma_names)[boolList]

            self.plasma_image_list = []
            self.plasma_data_list = []

            self.loadthread = ThreadGetPlasmaProbeData(self.S, selected_plasma_names)
            self.loadthread.finished.connect(self.onUpdateTabPlasmaFigure)
            self.loadthread.start()


        elif check_id <= 210: #SLIDER UPDATE
            tstart = time.perf_counter()
            if check_id == 101: #QLineEdit time
                time_edit_value = float(self.plasma_time_EDIT.text())
                time_idx = np.where(abs(self.plasma_t_range/l0-time_edit_value)==np.min(abs(self.plasma_t_range/l0-time_edit_value)))[0][0]
                self.plasma_time_SLIDER.setValue(time_idx)
                self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.plasma_time_SLIDER.sliderPosition()
                self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[time_idx]/l0,2)))

            if check_id == 201: #QLineEdit zcut
                zcut_edit_value = float(self.plasma_zcut_EDIT.text())
                zcut_idx = np.where(abs(self.plasma_paxisZ/l0-zcut_edit_value)==np.min(abs(self.plasma_paxisZ/l0-zcut_edit_value)))[0][0]
                self.plasma_zcut_SLIDER.setValue(zcut_idx)
            else:
                zcut_idx = self.plasma_zcut_SLIDER.sliderPosition()
                self.plasma_zcut_EDIT.setText(str(round(self.plasma_paxisZ_Bz[zcut_idx]/l0,2)))


            ne = self.S.namelist.ne

            for i,im in enumerate(self.plasma_image_list):

                boolList = [check.isChecked() for check in self.plasma_check_list]
                selected_plasma_names = np.array(self.plasma_names)[boolList]

                if "_trans" in selected_plasma_names[i]:
                    im.set_data(self.plasma_data_list[i][time_idx,:,:,zcut_idx])
                    self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($z={self.plasma_paxisZ_Weight[zcut_idx]/l0:.1f}~\lambda$)")
                else:
                    im.set_data(self.plasma_data_list[i][time_idx])

                 #bwr, RdYlBu
            self.figure_3.suptitle(f"$t={self.plasma_t_range[time_idx]/l0:.2f}~t_0$")
            self.canvas_3.draw()

        elif check_id == 1000 :

            if self.loop_in_process: return

            self.loop_in_process = True

            zcut_idx = self.plasma_zcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.plasma_t_range)):
                self.plasma_time_SLIDER.setValue(time_idx)

                for i,im in enumerate(self.plasma_image_list):

                    boolList = [check.isChecked() for check in self.plasma_check_list]
                    selected_plasma_names = np.array(self.plasma_names)[boolList]

                    if "_trans" in selected_plasma_names[i]:
                        im.set_data(self.plasma_data_list[i][time_idx,:,:,zcut_idx])
                        self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($z={self.plasma_paxisZ_Bz[zcut_idx]/l0:.1f}~\lambda$)")
                    else:
                        im.set_data(self.plasma_data_list[i][time_idx,:,:])

                self.figure_3.suptitle(f"t={self.plasma_t_range[time_idx]/self.l0:.2f}$~t_0$")
                self.canvas_3.draw()
                time.sleep(0.01)
                app.processEvents()

            self.loop_in_process = False


    def onCloseProgressBar(self, sim_id_int):

        self.finished_sim_hist.remove(sim_id_int)
        layout_to_del = self.layout_progress_bar_dict[str(sim_id_int)]
        print("to del:",layout_to_del)
        for i in range(self.layoutTornado.count()):
            layout_progressBar = self.layoutTornado.itemAt(i)
            print(i,layout_progressBar)
            if layout_progressBar == layout_to_del:
                print("delete:",layout_progressBar)
                self.deleteLayout(self.layoutTornado, i)


    def async_onUpdateTabTornado(self, download_trnd_json = True):
        print("async check")
        """
        asynchronous function called PERIODICALLY
        """
        sim_json_name = "simulation_info.json"
        with open(sim_json_name) as f:
            self.sim_dict = json.load(f)

        OLD_NB_SIM_RUNNING = len(self.running_sim_hist)
        CURRENT_NB_SIM_RUNNING = len(self.sim_dict) - 1 #-1 for datetime

        print(list(self.sim_dict))

        #================================
        # CHECK FOR FINISHED SIMULATIONS
        #================================
        if (CURRENT_NB_SIM_RUNNING <= OLD_NB_SIM_RUNNING) and (list(self.sim_dict) != list(self.running_sim_hist)): #AT LEAST ONE SIMULATION HAS FINISHED
            for n,old_sim_id_int in enumerate(self.running_sim_hist):
                if str(old_sim_id_int) not in list(self.sim_dict): #this simulation has finished
                    print(self.previous_sim_dict)
                    finished_sim_path = self.previous_sim_dict[str(old_sim_id_int)]["job_full_path"]
                    finished_sim_name = self.previous_sim_dict[str(old_sim_id_int)]["job_full_name"]
                    print(finished_sim_path,"is download is available ! \a") #\a
                    self.showToast('Tornado download is available', finished_sim_name)

                    self.finished_sim_hist.append(old_sim_id_int)
                    self.running_sim_hist.remove(old_sim_id_int)
                    self.can_download_sim_dict[int(old_sim_id_int)] = finished_sim_path

                    layout = self.layout_progress_bar_dict[str(old_sim_id_int)]
                    progress_bar = layout.itemAt(2).widget()
                    ETA_LABEL = layout.itemAt(3).widget()
                    dl_sim_BUTTON = layout.itemAt(4).widget()

                    progress_bar.setStyleSheet(self.qss_progressBar_COMPLETED)
                    progress_bar.setValue(100)
                    ETA_LABEL.setStyleSheet("background-color: lightpink")
                    ETA_LABEL.setText(" - ")
                    dl_sim_BUTTON.setStyleSheet("border-color: red")

                    close_BUTTON = QtWidgets.QPushButton()
                    close_BUTTON = QtWidgets.QPushButton()
                    close_BUTTON.setFixedSize(25,25)
                    close_BUTTON.setIcon(QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\close_button_trans.png"))
                    close_BUTTON.setIconSize(QtCore.QSize(25, 25))
                    close_BUTTON.setStyleSheet("border-radius:0px; border:0px")
                    close_BUTTON.clicked.connect(lambda: self.onCloseProgressBar(old_sim_id_int))
                    layout.addWidget(close_BUTTON)
        #================================
        # CHECK FOR NEW SIMULATIONS
        # OR UPDATE CURRENT ONE
        #================================
        for sim_id in self.sim_dict:
            if sim_id == "datetime": continue #only open sim data and not metadata (located at the end of dict)
            sim = self.sim_dict[sim_id]
            sim_id_int = int(sim_id)
            sim_progress = sim["progress"]*100
            sim_ETA = sim["ETA"].rjust(5)
            sim_name = sim["job_full_name"][:-3]
            sim_nodes = int(sim["NODES"])

            if (sim_id_int not in self.running_sim_hist) and (sim_id_int not in self.finished_sim_hist):

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA)
                self.layout_progress_bar_dict[sim_id] = layoutProgressBar

                item_count = self.layoutTornado.count()

                spacer_item = self.layoutTornado.itemAt(item_count-1)
                self.layoutTornado.removeItem(spacer_item)

                self.layoutTornado.addLayout(layoutProgressBar)
                verticalSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
                self.layoutTornado.addSpacerItem(verticalSpacer)

                self.running_sim_hist.append(sim_id_int)
            else:
                progress_bar = self.layout_progress_bar_dict[sim_id].itemAt(2).widget()
                progress_bar.setValue(round(sim_progress))
                ETA_label = self.layout_progress_bar_dict[sim_id].itemAt(3).widget()
                ETA_label.setText(sim_ETA)
        #Update label with Update datetime
        sim_datetime = self.sim_dict["datetime"]
        self.tornado_last_update_LABEL.setText(f"Last updated: {sim_datetime}")

        self.previous_sim_dict = self.sim_dict
        app.processEvents()
        return

    def call_ThreadDownloadSimJSON(self):
        self.loadthread = ThreadDownloadSimJSON("/sps3/jeremy/LULI/simulation_info.json", os.environ["SMILEI_QT"])
        self.loadthread.finished.connect(self.async_onUpdateTabTornado)
        self.loadthread.start()
        return

    def call_ThreadDownloadSimData(self,sim_id):
        self.loadthread = ThreadDownloadSimData(sim_id)
        self.loadthread.finished.connect(lambda:self.onDownloadSimDataFinished(sim_id))
        self.loadthread.start()
        return

    def onDownloadSimDataFinished(self,sim_id):
        layout = self.layout_progress_bar_dict[str(sim_id)]
        progress_bar = layout.itemAt(2).widget()
        dl_sim_BUTTON = layout.itemAt(4).widget()

        dl_sim_BUTTON.setStyleSheet("border-color: green")
        dl_sim_BUTTON.setEnabled(False)
        progress_bar.setStyleSheet(self.qss_progressBar_DOWNLOADED)
        return



    def onInitTabTornado(self):
        if self.INIT_tabTornado == None: return
        if self.INIT_tabTornado:

            self.tornado_update_TIMER = QtCore.QTimer()
            refresh_time_min = 10 #minute
            self.tornado_update_TIMER.setInterval(int(refresh_time_min*60*1000)) #in ms
            # self.tornado_update_TIMER.timeout.connect(self.async_onUpdateTabTornado)
            self.tornado_update_TIMER.timeout.connect(self.call_ThreadDownloadSimJSON)
            self.tornado_update_TIMER.start()

            sim_json_name = "simulation_info.json"
            with open(sim_json_name) as f:
                self.sim_dict = json.load(f)
                self.previous_sim_dict = self.sim_dict

            #=============================
            # INIT PROGRESS BARS
            #=============================
            self.running_sim_hist = []
            self.finished_sim_hist = []
            self.layout_progress_bar_dict = {}
            self.can_download_sim_dict = {}

            for sim_id in self.sim_dict:
                if sim_id == "datetime": continue #only open sim data and not metadata (located at the end of dict)
                sim = self.sim_dict[sim_id]
                sim_progress = sim["progress"]*100
                sim_ETA = sim["ETA"].rjust(5)
                sim_name = sim["job_full_name"][:-3]
                sim_nodes = int(sim["NODES"])

                self.running_sim_hist.append(int(sim_id))

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA)

                self.layout_progress_bar_dict[sim_id] = layoutProgressBar
                self.layoutTornado.addLayout(layoutProgressBar)
            verticalSpacer = QtWidgets.QSpacerItem(1, 1, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
            self.layoutTornado.addSpacerItem(verticalSpacer)
            # self.layoutTornado.addStretch(100)
            sim_datetime = self.sim_dict["datetime"]

            self.tornado_last_update_LABEL.setText(f"Last updated: {sim_datetime}")
            self.INIT_tabTornado = False
            self.call_ThreadDownloadSimJSON()
            app.processEvents()

    def createLayoutProgressBar(self, sim_id, sim_progress, sim_name, sim_nodes, sim_ETA):
        layoutProgressBar = QtWidgets.QHBoxLayout()

        tornado_PROGRESS_BAR = QtWidgets.QProgressBar(maximum=100)
        tornado_PROGRESS_BAR.setValue(round(sim_progress))
        tornado_PROGRESS_BAR.setFont(QFont('Arial', 15))
        tornado_PROGRESS_BAR.setAlignment(QtCore.Qt.AlignCenter)

        custom_FONT = QtGui.QFont("Courier New", 14,QFont.Bold)

        sim_name_LABEL = QtWidgets.QLabel(f"[{sim_id}] {sim_name}")
        sim_name_LABEL.setFont(custom_FONT)
        sim_name_LABEL.setMinimumWidth(650) #450 FOR LAPTOP
        sim_name_LABEL.setStyleSheet("background-color: lightblue")
        sim_name_LABEL.setWordWrap(True)
        sim_name_LABEL.setAlignment(QtCore.Qt.AlignCenter)

        sim_node_LABEL = QtWidgets.QLabel(f"NDS:{sim_nodes}")
        sim_node_LABEL.setFont(custom_FONT)
        sim_node_LABEL.setStyleSheet("background-color: lightblue")

        ETA_LABEL = QtWidgets.QLabel(sim_ETA)
        ETA_LABEL.setFont(custom_FONT)
        ETA_LABEL.setStyleSheet("background-color: lightblue")
        # ETA_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        ETA_LABEL.setMinimumWidth(75)

        dl_sim_BUTTON = QtWidgets.QPushButton()
        dl_sim_BUTTON.setIcon(QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\download_button.png"))
        dl_sim_BUTTON.setFixedSize(35,35)
        dl_sim_BUTTON.setIconSize(QtCore.QSize(25, 25))

        dl_sim_BUTTON.clicked.connect(partial(self.call_ThreadDownloadSimData, sim_id))



        layoutProgressBar.addWidget(sim_name_LABEL)
        layoutProgressBar.addWidget(sim_node_LABEL)
        layoutProgressBar.addWidget(tornado_PROGRESS_BAR)
        layoutProgressBar.addWidget(ETA_LABEL)
        layoutProgressBar.addWidget(dl_sim_BUTTON)

        layoutProgressBar.setContentsMargins(25,20,25,20) #left top right bottom
        return layoutProgressBar


    def creatPara(self,name, widget, adjust_label=False,fontsize=10):
        layout =  QtWidgets.QHBoxLayout()
        label = QtWidgets.QLabel(name)
        # label.setStyleSheet("background-color:rgba(0, 0, 0, 50); border: 1px solid black")
        custom_FONT = QtGui.QFont("Arial", fontsize)
        label.setFont(custom_FONT)
        if adjust_label:
            label.adjustSize()
            label.setFixedWidth(label.geometry().width())
        layout.addWidget(label)
        layout.addWidget(widget)
        return layout

    def onDownloadSimData(self, sim_id):
        print("===========================")
        print("downloading request for", sim_id)

        job_full_path = self.can_download_sim_dict[int(sim_id)]
        print(job_full_path)
        self.loadthread = ThreadDownloadSimData(job_full_path)
        self.loadthread.start()
        # self.downloadSimData(job_full_path) #"_NEW_PLASMA_/new_plasma_LG_optic_ne0.01_dx12/")

    def showToast(self,msg1,msg2=None):
        toast = Toast(self)
        toast.setDuration(5000)  # Hide after 5 seconds
        toast.setTitle(msg1)
        toast.setText(msg2)
        toast.applyPreset(ToastPreset.SUCCESS)  # Apply style preset
        toast.show()


    def printSI(self,x,baseunit,ndeci=2):
        prefix="yzafpnµm kMGTPEZY"
        shift=decimal.Decimal('1E24')
        d=(decimal.Decimal(str(x))*shift).normalize()
        m,e=d.to_eng_string().split('E')
        return m[:4] + " " + prefix[int(e)//3] + baseunit

class ProxyStyle(QtWidgets.QProxyStyle):
    """Overwrite the QSlider: left click place the cursor at cursor position"""
    def styleHint(self, hint, opt=None, widget=None, returnData=None):
        res = super().styleHint(hint, opt, widget, returnData)
        if hint == self.SH_Slider_AbsoluteSetButtons:
            res |= QtCore.Qt.LeftButton
        return res

if __name__ == '__main__':
    myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.Auto)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough);


    qdarktheme.enable_hi_dpi()


    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle()) #Apply slider style

    # app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))

    # qdarktheme.setup_theme("light")

    pixmap = QtGui.QPixmap(os.environ["SMILEI_QT"]+'\\Ressources\\smileiIcon.png')
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()

    main = MainWindow()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())