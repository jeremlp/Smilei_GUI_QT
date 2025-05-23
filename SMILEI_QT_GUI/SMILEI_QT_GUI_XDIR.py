# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 18:54:10 2024

@author: Jeremy La Porte
"""
import sys
from sys import getsizeof
import os
module_dir_happi = f"{os.environ['SMILEI_SRC']}"
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
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import time
import gc
import psutil
from scipy.interpolate import griddata
from scipy import integrate
import scipy
from scipy import special

import math

import ctypes

import tools_dialog
import log_dialog
import IPython_dialog
import memory_dialog
import tree_dialog
import paramiko_SSH_SCP_class
import class_threading
import generate_diag_id

import subprocess
import json
from pathlib import Path
from functools import partial
# from win11toast import toast
from pyqttoast import ToastPreset
import utils
# from utils import Popup, encrypt
import decimal
import re

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        # super().__init__()
        print("... INIT ...")
        screen = app.primaryScreen()
        size = screen.size()

        print("SCREEN SIZE:",size.width(),"x",size.height())
        is_screen_2K = size.width() > 1920

        window_height = int(size.height()/1.3333)
        window_width = int(size.width()/1.3333)
        
        self.resolution_scaling = 1.0
        self.toolBar_height = 40 #45 for Tower


        self.thread = QThreadPool()


        groupBox_bg_color_light = "#f0f0f0"
        groupBox_border_color_light = "#FF17365D"

        font_color_light = "black"
        font_color_dark = "white"
        
        self.qss_plt_title = {'fontname':'Courier New','fontweight':'bold'}
        
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

        self.qss_checkBox = """
        QCheckBox {
        font: 12pt "MS Shell Dlg 2";
            }
        """
        self.qss_radioButton = """
        QRadioButton {
        font: 12pt "MS Shell Dlg 2";
            }
        """
        self.theme = "light"
        if self.theme == "dark":
            self.qss = self.dark_qss_groupBox + self.dark_qss_tab + self.dark_qss_label + self.qss_button + self.qss_progressBar + self.qss_checkBox + self.qss_radioButton
        else:
            self.qss = self.light_qss_groupBox + self.light_qss_tab + self.light_qss_label + self.qss_button + self.qss_progressBar + self.qss_checkBox + self.qss_radioButton

        qdarktheme.setup_theme(self.theme,additional_qss=self.qss)
        self.setGeometry(155,50,window_width,window_height)

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
        self.DISK = psutil.disk_usage
        #======================================================================
        self.SCRIPT_VERSION_ID, self.SCRIPT_VERSION_NAME ='0.15.6', 'Compa Track'
        #======================================================================
        self.SCRIPT_VERSION = self.SCRIPT_VERSION_ID + " - " + self.SCRIPT_VERSION_NAME
        self.COPY_RIGHT = "Jeremy LA PORTE"
        self.spyder_default_stdout = sys.stdout

        #==============================
        # MENU BAR
        #==============================
        self.menuBar = self.menuBar()
        # self.menuBar.setGeometry(QtCore.QRect(0, 0, window_width, 21))
        self.fileMenu = self.menuBar.addMenu("&File")
        self.editMenu = self.menuBar.addMenu("&Edit")

        #Actions
        self.actionOpenSim = QtWidgets.QAction("Open Simulation",self)
        self.actionOpenLogs = QtWidgets.QAction("Open Logs",self)
        self.actionOpenIPython = QtWidgets.QAction("Open IPython",self)
        self.actionOpenMemory = QtWidgets.QAction("Open Memory Graph",self)
        self.actionOpenTree = QtWidgets.QAction("Open Tree Diag_ID",self)
        self.actionOpenAllTools = QtWidgets.QAction("OPEN ALL TOOLS",self)

        self.actionDiagScalar = QtWidgets.QAction("Scalar",self)
        self.actionDiagFields = QtWidgets.QAction("Fields",self)
        self.actionDiagIntensity = QtWidgets.QAction("Intensity",self)
        self.actionDiagTrack = QtWidgets.QAction("Track",self)
        self.actionDiagPlasma = QtWidgets.QAction("Plasma",self)
        self.actionDiagPlasma = QtWidgets.QAction("Plasma",self)
        self.actionDiagBinning = QtWidgets.QAction("Binning",self)
        self.actionDiagCompa = QtWidgets.QAction("Comparison",self)
        self.actionTornado = QtWidgets.QAction("Tornado",self)

        self.actionDiagScalar.setCheckable(True)
        self.actionDiagFields.setCheckable(True)
        self.actionDiagIntensity.setCheckable(True)

        self.actionDiagTrack.setCheckable(True)
        self.actionDiagPlasma.setCheckable(True)
        self.actionDiagBinning.setCheckable(True)
        self.actionDiagCompa.setCheckable(True)

        self.actionTornado.setCheckable(True)

        self.fileMenu.addAction(self.actionOpenSim)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenLogs)
        self.fileMenu.addAction(self.actionOpenIPython)
        self.fileMenu.addAction(self.actionOpenMemory)
        self.fileMenu.addAction(self.actionOpenTree)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.actionOpenAllTools)
        
        self.menuBar.addAction(self.fileMenu.menuAction())

        self.editMenu.addAction(self.actionDiagScalar)
        self.editMenu.addAction(self.actionDiagFields)
        self.editMenu.addAction(self.actionDiagIntensity)
        self.editMenu.addAction(self.actionDiagTrack)
        self.editMenu.addAction(self.actionDiagPlasma)
        self.editMenu.addAction(self.actionDiagBinning)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.actionDiagCompa)
        self.editMenu.addSeparator()
        self.editMenu.addAction(self.actionTornado)
        self.menuBar.addAction(self.editMenu.menuAction())

        #==============================
        # SETTINGS
        #==============================
        self.settings_groupBox = QtWidgets.QGroupBox("Open Simulation")
        self.settings_groupBox.setFixedWidth(300)
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

        # boxLayout_settings.addLayout(layoutSimulationBasePath)
        boxLayout_settings.addLayout(layoutLoadSim)
        boxLayout_settings.addWidget(self.sim_directory_name_LABEL)

        # boxLayout_settings.addLayout(layoutSimGeometry)
        self.settings_groupBox.setLayout(boxLayout_settings)

        #==============================
        # SIM INFO
        #==============================
        self.sim_info_groupBox = QtWidgets.QGroupBox("Simulation Parameters")
        self.sim_info_groupBox.setFixedWidth(300)
        boxLayout_sim_info = QtWidgets.QVBoxLayout()

        self.geometry_LABEL = QtWidgets.QLabel("")
        self.geometry_LABEL.setFont(self.medium_FONT)
        layoutGeometry = self.creatPara("Geometry :", self.geometry_LABEL)
        boxLayout_sim_info.addLayout(layoutGeometry)
        boxLayout_sim_info.addWidget(QtWidgets.QLabel("-"*25))
        laser_param_LABEL= QtWidgets.QLabel("LASER PARAMETERS")
        laser_param_LABEL.setFont(self.small_bold_FONT)
        laser_param_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        boxLayout_sim_info.addWidget(laser_param_LABEL)
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
        box_param_LABEL= QtWidgets.QLabel("BOX PARAMETERS")
        box_param_LABEL.setFont(self.small_bold_FONT)
        box_param_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        boxLayout_sim_info.addWidget(box_param_LABEL)
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

        self.SI_assume_LABEL= QtWidgets.QLabel("SI UNITS (𝝀 = 1 µm)")
        self.SI_assume_LABEL.setFont(self.small_bold_FONT)
        self.SI_assume_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        boxLayout_sim_info.addWidget(self.SI_assume_LABEL)

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

        self.run_time_LABEL = QtWidgets.QLabel("")
        self.run_time_LABEL.setFont(self.medium_bold_FONT)
        layoutRunTime = self.creatPara("Run time :", self.run_time_LABEL)
        boxLayout_sim_info.addLayout(layoutRunTime)

        self.diag_id_LABEL = QtWidgets.QLabel("")
        self.diag_id_LABEL.setFont(self.medium_FONT)
        layoutRunTime = self.creatPara("Diag ID :", self.diag_id_LABEL)
        boxLayout_sim_info.addLayout(layoutRunTime)

        verticalSpacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        boxLayout_sim_info.addItem(verticalSpacer)
        self.sim_info_groupBox.setLayout(boxLayout_sim_info)

        boxLayoutLEFT = QtWidgets.QVBoxLayout()
        boxLayoutLEFT.addWidget(self.settings_groupBox)
        boxLayoutLEFT.addWidget(self.sim_info_groupBox)
        boxLayoutLEFT.setContentsMargins(2,0,0,0)


        #=====================================================================
        # TAB WIDGET
        #=====================================================================
        #---------------------------------------------------------------------
        # TAB 0
        #---------------------------------------------------------------------
        self.figure_0 = Figure()
        self.canvas_0 = FigureCanvas(self.figure_0)
        self.plt_toolbar_0 = NavigationToolbar(self.canvas_0)
        self.plt_toolbar_0.setFixedHeight(self.toolBar_height)
        self.ax0 = self.figure_0.add_subplot(1,1,1)
        # self.ax0.grid()
        # self.ax0.set_xlabel("t/t0",fontsize=14)

        fontsize = 12

        self.scalar_check_list = []
        self.scalar_names = ["Utot","Uelm", "Ukin","AM", "Uelm/Utot","α_abs"]
        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        for name in self.scalar_names:
            check = QtWidgets.QCheckBox(name)
            self.scalar_check_list.append(check)
            # layoutScalarCheck = self.creatPara(f"{name} ", check,adjust_label=True,fontsize=fontsize)
            # layoutTabSettingsCheck.addLayout(layoutScalarCheck)
            layoutTabSettingsCheck.addWidget(check)

        self.scalar_check_list[0].setChecked(True)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addWidget(self.plt_toolbar_0)


        self.scalar_groupBox = QtWidgets.QGroupBox("Scalar Diagnostics")
        self.scalar_groupBox.setFixedHeight(int(110*self.resolution_scaling))
        self.scalar_groupBox.setLayout(layoutTabSettings)

        self.layoutScalar = QtWidgets.QVBoxLayout()
        self.layoutScalar.addWidget(self.scalar_groupBox)
        self.layoutScalar.addWidget(self.canvas_0)

        self.scalar_Widget = QtWidgets.QWidget()
        self.scalar_Widget.setLayout(self.layoutScalar)

        #---------------------------------------------------------------------
        # TAB 1
        #---------------------------------------------------------------------
        self.figure_1 = Figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.plt_toolbar_1 = NavigationToolbar(self.canvas_1, self)
        self.plt_toolbar_1.setFixedHeight(self.toolBar_height)

        fontsize = 12

        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        self.fields_names =["Ex","Ey","Ez","Bx","By","Bz","Er","Eθ","Br","Bθ"]
        self.fields_check_list = []
        for i, name in enumerate(self.fields_names):
            fields_CHECK = QtWidgets.QCheckBox(name)
            self.fields_check_list.append(fields_CHECK)
            # layoutFieldsCheck = self.creatPara(name + " ", fields_CHECK,adjust_label=True,fontsize=12)
            # layoutFieldsCheck.setSpacing(0)
            if i in [3,6,8]:
                separator1 = QtWidgets.QFrame()
                separator1.setFrameShape(QtWidgets.QFrame.VLine)
                separator1.setLineWidth(1)
                layoutTabSettingsCheck.addWidget(separator1)
            # layoutTabSettingsCheck.addLayout(layoutFieldsCheck)
            layoutTabSettingsCheck.addWidget(fields_CHECK)

        self.fields_check_list[0].setChecked(True)
        # self.Ey_CHECK.setChecked(True)
        self.fields_check_list[1].setChecked(True)

        layoutTabSettingsCheck.setSpacing(20)
        layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)

        self.sim_cut_direction_BOX = QtWidgets.QComboBox()
        self.sim_cut_direction_BOX.addItem("Longitudinal cut")
        self.sim_cut_direction_BOX.addItem("Transverse cut")

        self.fields_use_autoscale_CHECK = QtWidgets.QCheckBox("Use autoscale")

        layoutTabSettingsCutDirection = QtWidgets.QHBoxLayout()
        layoutTabSettingsCutDirection.addWidget(self.sim_cut_direction_BOX)
        layoutTabSettingsCutDirection.addWidget(self.fields_use_autoscale_CHECK)


        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addLayout(layoutTabSettingsCutDirection)


        self.fields_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fields_time_SLIDER.setRange(0,1)
        self.fields_xcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.fields_xcut_SLIDER.setRange(0,1)

        self.fields_time_EDIT = QtWidgets.QLineEdit("0")

        self.fields_time_EDIT.setValidator(self.float_validator)
        self.fields_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.fields_xcut_EDIT = QtWidgets.QLineEdit("0")
        self.fields_xcut_EDIT.setValidator(self.float_validator)
        self.fields_xcut_EDIT.setMaximumWidth(70)

        self.fields_play_time_BUTTON = QtWidgets.QPushButton("Play")
        self.fields_play_time_BUTTON.setMinimumHeight(15)
        layoutTimeSlider = self.creatPara("t/t0=", self.fields_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.fields_time_SLIDER)
        layoutTimeSlider.addWidget(self.fields_play_time_BUTTON)


        layoutXcutSlider = self.creatPara("x/𝝀=", self.fields_xcut_EDIT)
        layoutXcutSlider.addWidget(self.fields_xcut_SLIDER)

        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addLayout(layoutXcutSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_1)

        self.fields_groupBox = QtWidgets.QGroupBox("Fields Diagnostics")
        self.fields_groupBox.setFixedHeight(int(210*self.resolution_scaling))
        self.fields_groupBox.setLayout(layoutTabSettings)

        self.layoutFields = QtWidgets.QVBoxLayout()
        self.layoutFields.addWidget(self.fields_groupBox)
        self.layoutFields.addWidget(self.canvas_1)

        self.fields_Widget = QtWidgets.QWidget()
        self.fields_Widget.setLayout(self.layoutFields)

        #---------------------------------------------------------------------
        # TAB 2 TRACK
        #---------------------------------------------------------------------
        self.figure_2 = Figure()
        self.canvas_2 = FigureCanvas(self.figure_2)
        self.plt_toolbar_2 = NavigationToolbar(self.canvas_2, self)
        self.plt_toolbar_2.setFixedHeight(self.toolBar_height)
        
        self.figure_2_displace = Figure()
        self.canvas_2_displace = FigureCanvas(self.figure_2_displace)
        self.plt_toolbar_2_displace = NavigationToolbar(self.canvas_2_displace, self)
        self.plt_toolbar_2_displace.setFixedHeight(self.toolBar_height)
        self.canvas_2_displace.hide()
        self.plt_toolbar_2_displace.hide()
        

        self.track_file_BOX = QtWidgets.QComboBox()
        self.track_file_BOX.addItem("track_eon")
        self.track_file_BOX.addItem("track_eon_full")
        self.track_file_BOX.addItem("track_eon_dense")
        self.track_file_BOX.addItem("track_eon_net")

        layoutTabSettingsTrackFile = QtWidgets.QHBoxLayout()
        self.track_Npart_EDIT = QtWidgets.QLineEdit("10")
        self.track_Npart_EDIT.setValidator(self.int_validator)
        self.track_Npart_EDIT.setMaximumWidth(45)

        self.track_update_offset_CHECK = QtWidgets.QCheckBox("Update offsets")
        
        self.track_pannel_BOX = QtWidgets.QComboBox()
        self.track_pannel_BOX.addItem("Angular Momentum")
        self.track_pannel_BOX.addItem("Displacements")
        # self.track_pannel_BOX.addItem("particle trajectories")

        layoutNpart = self.creatPara("Npart=", self.track_Npart_EDIT,adjust_label=True)

        layoutTabSettingsTrackFile.addLayout(layoutNpart)
        layoutTabSettingsTrackFile.addWidget(self.track_file_BOX)
        layoutTabSettingsTrackFile.addWidget(self.track_update_offset_CHECK)
        layoutTabSettingsTrackFile.addWidget(self.track_pannel_BOX)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsTrackFile)

        self.track_time_EDIT = QtWidgets.QLineEdit("0")
        self.track_time_EDIT.setValidator(self.float_validator)
        self.track_time_EDIT.setMaximumWidth(70)
        self.track_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.track_time_SLIDER.setRange(0,1)
        self.track_play_time_BUTTON = QtWidgets.QPushButton("Play")

        layoutTimeSlider = self.creatPara("t/t0=", self.track_time_EDIT,adjust_label=True)
        layoutTimeSlider.addWidget(self.track_time_SLIDER)
        layoutTimeSlider.addWidget(self.track_play_time_BUTTON)
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_2)
        layoutTabSettings.addWidget(self.plt_toolbar_2_displace)

        self.track_groupBox = QtWidgets.QGroupBox("Track Particles Diagnostic")
        self.track_groupBox.setFixedHeight(int(125*self.resolution_scaling))
        self.track_groupBox.setLayout(layoutTabSettings)

        self.layoutTrack = QtWidgets.QVBoxLayout()
        self.layoutTrack.addWidget(self.track_groupBox)
        self.layoutTrack.addWidget(self.canvas_2)
        self.layoutTrack.addWidget(self.canvas_2_displace)

        self.track_Widget = QtWidgets.QWidget()
        self.track_Widget.setLayout(self.layoutTrack)

        #---------------------------------------------------------------------
        # TAB 3 PLASMA
        #---------------------------------------------------------------------
        self.figure_3 = Figure()
        self.canvas_3 = FigureCanvas(self.figure_3)
        self.plt_toolbar_3 = NavigationToolbar(self.canvas_3, self)
        self.plt_toolbar_3.setFixedHeight(self.toolBar_height)

        layoutTabSettingsCheck = QtWidgets.QGridLayout()
        self.plasma_names = ["Bx","Bx_av","Bx_trans","ne","ne_av","ne_trans","ni","Lx_av","Lx_trans","jx_av","jx_trans","px","pθ_av","pθ_trans", "Jx","Jx_trans","Jθ", "Jθ_trans","rho", "rho_trans","Ekin", "Ekin_trans"]
        self.plasma_check_list = []
        N_plasma = len(self.plasma_names)
        for i, name in enumerate(self.plasma_names):
            plasma_CHECK = QtWidgets.QCheckBox(name)
            self.plasma_check_list.append(plasma_CHECK)

            row = int(i>=N_plasma//2)
            col = i
            if row>0: col = i-N_plasma//2
            layoutTabSettingsCheck.addWidget(plasma_CHECK, row,col)

        # layoutTabSettingsCheck.addStretch(50)
        layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)


        self.plasma_check_list[1].setChecked(True)
        # self.Ey_CHECK.setChecked(True)
        self.plasma_check_list[4].setChecked(True)

        layoutTabSettingsCheck.setSpacing(0)

        self.plasma_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plasma_time_SLIDER.setRange(0,1)
        self.plasma_xcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.plasma_xcut_SLIDER.setRange(0,1)

        self.plasma_time_EDIT = QtWidgets.QLineEdit("0")
        self.plasma_time_EDIT.setValidator(self.float_validator)
        self.plasma_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC
        self.plasma_xcut_EDIT = QtWidgets.QLineEdit("0")
        self.plasma_xcut_EDIT.setValidator(self.float_validator)
        self.plasma_xcut_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.plasma_play_time_BUTTON = QtWidgets.QPushButton("Play")
        layoutTimeSlider = self.creatPara("t/t0=", self.plasma_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.plasma_time_SLIDER)
        layoutTimeSlider.addWidget(self.plasma_play_time_BUTTON)

        layoutXcutSlider = self.creatPara("x/𝝀=", self.plasma_xcut_EDIT)
        layoutXcutSlider.addWidget(self.plasma_xcut_SLIDER)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addLayout(layoutXcutSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_3)

        self.plasma_groupBox = QtWidgets.QGroupBox("Plasma Diagnostics")
        self.plasma_groupBox.setFixedHeight(int(210*self.resolution_scaling))
        # self.plasma_groupBox.setMaximumWidth(400)
        self.plasma_groupBox.setLayout(layoutTabSettings)

        self.layoutPlasma = QtWidgets.QVBoxLayout()
        self.layoutPlasma.addWidget(self.plasma_groupBox)
        self.layoutPlasma.addWidget(self.canvas_3)

        self.plasma_Widget = QtWidgets.QWidget()
        self.plasma_Widget.setLayout(self.layoutPlasma)

        #---------------------------------------------------------------------
        # TAB 4 COMPA
        #---------------------------------------------------------------------
        self.figure_4_scalar = Figure()
        self.canvas_4_scalar = FigureCanvas(self.figure_4_scalar)
        self.plt_toolbar_4_scalar = NavigationToolbar(self.canvas_4_scalar)
        self.plt_toolbar_4_scalar.setFixedHeight(self.toolBar_height)
        self.ax4_scalar = self.figure_4_scalar.add_subplot(1,1,1)
        self.ax4_scalar.grid()
        self.ax4_scalar.set_xlabel("t/t0",fontsize=14)
        self.figure_4_scalar.tight_layout()

        self.figure_4_plasma = Figure()
        self.canvas_4_plasma = FigureCanvas(self.figure_4_plasma)
        self.plt_toolbar_4_plasma = NavigationToolbar(self.canvas_4_plasma)
        self.plt_toolbar_4_plasma.setFixedHeight(self.toolBar_height)
        self.ax4_plasma1 = self.figure_4_plasma.add_subplot(2,1,1)
        self.ax4_plasma2 = self.figure_4_plasma.add_subplot(2,1,2)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))

        self.cbar_4_plasma2 = self.figure_4_plasma.colorbar(sm, ax=self.ax4_plasma2, pad=0.01)
        self.cbar_4_plasma1 = self.figure_4_plasma.colorbar(sm, ax=self.ax4_plasma1, pad=0.01)
        self.figure_4_plasma.tight_layout()

        #-------------- MAIN Groupbox -----------------
        self.compa_load_sim_BUTTON = QtWidgets.QPushButton('Open Comparison')
        self.compa_load_sim_BUTTON.setFixedWidth(int(150*self.resolution_scaling))
        self.compa_load_status_LABEL = QtWidgets.QLabel("")
        self.compa_load_status_LABEL.setStyleSheet("color: black")
        self.compa_load_status_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        self.compa_load_status_LABEL.setFont(self.medium_bold_FONT)
        self.compa_sim_directory_name_LABEL = QtWidgets.QLabel("")
        self.compa_sim_directory_name_LABEL.setFont(self.medium_bold_FONT)
        self.compa_sim_directory_name_LABEL.adjustSize()
        self.compa_groupBox = QtWidgets.QGroupBox("Settings")
        self.compa_groupBox.setFixedHeight(int(60*self.resolution_scaling))
        layoutCompaLoadSim =  QtWidgets.QHBoxLayout()
        layoutCompaLoadSim.addWidget(self.compa_load_sim_BUTTON)
        layoutCompaLoadSim.addWidget(self.compa_load_status_LABEL)
        layoutCompaLoadSim.addWidget(self.compa_sim_directory_name_LABEL)
        boxLayout_settings = QtWidgets.QVBoxLayout()
        boxLayout_settings.addLayout(layoutCompaLoadSim)
        self.diag_type_BOX = QtWidgets.QComboBox()
        self.diag_type_BOX.addItem("Scalar")
        self.diag_type_BOX.addItem("Plasma")
        self.diag_type_BOX.addItem("Binning")
        self.diag_type_BOX.addItem("Intensity")
        self.diag_type_BOX.addItem("Track")
        layoutCompaLoadSim.addWidget(self.diag_type_BOX)
        self.compa_groupBox.setLayout(boxLayout_settings)


        #-------------- COMPA SCALAR Groupbox -----------------#
        layoutCompaTabSettingsCheck = QtWidgets.QHBoxLayout()
        self.compa_scalar_check_list = []
        for name in self.scalar_names:
            compa_scalar_CHECK = QtWidgets.QCheckBox(name)
            self.compa_scalar_check_list.append(compa_scalar_CHECK)
            layoutCompaTabSettingsCheck.addWidget(compa_scalar_CHECK)


        layoutTabSettingsCompaScalar = QtWidgets.QVBoxLayout()
        layoutTabSettingsCompaScalar.addLayout(layoutCompaTabSettingsCheck)
        layoutTabSettingsCompaScalar.addWidget(self.plt_toolbar_4_scalar)
        self.compa_scalar_groupBox = QtWidgets.QGroupBox("Compa Scalar Diagnostics")
        self.compa_scalar_groupBox.setFixedHeight(int(110*self.resolution_scaling))
        self.compa_scalar_groupBox.setLayout(layoutTabSettingsCompaScalar)

        #-------------- COMPA BINNING Groupbox -----------------#
        self.figure_4_binning = Figure()
        self.canvas_4_binning = FigureCanvas(self.figure_4_binning)
        self.plt_toolbar_4_binning = NavigationToolbar(self.canvas_4_binning)
        self.plt_toolbar_4_binning.setFixedHeight(self.toolBar_height)
        self.ax4_binning = self.figure_4_binning.add_subplot(1,1,1)
        self.figure_4_binning.tight_layout()
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        self.compa_binning_colorbar = self.figure_4_binning.colorbar(sm, ax=self.ax4_binning, pad=0.01)
        layoutTabSettingsCompaBinning = QtWidgets.QVBoxLayout()
        self.compa_binning_diag_name_EDIT = QtWidgets.QLineEdit("ekin")
        layoutTabSettingsCompaBinning.addWidget(self.compa_binning_diag_name_EDIT)

        self.compa_binning_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_binning_time_SLIDER.setRange(0,1)
        layoutCompaBinningTimeSlider = self.creatPara("t/t0=", self.compa_binning_time_SLIDER ,adjust_label=True)

        layoutTabSettingsCompaBinning.addLayout(layoutCompaBinningTimeSlider)

        layoutTabSettingsCompaBinning.addWidget(self.plt_toolbar_4_binning)
        self.compa_binning_groupBox = QtWidgets.QGroupBox("Particle Binning Diagnostics")
        self.compa_binning_groupBox.setFixedHeight(int(150*self.resolution_scaling))
        self.compa_binning_groupBox.setLayout(layoutTabSettingsCompaBinning)

        # self.layoutCompaBinning = QtWidgets.QVBoxLayout()
        # self.layoutCompaBinning.addWidget(self.compa_binning_groupBox)
        # self.layoulayoutCompaBinningtBinning.addWidget(self.canvas_4_binning)
        # self.compa_binning_Widget = QtWidgets.QWidget()
        # self.compa_binning_Widget.setLayout(self.layoutCompaBinning)
        #-------------- COMPA PLASMA Groupbox -----------------#
        layoutCompaTabSettingsCheck = QtWidgets.QGridLayout()
        self.compa_plasma_check_list = []
        for i,name in enumerate(self.plasma_names):
            compa_plasma_RADIO = QtWidgets.QRadioButton(name)
            self.compa_plasma_check_list.append(compa_plasma_RADIO)
            row = int(i>=N_plasma//2)
            col = i
            if row>0: col = i-N_plasma//2
            layoutCompaTabSettingsCheck.addWidget(compa_plasma_RADIO, row,col)
            
            # layoutCompaTabSettingsCheck.addWidget(compa_plasma_RADIO)

        self.compa_plasma_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_plasma_time_SLIDER.setRange(0,1)
        self.compa_plasma_xcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_plasma_xcut_SLIDER.setRange(0,1)

        self.compa_plasma_time_EDIT = QtWidgets.QLineEdit("0")
        self.compa_plasma_time_EDIT.setValidator(self.float_validator)
        self.compa_plasma_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC
        self.compa_plasma_xcut_EDIT = QtWidgets.QLineEdit("0")
        self.compa_plasma_xcut_EDIT.setValidator(self.float_validator)
        self.compa_plasma_xcut_EDIT.setMaximumWidth(70) #42 FOR TOWER PC

        self.compa_plasma_play_time_BUTTON = QtWidgets.QPushButton("Play")
        layoutCompaTimeSlider = self.creatPara("t/t0=", self.compa_plasma_time_EDIT ,adjust_label=True)
        layoutCompaTimeSlider.addWidget(self.compa_plasma_time_SLIDER)
        layoutCompaTimeSlider.addWidget(self.compa_plasma_play_time_BUTTON)

        layoutCompaXcutSlider = self.creatPara("x/𝝀=", self.compa_plasma_xcut_EDIT)
        layoutCompaXcutSlider.addWidget(self.compa_plasma_xcut_SLIDER)

        layoutTabSettingsCompaPlasma = QtWidgets.QVBoxLayout()
        layoutTabSettingsCompaPlasma.addLayout(layoutCompaTabSettingsCheck)
        layoutTabSettingsCompaPlasma.addLayout(layoutCompaTimeSlider)
        layoutTabSettingsCompaPlasma.addLayout(layoutCompaXcutSlider)
        layoutTabSettingsCompaPlasma.addWidget(self.plt_toolbar_4_plasma)
        self.compa_plasma_groupBox = QtWidgets.QGroupBox("Compa Plasma Diagnostics")
        self.compa_plasma_groupBox.setFixedHeight(int(210*self.resolution_scaling))
        self.compa_plasma_groupBox.setLayout(layoutTabSettingsCompaPlasma)

        #-------------- COMPA INTENSITY Groupbox -----------------#
        self.figure_4_intensity = Figure()
        self.canvas_4_intensity = FigureCanvas(self.figure_4_intensity)
        self.plt_toolbar_4_intensity = NavigationToolbar(self.canvas_4_intensity, self)
        self.plt_toolbar_4_intensity.setFixedHeight(self.toolBar_height)
        
        self.figure_4_intensity_time = Figure()
        self.canvas_4_intensity_time = FigureCanvas(self.figure_4_intensity_time)
        self.plt_toolbar_4_intensity_time = NavigationToolbar(self.canvas_4_intensity_time, self)
        self.plt_toolbar_4_intensity_time.setFixedHeight(self.toolBar_height)
        self.canvas_4_intensity_time.hide()
        self.plt_toolbar_4_intensity_time.hide()
        
        self.figure_4_intensity.tight_layout()
        self.figure_4_intensity_time.tight_layout()

                
        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        self.intensity_names = ["Ex","Ey"]
        self.compa_intensity_check_list = []
        for i, name in enumerate(self.intensity_names):
            intensity_CHECK = QtWidgets.QRadioButton(name)
            self.compa_intensity_check_list.append(intensity_CHECK)
            if i in [3,6,8]:
                separator1 = QtWidgets.QFrame()
                separator1.setFrameShape(QtWidgets.QFrame.VLine)
                separator1.setLineWidth(1)
                layoutTabSettingsCheck.addWidget(separator1)
            layoutTabSettingsCheck.addWidget(intensity_CHECK)
        
        # self.compa_intensity_check_list[1].setChecked(True)
        
        layoutTabSettingsCheck.setSpacing(20)
        layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)
        
        self.compa_intensity_spatial_time_BOX = QtWidgets.QComboBox()
        self.compa_intensity_spatial_time_BOX.addItem("Spatial distributions")
        self.compa_intensity_spatial_time_BOX.addItem("Time distributions")
        
        self.compa_intensity_follow_laser_CHECK = QtWidgets.QCheckBox("Follow Laser")
        self.compa_intensity_follow_laser_CHECK.setChecked(True)
        self.compa_intensity_use_vg_CHECK = QtWidgets.QCheckBox("Use group vel")

        # layoutTabSettingsCutDirection = QtWidgets.QHBoxLayout()
        layoutTabSettingsCheck.addWidget(self.compa_intensity_spatial_time_BOX)
        layoutTabSettingsCheck.addWidget(self.compa_intensity_follow_laser_CHECK)
        layoutTabSettingsCheck.addWidget(self.compa_intensity_use_vg_CHECK)
    
        layoutTabSettingsCompaIntensity = QtWidgets.QVBoxLayout()
        layoutTabSettingsCompaIntensity.addLayout(layoutTabSettingsCheck)
        # layoutTabSettingsCompaIntensity.addLayout(layoutTabSettingsCutDirection)
        
        self.compa_intensity_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_intensity_time_SLIDER.setRange(0,1)
        self.compa_intensity_xcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_intensity_xcut_SLIDER.setRange(0,1)
        
        self.compa_intensity_time_EDIT = QtWidgets.QLineEdit("0")
        
        self.compa_intensity_time_EDIT.setValidator(self.float_validator)
        self.compa_intensity_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC
        
        self.compa_intensity_xcut_EDIT = QtWidgets.QLineEdit("0")
        self.compa_intensity_xcut_EDIT.setValidator(self.float_validator)
        self.compa_intensity_xcut_EDIT.setMaximumWidth(70)
        
        self.compa_intensity_play_time_BUTTON = QtWidgets.QPushButton("Play")
        self.compa_intensity_play_time_BUTTON.setMinimumHeight(15)
        layoutTimeSlider = self.creatPara("t/t0=", self.compa_intensity_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.compa_intensity_time_SLIDER)
        layoutTimeSlider.addWidget(self.compa_intensity_play_time_BUTTON)
        
        layoutXcutSlider = self.creatPara("x/𝝀=", self.compa_intensity_xcut_EDIT)
        layoutXcutSlider.addWidget(self.compa_intensity_xcut_SLIDER)
        
        layoutTabSettingsCompaIntensity.addLayout(layoutTimeSlider)
        layoutTabSettingsCompaIntensity.addLayout(layoutXcutSlider)
        layoutTabSettingsCompaIntensity.addWidget(self.plt_toolbar_4_intensity)
        layoutTabSettingsCompaIntensity.addWidget(self.plt_toolbar_4_intensity_time)
        
        self.compa_intensity_groupBox = QtWidgets.QGroupBox("Compa Intensity Diagnostics")
        self.compa_intensity_groupBox.setFixedHeight(int(170*self.resolution_scaling))
        self.compa_intensity_groupBox.setLayout(layoutTabSettingsCompaIntensity)
        
        #-------------- COMPA TRACK Groupbox -----------------#
        self.figure_4_track = Figure()
        self.canvas_4_track = FigureCanvas(self.figure_4_track)
        self.plt_toolbar_4_track = NavigationToolbar(self.canvas_4_track, self)
        self.plt_toolbar_4_track.setFixedHeight(self.toolBar_height)
        self.figure_4_track.tight_layout()
        self.ax4_track = self.figure_4_track.add_subplot(1,1,1)
        self.ax4_track.grid()
        

        self.compa_track_file_BOX = QtWidgets.QComboBox()
        self.compa_track_file_BOX.addItem("track_eon")
        self.compa_track_file_BOX.addItem("track_eon_full")
        self.compa_track_file_BOX.addItem("track_eon_dense")
        self.compa_track_file_BOX.addItem("track_eon_net")

        layoutTabSettingsTrackFile = QtWidgets.QHBoxLayout()
        self.compa_track_Npart_EDIT = QtWidgets.QLineEdit("10")
        self.compa_track_Npart_EDIT.setValidator(self.int_validator)
        self.compa_track_Npart_EDIT.setMaximumWidth(45)

        self.compa_track_update_offset_CHECK = QtWidgets.QCheckBox("Update offsets")
        
        self.compa_track_pannel_BOX = QtWidgets.QComboBox()
        self.compa_track_pannel_BOX.addItem("Angular Momentum")
        self.compa_track_pannel_BOX.addItem("Displacements")
        # self.track_pannel_BOX.addItem("particle trajectories")

        layoutNpart = self.creatPara("Npart=", self.compa_track_Npart_EDIT,adjust_label=True)

        layoutTabSettingsTrackFile.addLayout(layoutNpart)
        layoutTabSettingsTrackFile.addWidget(self.compa_track_file_BOX)
        layoutTabSettingsTrackFile.addWidget(self.compa_track_update_offset_CHECK)
        layoutTabSettingsTrackFile.addWidget(self.compa_track_pannel_BOX)

        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsTrackFile)

        self.compa_track_time_EDIT = QtWidgets.QLineEdit("0")
        self.compa_track_time_EDIT.setValidator(self.float_validator)
        self.compa_track_time_EDIT.setMaximumWidth(70)
        self.compa_track_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.compa_track_time_SLIDER.setRange(0,1)
        self.compa_track_play_time_BUTTON = QtWidgets.QPushButton("Play")

        layoutTimeSlider = self.creatPara("t/t0=", self.compa_track_time_EDIT,adjust_label=True)
        layoutTimeSlider.addWidget(self.compa_track_time_SLIDER)
        layoutTimeSlider.addWidget(self.compa_track_play_time_BUTTON)
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_4_track)

        self.compa_track_groupBox = QtWidgets.QGroupBox("Track Particles Diagnostic")
        self.compa_track_groupBox.setFixedHeight(int(125*self.resolution_scaling))
        self.compa_track_groupBox.setLayout(layoutTabSettings)

        # self.compa_layoutTrack = QtWidgets.QVBoxLayout()
        # self.compa_layoutTrack.addWidget(self.track_groupBox)
        # self.compa_layoutTrack.addWidget(self.canvas_4_track)

        # self.compa_track_Widget = QtWidgets.QWidget()
        # self.compa_track_Widget.setLayout(self.compa_layoutTrack)
        
        #---- add to layout ----# 
        self.layoutCompa = QtWidgets.QVBoxLayout()
        self.layoutCompa.addWidget(self.compa_groupBox)
        self.layoutCompa.addWidget(self.compa_scalar_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_scalar)
        self.layoutCompa.addWidget(self.compa_plasma_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_plasma)
        self.layoutCompa.addWidget(self.compa_binning_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_binning)
        
        self.layoutCompa.addWidget(self.compa_intensity_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_intensity)
        self.layoutCompa.addWidget(self.canvas_4_intensity_time)
        
        self.layoutCompa.addWidget(self.compa_track_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_track)

        
        self.compa_plasma_groupBox.hide()
        self.canvas_4_plasma.hide()
        self.compa_binning_groupBox.hide()
        self.canvas_4_binning.hide()
        self.compa_intensity_groupBox.hide()
        self.canvas_4_intensity.hide()
        self.canvas_4_intensity_time.hide()
        self.compa_track_groupBox.hide()
        self.canvas_4_track.hide()

        self.compa_Widget = QtWidgets.QWidget()
        self.compa_Widget.setLayout(self.layoutCompa)
        
        #---------------------------------------------------------------------
        # TAB 5 BINNING
        #---------------------------------------------------------------------
        self.figure_5 = Figure()
        self.canvas_5 = FigureCanvas(self.figure_5)
        self.plt_toolbar_5 = NavigationToolbar(self.canvas_5)
        self.plt_toolbar_5.setFixedHeight(self.toolBar_height)
        self.ax5 = self.figure_5.add_subplot(1,1,1)
        self.figure_5.tight_layout()
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        self.binning_colorbar = self.figure_5.colorbar(sm, ax=self.ax5, pad=0.01)

        layoutTabSettingsBinning = QtWidgets.QVBoxLayout()
        self.binning_diag_name_EDIT = QtWidgets.QLineEdit("ekin")
        self.binning_log_CHECK = QtWidgets.QCheckBox("Log10")

        layoutTabSettingsBinningNameLog = QtWidgets.QHBoxLayout()
        layoutTabSettingsBinningNameLog.addWidget(self.binning_diag_name_EDIT)
        layoutTabSettingsBinningNameLog.addWidget(self.binning_log_CHECK)
        layoutTabSettingsBinning.addLayout(layoutTabSettingsBinningNameLog)


        self.binning_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.binning_time_SLIDER.setRange(0,1)
        layoutBinningTimeSlider = self.creatPara("t/t0=", self.binning_time_SLIDER ,adjust_label=True)

        layoutTabSettingsBinning.addLayout(layoutBinningTimeSlider)


        layoutTabSettingsBinning.addWidget(self.plt_toolbar_5)
        self.binning_groupBox = QtWidgets.QGroupBox("Particle Binning Diagnostics")
        self.binning_groupBox.setFixedHeight(int(150*self.resolution_scaling))
        self.binning_groupBox.setLayout(layoutTabSettingsBinning)

        self.layoutBinning = QtWidgets.QVBoxLayout()
        self.layoutBinning.addWidget(self.binning_groupBox)
        self.layoutBinning.addWidget(self.canvas_5)
        self.binning_Widget = QtWidgets.QWidget()
        self.binning_Widget.setLayout(self.layoutBinning)

        #---------------------------------------------------------------------
        # TAB 6 INTENSITY
        #---------------------------------------------------------------------
        self.figure_6 = Figure()
        self.canvas_6 = FigureCanvas(self.figure_6)
        self.plt_toolbar_6 = NavigationToolbar(self.canvas_6, self)
        self.plt_toolbar_6.setFixedHeight(self.toolBar_height)
        
        self.figure_6_time = Figure()
        self.canvas_6_time = FigureCanvas(self.figure_6_time)
        self.plt_toolbar_6_time = NavigationToolbar(self.canvas_6_time, self)
        self.plt_toolbar_6_time.setFixedHeight(self.toolBar_height)
        self.canvas_6_time.hide()
        self.plt_toolbar_6_time.hide()
                       
        layoutTabSettingsCheck = QtWidgets.QHBoxLayout()
        self.intensity_names =["Ex","Ey"]
        self.intensity_check_list = []
        for i, name in enumerate(self.intensity_names):
            intensity_CHECK = QtWidgets.QRadioButton(name)
            self.intensity_check_list.append(intensity_CHECK)
            layoutTabSettingsCheck.addWidget(intensity_CHECK)

        # self.intensity_check_list[1].setChecked(True)
        
        layoutTabSettingsCheck.setSpacing(20)
        layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)
        
        self.intensity_spatial_time_BOX = QtWidgets.QComboBox()
        self.intensity_spatial_time_BOX.addItem("Spatial distributions")
        self.intensity_spatial_time_BOX.addItem("Time distributions")
        
        self.intensity_follow_laser_CHECK = QtWidgets.QCheckBox("Follow Laser")
        self.intensity_follow_laser_CHECK.setChecked(True)
        self.intensity_use_vg_CHECK = QtWidgets.QCheckBox("Use group vel")

        
        layoutTabSettingsCheck.addWidget(self.intensity_spatial_time_BOX)
        layoutTabSettingsCheck.addWidget(self.intensity_follow_laser_CHECK)
        layoutTabSettingsCheck.addWidget(self.intensity_use_vg_CHECK)

        
        layoutTabSettings = QtWidgets.QVBoxLayout()
        layoutTabSettings.addLayout(layoutTabSettingsCheck)        
        
        self.intensity_time_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_time_SLIDER.setRange(0,1)
        self.intensity_xcut_SLIDER = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_xcut_SLIDER.setRange(0,1)
        
        self.intensity_time_EDIT = QtWidgets.QLineEdit("0")
        
        self.intensity_time_EDIT.setValidator(self.float_validator)
        self.intensity_time_EDIT.setMaximumWidth(70) #42 FOR TOWER PC
        
        self.intensity_xcut_EDIT = QtWidgets.QLineEdit("0")
        self.intensity_xcut_EDIT.setValidator(self.float_validator)
        self.intensity_xcut_EDIT.setMaximumWidth(70)
        
        self.intensity_play_time_BUTTON = QtWidgets.QPushButton("Play")
        self.intensity_play_time_BUTTON.setMinimumHeight(15)
        layoutTimeSlider = self.creatPara("t/t0=", self.intensity_time_EDIT ,adjust_label=True)
        layoutTimeSlider.addWidget(self.intensity_time_SLIDER)
        layoutTimeSlider.addWidget(self.intensity_play_time_BUTTON)
        
        layoutXcutSlider = self.creatPara("x/𝝀=", self.intensity_xcut_EDIT)
        layoutXcutSlider.addWidget(self.intensity_xcut_SLIDER)
        
        layoutTabSettings.addLayout(layoutTimeSlider)
        layoutTabSettings.addLayout(layoutXcutSlider)
        layoutTabSettings.addWidget(self.plt_toolbar_6)
        layoutTabSettings.addWidget(self.plt_toolbar_6_time)
        
        self.intensity_groupBox = QtWidgets.QGroupBox("Intensity Diagnostics")
        self.intensity_groupBox.setFixedHeight(int(170*self.resolution_scaling))
        self.intensity_groupBox.setLayout(layoutTabSettings)
        
        self.layoutIntensity = QtWidgets.QVBoxLayout()
        self.layoutIntensity.addWidget(self.intensity_groupBox)
        self.layoutIntensity.addWidget(self.canvas_6)
        self.layoutIntensity.addWidget(self.canvas_6_time)
        
        self.intensity_Widget = QtWidgets.QWidget()
        self.intensity_Widget.setLayout(self.layoutIntensity)



        #---------------------------------------------------------------------
        # TAB 7 TORNADO
        #---------------------------------------------------------------------
        self.tornado_Widget = QtWidgets.QWidget()

        self.layoutTornado = QtWidgets.QVBoxLayout()
        self.tornado_groupBox = QtWidgets.QGroupBox("Infos")
        self.tornado_groupBox.setFixedHeight(100)

        tornado_group_box_layout = QtWidgets.QHBoxLayout()

        self.tornado_last_update_LABEL = QtWidgets.QLabel("LOADING...")
        self.tornado_last_update_LABEL.setFont(QFont('Arial', 12))
        tornado_group_box_layout.addWidget(self.tornado_last_update_LABEL)

        self.tornado_refresh_BUTTON = QtWidgets.QPushButton("Refresh")
        tornado_group_box_layout.addWidget(self.tornado_refresh_BUTTON)

        self.tornado_groupBox.setLayout(tornado_group_box_layout)
        self.layoutTornado.addWidget(self.tornado_groupBox)

        self.tornado_Widget.setLayout(self.layoutTornado)

        #---------------------------------------------------------------------
        # ADD TABS TO QTabWidget
        #---------------------------------------------------------------------
        self.programm_TABS = QtWidgets.QTabWidget()
        self.programm_TABS.setMovable(True)
        self.programm_TABS.setTabsClosable(True)

        layoutTabsAndLeft = QtWidgets.QHBoxLayout()
        layoutTabsAndLeft.addLayout(boxLayoutLEFT)
        layoutTabsAndLeft.addWidget(self.programm_TABS)


        settings_width = self.settings_groupBox.geometry().width()
        window_width = self.geometry().width()
        logo_width = window_width-settings_width
        
        # Create a Matplotlib figure and canvas
        self.figure_verlet = Figure()
        self.canvas_verlet = FigureCanvas(self.figure_verlet)
        self.ax_verlet = self.figure_verlet.add_subplot(1, 1, 1)
        self.ax_verlet.set_axis_off()
        # self.ax_verlet.grid()
        
        #Smilei Logo as Button + matplotlib figure for Verlet mini-game
        smilei_icon_path = os.environ["SMILEI_QT"] + "\\Ressources\\smilei_gui_svg_v3.png"
        self.SMILEI_ICON_LABEL = QtGui.QIcon(smilei_icon_path)
        self.verlet_container = QtWidgets.QWidget(self)
        self.smilei_icon_BUTTON = QtWidgets.QPushButton(self.verlet_container)
        self.smilei_icon_BUTTON.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.smilei_icon_BUTTON.setIcon(self.SMILEI_ICON_LABEL)
        self.smilei_icon_BUTTON.setIconSize(QtCore.QSize(400, 400))  # Adjust as needed
        self.smilei_icon_BUTTON.setStyleSheet("QPushButton { border: none; background: transparent; }")
        self.smilei_icon_BUTTON.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        
        # Used QStackedLayout to overlay the button on top of the figure
        stackedLayout = QtWidgets.QStackedLayout(self.verlet_container)
        stackedLayout.addWidget(self.canvas_verlet)  
        stackedLayout.addWidget(self.smilei_icon_BUTTON) 
        stackedLayout.setStackingMode(QtWidgets.QStackedLayout.StackAll)  
        
        self.N_verlet_circles = 50

        self.verlet_window = 200
        self.verlet_size = 10
        self.verlet_circles = []
        self.verlet_POS = np.zeros((self.N_verlet_circles,2))
        
        self.verlet_spawn_circle = True
        self.verlet_circles_colors = ["#646464","#0672ba","#7ecdff","#c5dcea"]
        self.verlet_circles_size = []
        
        for i in range(self.N_verlet_circles):
            circle_size = np.random.randint(3,14)
            x0,y0 = np.random.uniform(1, self.verlet_window), np.random.uniform(1, self.verlet_window)
            circle = patches.Circle((x0, y0), 
                                    circle_size, fc=np.random.choice(self.verlet_circles_colors),ec="k")
            circle.set_visible(False)

            self.verlet_POS[i] = [x0,y0]
            self.verlet_circles_size.append(circle_size)
            self.ax_verlet.add_patch(circle)
            self.verlet_circles.append(circle)
        self.verlet_OLD_POS = self.verlet_POS

        self.ax_verlet.set_xlim(0,self.verlet_window)
        self.ax_verlet.set_ylim(0,self.verlet_window)
        
        verlet_refresh_time_s = 0.05 #s
        self.verlet_update_TIMER = QtCore.QTimer()
        
        self.verlet_mouse_pos = np.array([self.verlet_window/2,self.verlet_window])
        self.canvas_verlet.mpl_connect("motion_notify_event", self.onMouseMoveVerlet)
        
        self.verlet_update_interval = 20 #ms
        self.verlet_spawn_interval = 2000 #ms
        self.verlet_update_TIMER.setInterval(self.verlet_update_interval) #in ms
        self.verlet_update_TIMER.timeout.connect(self.call_ThreadUpdateVerlet)
        # self.verlet_update_TIMER.start()
        
        self.verlet_spawn_TIMER = QtCore.QTimer()
        self.verlet_spawn_TIMER.setInterval(self.verlet_spawn_interval) #in ms
        self.verlet_spawn_TIMER.timeout.connect(self.onVerletSpawn)
        # self.verlet_spawn_TIMER.start()
        self.is_verlet_timer_active = False
        self.figure_verlet.tight_layout()

        
        # Set main layout
        mainLayout = QtWidgets.QVBoxLayout(self)
        mainLayout.addWidget(self.verlet_container)
        self.programm_TABS.setLayout(mainLayout)
        
        # Ensure the button is always on top
        self.smilei_icon_BUTTON.raise_()

        # self.programm_TABS.setLayout(layout)

        layoutBottom = QtWidgets.QHBoxLayout()
        layoutBottom.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)


        self.general_info_LABEL = QtWidgets.QLabel(f"Version: {self.SCRIPT_VERSION}  |  Memory: {self.MEMORY().used*100/self.MEMORY().total:.0f}% | Storage: {self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30):.1f} Go | {self.COPY_RIGHT}")


        layoutMAIN = QtWidgets.QVBoxLayout()
        layoutMAIN.addLayout(layoutTabsAndLeft)
        layoutMAIN.addWidget(self.general_info_LABEL,alignment=QtCore.Qt.AlignRight)

        layoutMAIN.setContentsMargins(0, 0, 0, 0)

        widget = QtWidgets.QWidget()
        widget.setLayout(layoutMAIN)
        self.setCentralWidget(widget)

        #=====================================================================
        # CONNECTS
        #=====================================================================
        self.load_sim_BUTTON.clicked.connect(self.onOpenSim)
        self.compa_load_sim_BUTTON.clicked.connect(self.onOpenCompaSim)

        self.sim_directory_name_LABEL.mousePressEvent = self.onCopySimName



        for i in range(len(self.scalar_check_list)):
            self.scalar_check_list[i].clicked.connect(partial(self.onUpdateTabScalar,i,is_compa=False))

        for i in range(len(self.fields_check_list)):
            self.fields_check_list[i].clicked.connect(partial(self.onUpdateTabFields,i))

        self.fields_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabFields(100))
        self.fields_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabFields(100))
        self.fields_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabFields(101))
        self.fields_xcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabFields(200))
        self.fields_xcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabFields(200))
        self.fields_xcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabFields(201))

        self.fields_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabFields(1000))
        self.sim_cut_direction_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabFields(-100))

        self.track_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabTrack(100))
        self.track_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabTrack(100))
        self.track_time_EDIT.returnPressed.connect(lambda:  self.onUpdateTabTrack(101))
        self.track_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabTrack(1000))
        self.track_file_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabTrack(-1))
        self.track_Npart_EDIT.returnPressed.connect(lambda:  self.onUpdateTabTrack(-1))
        self.track_update_offset_CHECK.clicked.connect(lambda:  self.onUpdateTabTrack(-1))
        self.track_pannel_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabTrack(5000))
        #--------------------------------
        self.compa_track_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabCompaTrack(100))
        self.compa_track_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabCompaTrack(100))
        self.compa_track_time_EDIT.returnPressed.connect(lambda:  self.onUpdateTabCompaTrack(101))
        self.compa_track_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabCompaTrack(1000))
        self.compa_track_file_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabCompaTrack(-1))
        self.compa_track_Npart_EDIT.returnPressed.connect(lambda:  self.onUpdateTabCompaTrack(-1))
        self.compa_track_update_offset_CHECK.clicked.connect(lambda:  self.onUpdateTabCompaTrack(-1))
        self.compa_track_pannel_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabCompaTrack(5000))


        for i in range(len(self.plasma_check_list)):
            self.plasma_check_list[i].clicked.connect(partial(self.onUpdateTabPlasma,i))


        self.plasma_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabPlasma(100))
        self.plasma_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabPlasma(100))
        self.plasma_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabPlasma(101))
        self.plasma_xcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabPlasma(200))
        self.plasma_xcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabPlasma(200))
        self.plasma_xcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabPlasma(201))
        self.plasma_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabPlasma(1000))

        self.diag_type_BOX.currentIndexChanged.connect(self.onUpdateTabCompa)

        for i in range(len(self.scalar_check_list)):
            self.compa_scalar_check_list[i].clicked.connect(partial(self.onUpdateTabScalar,i,is_compa=True))

        for i in range(len(self.compa_plasma_check_list)):
            self.compa_plasma_check_list[i].clicked.connect(partial(self.onUpdateTabCompaPlasma,i))

        self.compa_plasma_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabCompaPlasma(100))
        self.compa_plasma_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabCompaPlasma(100))
        self.compa_plasma_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabCompaPlasma(101))
        self.compa_plasma_xcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabCompaPlasma(200))
        self.compa_plasma_xcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabCompaPlasma(200))
        self.compa_plasma_xcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabCompaPlasma(201))
        self.compa_plasma_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabCompaPlasma(1000))

        self.compa_binning_diag_name_EDIT.returnPressed.connect(partial(self.onUpdateTabBinning, id=0, is_compa=True))
        self.compa_binning_time_SLIDER.sliderMoved.connect(partial(self.onUpdateTabBinning, id=100, is_compa=True))


        self.binning_diag_name_EDIT.returnPressed.connect(lambda: self.onUpdateTabBinning(0))
        self.binning_log_CHECK.clicked.connect(lambda: self.onUpdateTabBinning(100))
        self.binning_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabBinning(100))

        for i in range(len(self.intensity_check_list)):
            self.intensity_check_list[i].clicked.connect(partial(self.onUpdateTabIntensity,i))

        self.intensity_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabIntensity(100))
        self.intensity_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabIntensity(100))
        self.intensity_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabIntensity(101))
        self.intensity_xcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabIntensity(200))
        self.intensity_xcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabIntensity(200))
        self.intensity_xcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabIntensity(201))

        self.intensity_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabIntensity(1000))
        self.intensity_follow_laser_CHECK.clicked.connect(lambda: self.onUpdateTabIntensity(100))
        self.intensity_spatial_time_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabIntensity(2000))
        self.intensity_use_vg_CHECK.clicked.connect(lambda: self.onUpdateTabIntensity(5000))
        
        #--------------------------------
        for i in range(len(self.intensity_check_list)):
            self.compa_intensity_check_list[i].clicked.connect(partial(self.onUpdateTabCompaIntensity,i))
        self.compa_intensity_time_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabCompaIntensity(100))
        self.compa_intensity_time_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabCompaIntensity(100))
        self.compa_intensity_time_EDIT.returnPressed.connect(lambda: self.onUpdateTabCompaIntensity(101))
        self.compa_intensity_xcut_SLIDER.sliderMoved.connect(lambda: self.onUpdateTabCompaIntensity(200))
        self.compa_intensity_xcut_SLIDER.sliderPressed.connect(lambda: self.onUpdateTabCompaIntensity(200))
        self.compa_intensity_xcut_EDIT.returnPressed.connect(lambda: self.onUpdateTabCompaIntensity(201))

        self.compa_intensity_play_time_BUTTON.clicked.connect(lambda: self.onUpdateTabCompaIntensity(1000))
        self.compa_intensity_follow_laser_CHECK.clicked.connect(lambda: self.onUpdateTabCompaIntensity(100))
        self.compa_intensity_spatial_time_BOX.currentIndexChanged.connect(lambda: self.onUpdateTabCompaIntensity(2000))
        self.compa_intensity_use_vg_CHECK.clicked.connect(lambda: self.onUpdateTabCompaIntensity(5000))


        self.tornado_refresh_BUTTON.clicked.connect(self.call_ThreadDownloadSimJSON)

        #Open and Close Tabs
        self.actionDiagScalar.toggled.connect(lambda: self.onMenuTabs("SCALAR"))
        self.actionDiagFields.toggled.connect(lambda: self.onMenuTabs("FIELDS"))
        self.actionDiagIntensity.toggled.connect(lambda: self.onMenuTabs("INTENSITY"))
        self.actionDiagTrack.toggled.connect(lambda: self.onMenuTabs("TRACK"))
        self.actionDiagPlasma.toggled.connect(lambda: self.onMenuTabs("PLASMA"))
        self.actionDiagBinning.toggled.connect(lambda: self.onMenuTabs("BINNING"))
        self.actionDiagCompa.toggled.connect(lambda: self.onMenuTabs("COMPA"))
        self.actionTornado.toggled.connect(lambda: self.onMenuTabs("TORNADO"))
        self.programm_TABS.tabCloseRequested.connect(self.onCloseTab)


        self.actionOpenSim.triggered.connect(self.onOpenSim)
        self.actionOpenLogs.triggered.connect(self.onOpenLogs)
        self.actionOpenIPython.triggered.connect(self.onOpenIPython)
        self.actionOpenMemory.triggered.connect(self.onOpenMemory)
        self.actionOpenTree.triggered.connect(self.onOpenTree)
        self.actionOpenAllTools.triggered.connect(self.onOpenAllTools)
        
        self.memory_update_TIMER = QtCore.QTimer()
        self.memory_update_TIMER.setInterval(5000) #in ms
        self.memory_update_TIMER.timeout.connect(self.updateInfoLabel)
        self.memory_update_TIMER.start()

        # self.reset.clicked.connect(self.onReset)
        # self.interpolation.currentIndexChanged.connect(self.plot)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle(f"Smilei GUI XDIR ({self.SCRIPT_VERSION_ID})")


        # C:\_DOSSIERS_PC\_STAGE_LULI_\SMILEI_QT_GUI
        self.setWindowIcon(QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\smileiIcon.png"))
        #============================
        # GENERAL VARIABLES
        #============================
        self.INIT_tabScalar = None
        self.INIT_tabFields = None
        self.INIT_tabTrack = None
        self.INIT_tabPlasma = None
        self.INIT_tabCompa = None
        self.INIT_tabCompaIntensity = None
        self.INIT_tabCompaTrack = None

        self.INIT_tabIntensity = None
        self.INIT_tabTornado = True

        self.loop_in_process = False
        self.is_sim_loaded = False
        self.is_compa_sim_loaded = False
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
                    self.deleteItemsOfLayout(item.layout())
        return

    def deleteLayout(self, main_layout, layout_ID_to_del):
        """Delate layout and all its components"""
        layout_item = main_layout.itemAt(layout_ID_to_del)
        #delete sub widgets from layout before deleting layout itself
        self.deleteItemsOfLayout(layout_item.layout())
        main_layout.removeItem(layout_item)
        return

    def closeEvent(self, event):
        confirmation = QtWidgets.QMessageBox.question(self, "Confirmation", "Are you sure you want to close the application?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if confirmation == QtWidgets.QMessageBox.Yes:
            event.accept()  # Close the app
        else:
            event.ignore()  # Don't close the app
        # sys.exit(0)

    def onCopySimName(self, event):
        if self.is_sim_loaded:
            print(self.sim_directory_path)
            clipboard = QtWidgets.QApplication.clipboard()
            text = f"""import os\nimport sys\nimport numpy as np\nimport matplotlib.pyplot as plt\nmodule_dir_happi = 'C:/Users/Jeremy/_LULI_/Smilei'\nsys.path.insert(0, module_dir_happi)\nimport happi\nS = happi.Open('{self.sim_directory_path}')\nl0=2*np.pi\n"""
            clipboard.setText(text)
            # self.sim_directory_name_LABEL.setStyleSheet("background-color: lightgreen")
            # app.processEvents
            # self.sim_directory_name_LABEL.setStyleSheet("background-color: rgba(255, 255, 255, 0)")
            # app.processEvents

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
        self.logs_DIALOG = log_dialog.LogsDialog(self.spyder_default_stdout, app)
        if self.logs_history_STR is not None: self.logs_DIALOG.initHistory(self.logs_history_STR)
        self.logs_DIALOG.show()

    def onOpenIPython(self):
        self.ipython_DIALOG = IPython_dialog.IPythonDialog(self)
        self.ipython_DIALOG.show()
        return

    def onOpenMemory(self):
        self.memory_DIALOG = memory_dialog.MemoryDialog(self)
        self.memory_DIALOG.show()
        return
    def onOpenTree(self):
        self.tree_DIALOG = tree_dialog.TreeDialog(self)
        self.tree_DIALOG.show()
        return
    def onOpenAllTools(self):
        
        self.all_tools_DIALOG = QtWidgets.QMainWindow()
        central_widget = QtWidgets.QWidget(self.all_tools_DIALOG)
        self.all_tools_DIALOG.setCentralWidget(central_widget)
        grid_layout = QtWidgets.QGridLayout()
        height = 800

        widget1 = tree_dialog.TreeDialog(self)
        widget1.setMinimumWidth(int(1.0*height))
        widget2 = memory_dialog.MemoryDialog(self)
        widget3 = IPython_dialog.IPythonDialog(self)
        widget4 = log_dialog.LogsDialog(self.spyder_default_stdout, app)
        if self.logs_history_STR is not None: widget4.initHistory(self.logs_history_STR)

        # widget1 = QtWidgets.QLabel("Widget 1", self)
        # widget2 = QtWidgets.QLabel("Widget 2", self)
        # widget3 = QtWidgets.QLabel("Widget 3", self)
        # widget4 = QtWidgets.QLabel("Widget 4", self)
        # # Add widgets to the layout
        grid_layout.addWidget(widget1, 0, 0)  # Row 0, Column 0
        grid_layout.addWidget(widget2, 0, 1)  # Row 0, Column 1
        grid_layout.addWidget(widget3, 1, 0)  # Row 1, Column 0
        grid_layout.addWidget(widget4, 1, 1)  # Row 1, Column 1


        # Set the window title and dimensions
        self.all_tools_DIALOG.setWindowTitle("Smilei ALL TOOLS")
        self.all_tools_DIALOG.setGeometry(100, 100, int(height*16/9), height)
        
        central_widget.setLayout(grid_layout)
        self.all_tools_DIALOG.show()
        
        # self.all_tools_DIALOG = tools_dialog.ToolsDialog(self, app)
        # self.all_toools_DIALOG.show()
        return
    
    
    def updateInfoLabel(self):
        mem_prc = self.MEMORY().used*100/self.MEMORY().total
        stor_go = self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30)
        memory_str = f"Memory: {mem_prc:.0f}%"
        storage_str = f"Storage: {stor_go:.1f} Go"
        if mem_prc > 85:
            memory_str = f"<font color='red'>Memory: {mem_prc:.0f}%</font>"
        if stor_go<20:
            storage_str = f"<font color='red'>Storage: {stor_go:.1f} Go</font>"

        self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | {memory_str} | {storage_str} | {self.COPY_RIGHT}")
        app.processEvents()
        return
    def onOpenSim(self):
        sim_file_DIALOG= QtWidgets.QFileDialog()
        sim_file_DIALOG.setDirectory(os.environ["SMILEI_CLUSTER"])

        file = str(sim_file_DIALOG.getExistingDirectory(self, "Select Directory"))

        if file !="":
            self.sim_directory_path = file
            self.onLoadSim()

    def onOpenCompaSim(self):
        compa_sim_file_DIALOG= QtWidgets.QFileDialog()
        compa_sim_file_DIALOG.setDirectory(os.environ["SMILEI_CLUSTER"])

        file = str(compa_sim_file_DIALOG.getExistingDirectory(self, "Select Directory"))

        if file !="":
            self.compa_sim_directory_path = file
            self.onLoadCompaSim()

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
            self.is_sim_loaded = True
            self.load_status_LABEL.setText(self.load_sim_status)
        else:
            self.load_sim_status = "Invalid"
            self.load_status_LABEL.setStyleSheet("color: red")
            self.is_sim_loaded = False
            self.load_status_LABEL.setText(self.load_sim_status)
            return

        l0 = 2*pi
        self.w0 = self.S.namelist.w0
        self.xfoc = self.S.namelist.xfoc
        self.a0 = self.S.namelist.a0
        self.Tp = self.S.namelist.Tp
        try:
            self.dx = self.S.namelist.dx
        except : self.dx = self.S.namelist.dz
        self.Ltrans = self.S.namelist.Ltrans
        self.Llong = self.S.namelist.Llong
        self.tsim = self.S.namelist.tsim
        self.l1 = self.S.namelist.l1
        self.eps = self.S.namelist.eps
        self.ne = self.S.namelist.ne
        self.nppc = self.S.namelist.nppc_plasma
        self.sim_geometry = self.S.namelist.Main.geometry

        self.geometry_LABEL.setText(f"{self.sim_geometry}")
        self.w0_LABEL.setText(f"{self.w0/l0:.1f}𝝀")
        self.a0_LABEL.setText(f"{self.a0:.2f}")
        self.Tp_LABEL.setText(f"{self.Tp/l0:.1f}𝝀")
        self.Pola_LABEL.setText(f"{self.eps}, {self.l1}")
        self.dx_LABEL.setText(f"𝝀/{l0/self.dx:.0f}")
        mesh_trans = int(self.Ltrans/self.dx)
        mesh_long = int(self.Llong/self.dx)
        self.mesh_LABEL.setText(f"{mesh_long} x {mesh_trans} x {mesh_trans}")

        self.Ltrans_LABEL.setText(f"{self.Ltrans/l0:.1f}𝝀")
        self.Llong_LABEL.setText(f"{self.Llong/l0:.1f}𝝀")
        self.tsim_LABEL.setText(f"{self.tsim/l0:.1f}𝝀")

        run_time, push_time = utils.getSimRunTime(self.S._results_path[0])
        NODES = self.S.namelist.smilei_mpi_size//2

        self.run_time_LABEL.setText(f"{(run_time/60)//60:.0f}h{(run_time/60)%60:0>2.0f} | {NODES} nds ({push_time:.0f} ns)")
        self.diag_id = generate_diag_id.get_diag_id(self.sim_directory_path+"/laser_propagation_3d.py")
        self.diag_id_LABEL.setText(f"D{self.diag_id}")
        
        
        self.LAMBDA_UM = 1#um
        self.SI_assume_LABEL= QtWidgets.QLabel(f"SI UNITS (𝝀 = {self.LAMBDA_UM} µm)")

        me = 9.1093837*10**-31
        e = 1.60217663*10**-19
        self.c = 299792458
        eps0 = 8.854*10**-12
        self.toTesla = 10709
        self.wr = 2*pi*self.c/(self.LAMBDA_UM*1e-6)
        self.ne_SI = self.ne*eps0*me/e**2*self.wr**2
        self.wp = np.sqrt(self.ne)*self.wr
        self.wi = np.sqrt(self.ne_SI*e**2/(1836*me*eps0))
        # self.lmbd_D = sqrt(eps0*kB*T)
        self.nc = eps0*me/e**2*self.wr**2*(10**-6) #cm-3
        K = me*self.c**2
        N = eps0*me*self.wr**2/e**2
        L = self.c/self.wr
        KNL3 = K*N*L**3
        self.energy_SI = np.max(self.S.Scalar("Utot").getData())*1000*KNL3
        self.Tp_SI = self.Tp/self.wr*10**15
        
        self.intensity_SI = (self.a0/0.85)**2 *10**18/self.LAMBDA_UM**2 #W/cm^2

        self.power_SI = self.intensity_SI * pi*(self.w0/l0*10**-4)**2/2
        self.power_SI_from_energy = self.energy_SI/1000/(100*1e-15) #P = E/Tp
        
        self.SI_assume_LABEL.setText(f"SI UNITS (𝝀 = {self.LAMBDA_UM} µm)")

        
        self.intensity_SI_LABEL.setText(f"{'%.1E' % decimal.Decimal(str(self.intensity_SI))} W/cm²")
        self.power_SI_LABEL.setText(f"{self.printSI(self.power_SI_from_energy,'W',ndeci=2):}")
        self.energy_SI_LABEL.setText(f"{self.energy_SI:.2f} mJ")
        self.Tp_SI_LABEL.setText(f"{self.Tp_SI:.0f} fs")

        self.nppc_LABEL.setText(f"{self.nppc}")
        self.density_LABEL.setText(f"{self.ne} nc")

        self.scalar_t_range = self.S.Scalar("Utot").getTimes()
        # self.Utot_tot = integrate.simpson(self.S.Scalar("Utot").getData(), x = self.scalar_t_range)
        # self.Uelm_tot = integrate.simpson(self.S.Scalar("Uelm").getData(), x = self.scalar_t_range)
        # self.Ukin_tot = integrate.simpson(self.S.Scalar("Ukin").getData(), x = self.scalar_t_range)

        self.Utot_tot_max = np.max(self.S.Scalar("Utot").getData())
        self.Uelm_tot_max = np.max(self.S.Scalar("Uelm").getData())
        self.Ukin_tot_max = np.max(self.S.Scalar("Ukin").getData())

        self.Utot_tot_end = self.S.Scalar("Utot").getData()[-1]
        self.Uelm_tot_end = self.S.Scalar("Uelm").getData()[-1]
        self.Ukin_tot_end = self.S.Scalar("Ukin").getData()[-1]

        self.KNL3 = K*N*L**3

        self.INIT_tabScalar = True
        self.INIT_tabFields = True
        self.INIT_tabTrack = True
        self.INIT_tabPlasma = True
        self.INIT_tabCompa = True
        self.INIT_tabCompaIntensity = True
        self.INIT_tabIntensity = True
        self.INIT_tabCompaTrack = True

        self.updateInfoLabel()
        if self.actionDiagScalar.isChecked(): self.onUpdateTabScalar(0)
        if self.actionDiagFields.isChecked(): self.onUpdateTabFields(-1)
        if self.actionDiagTrack.isChecked(): self.onUpdateTabTrack(-1)
        if self.actionDiagPlasma.isChecked(): self.onUpdateTabPlasma(-1)
        if self.actionDiagIntensity.isChecked(): self.onUpdateTabIntensity(-1)
        return

    def onLoadCompaSim(self):

        self.compa_load_sim_status = "verifying..."
        self.compa_load_status_LABEL.setStyleSheet("color: black")
        self.compa_load_status_LABEL.setText(self.compa_load_sim_status)
        app.processEvents()
        path = self.compa_sim_directory_path
        path_list = path.split("/")
        self.compa_sim_directory_name = path_list[-1]
        self.compa_sim_directory_parent = path_list[-2]
        self.compa_sim_directory_name_LABEL.setText(self.compa_sim_directory_parent+"/"+ self.compa_sim_directory_name)
        self.compa_S = happi.Open(path)

        if self.compa_S.valid:
            self.compa_load_sim_status = "Loaded"
            self.compa_load_status_LABEL.setStyleSheet("color: green")
            self.is_compa_sim_loaded = True
        else:
            self.compa_load_sim_status = "Invalid"
            self.compa_load_status_LABEL.setStyleSheet("color: red")
            self.is_compa_sim_loaded = False
        self.compa_load_status_LABEL.setText(self.compa_load_sim_status)
        
        self.INIT_tabCompaTrack = True
        #Add text to main parameters
        
        l0=2*pi

        self.compa_eps, self.compa_l1 = self.compa_S.namelist.eps,self.compa_S.namelist.l1
        self.compa_w0 = self.compa_S.namelist.w0
        self.compa_sim_geometry = self.compa_S.namelist.Main.geometry

        compa_w0_txt = f" / <font color='blue'>{self.compa_w0/l0:.1f}𝝀</font>"
        compa_a0_txt = f" / <font color='blue'>{self.compa_S.namelist.a0:.2f}</font>"
        compa_Tp_txt = f" / <font color='blue'>{self.compa_S.namelist.Tp/l0:.1f}𝝀/c</font>"
        compa_Pola_txt = f" / <font color='blue'>{self.compa_S.namelist.eps}, {self.compa_S.namelist.l1}</font>"
        compa_dx_txt = f" / <font color='blue'>𝝀/{l0/self.compa_S.namelist.dx:.0f}</font>"
        
        compa_Ltrans_txt = f" / <font color='blue'>{self.compa_S.namelist.Ltrans/l0:.1f}𝝀/c</font>"
        compa_Llong_txt = f" / <font color='blue'>{self.compa_S.namelist.Llong/l0:.1f}𝝀/c</font>"
        compa_tsim_txt = f" / <font color='blue'>{self.compa_S.namelist.tsim/l0:.1f}𝝀/c</font>"
        compa_nppc_txt = f" / <font color='blue'>{self.compa_S.namelist.nppc_plasma}</font>"
        compa_density_txt = f" / <font color='blue'>{self.compa_S.namelist.ne} nc</font>"
        compa_diag_id = generate_diag_id.get_diag_id(self.compa_sim_directory_path+"/laser_propagation_3d.py")

        compa_diag_id_txt = f" / <font color='blue'>D{compa_diag_id}</font>"
        
        self.w0_LABEL.setText(f"{self.w0/l0:.1f}𝝀"+compa_w0_txt)
        self.a0_LABEL.setText(f"{self.a0:.2f}"+compa_a0_txt)
        self.Tp_LABEL.setText(f"{self.Tp/l0:.1f}t0"+compa_Tp_txt)
        self.Pola_LABEL.setText(f"{self.eps}, {self.l1}"+compa_Pola_txt)
        self.dx_LABEL.setText(f"𝝀/{l0/self.dx:.0f}"+compa_dx_txt)
        mesh_trans = int(self.Ltrans/self.dx)
        mesh_long = int(self.Llong/self.dx)
        self.mesh_LABEL.setText(f"{mesh_long} x {mesh_trans} x {mesh_trans}")

        self.Ltrans_LABEL.setText(f"{self.Ltrans/l0:.1f}𝝀"+compa_Ltrans_txt)
        self.Llong_LABEL.setText(f"{self.Llong/l0:.1f}𝝀"+compa_Llong_txt)
        self.tsim_LABEL.setText(f"{self.tsim/l0:.1f}𝝀/c"+compa_tsim_txt)
        self.density_LABEL.setText(f"{self.ne} nc"+compa_density_txt)
        self.nppc_LABEL.setText(f"{self.nppc}"+compa_nppc_txt)
        
        self.diag_id_LABEL.setText(f"D{self.diag_id}"+ compa_diag_id_txt)

        return

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
        if tab_name=="BINNING":
            self.actionDiagBinning.setChecked(False)
            self.onRemoveBinning()
        if tab_name=="INTENSITY":
            self.actionDiagIntensity.setChecked(False)
            self.onRemoveIntensity()
        if tab_name == "COMPA":
            self.actionDiagCompa.setChecked(False)
            self.onRemoveCompa()
            self.onRemovePlasma()
        # print(self.programm_TABS.count())
        if self.programm_TABS.count() ==0:
            self.verlet_container.show()
        return

    def onMenuTabs(self, tab_name):
        if tab_name == "SCALAR":
            if not self.actionDiagScalar.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "SCALAR":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveScalar()
            else:
                self.programm_TABS.addTab(self.scalar_Widget,"SCALAR")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                # self.smilei_icon_BUTTON.deleteLater()
                self.INIT_tabScalar = True
                app.processEvents()
                self.onUpdateTabScalar(0)

        if tab_name == "FIELDS":
            if not self.actionDiagFields.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "FIELDS":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveFields()
            else:
                self.programm_TABS.addTab(self.fields_Widget,"FIELDS")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                self.INIT_tabFields = True
                app.processEvents()
                self.onUpdateTabFields(0)
                
        if tab_name == "INTENSITY":
            if not self.actionDiagIntensity.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "INTENSITY":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveIntensity()
            else:
                self.programm_TABS.addTab(self.intensity_Widget,"INTENSITY")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                self.INIT_tabIntensity = True
                app.processEvents()
                self.onUpdateTabIntensity(0)

        if tab_name == "TRACK":
            if not self.actionDiagTrack.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TRACK":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveTrack()
            else:
                self.programm_TABS.addTab(self.track_Widget,"TRACK")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                if self.INIT_tabTrack != None: self.INIT_tabTrack = True
                self.INIT_tabTrack = True
                app.processEvents()
                self.onUpdateTabTrack(0)

        if tab_name == "PLASMA":
            if not self.actionDiagPlasma.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "PLASMA":
                        self.programm_TABS.removeTab(currentIndex)
                        print(self.programm_TABS.count)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemovePlasma()
            else:
                self.verlet_container.hide()
                self.programm_TABS.addTab(self.plasma_Widget,"PLASMA")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()

                if self.INIT_tabPlasma != None: self.INIT_tabPlasma = True
                self.INIT_tabPlasma = True
                app.processEvents()
                self.onUpdateTabPlasma(0)

        if tab_name == "COMPA":
            if not self.actionDiagCompa.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "COMPA":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveCompa()
            else:
                self.programm_TABS.addTab(self.compa_Widget,"COMPA")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                if self.INIT_tabCompa != None: self.INIT_tabCompa = True
                self.INIT_tabCompa = True
                app.processEvents()
                self.onUpdateTabCompa(0)

        if tab_name == "BINNING":
            if not self.actionDiagBinning.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "BINNING":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveBinning()
            else:
                self.programm_TABS.addTab(self.binning_Widget,"BINNING")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                # if self.INIT_tabCompa != None: self.INIT_tabCompa = True
                # self.INIT_tabCompa = True
                app.processEvents()
                # self.onUpdateTabBinning()

        if tab_name == "TORNADO":
            if not self.actionTornado.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TORNADO":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.verlet_container.show()
                        self.onRemoveTornado()
            else:
                self.programm_TABS.addTab(self.tornado_Widget,"TORNADO")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.verlet_container.hide()
                app.processEvents()
                self.onInitTabTornado()
                # self.loadthread = ThreadDownloadSimJSON("/sps3/jeremy/LULI/simulations_info.json", os.environ["SMILEI_QT"])
                # self.loadthread.finished.connect(self.onInitTabTornado)
                # self.loadthread.start()
                # self.onInitTabTornado()
        self.updateInfoLabel()
        return

    def writeToFileCompa(self, sim_dir, file_name, file_data):
        file_path = f"{sim_dir}\data_{file_name}.npy"
        np.savetxt(file_path, file_data)
        return

    def call_ThreadGetAMIntegral(self, S, is_compa=False):
        """If not npy file available, use expensive computation"""

        self.loadthread = class_threading.ThreadGetAMIntegral(S)
        self.loadthread.finished.connect(partial(self.onUpdateTabScalar_AM,is_compa=is_compa))
        self.loadthread.start()
        return

    def call_ThreadGetAMIntegral_compa(self, S):
        """If not npy file available for comparison """
        self.loadthread = class_threading.ThreadGetAMIntegral(S)
        self.loadthread.finished.connect(partial(self.onUpdateTabScalar_AM_compa))
        self.loadthread.start()
        return
    def onUpdateTabScalar_AM_compa(self, AM_full_int_compa):
        canvas = self.canvas_4_scalar
        figure = canvas.figure
        ax = figure.axes[0]
        fields_t_range = self.compa_S.Probe(0,"Ex").getTimes()

        ax.plot(fields_t_range/self.l0, AM_full_int_compa,"--", label="AM", c=f"C{len(ax.get_lines())}")
        AM_max = np.nanmax(AM_full_int_compa)
        figure.suptitle(f"""{self.sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_max/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ""",fontsize=14)

        self.writeToFileCompa(self.compa_sim_directory_path,"AM",np.vstack([fields_t_range,AM_full_int_compa]).T)
        ax.legend()
        canvas.draw() 
        ax.relim()            # Recompute the limits
        ax.autoscale_view()  
        figure.tight_layout()
        canvas.draw()


    def onUpdateTabScalar_AM(self, AM_full_int, is_compa=False):
        if not is_compa:
            canvas = self.canvas_0
        else: canvas = self.canvas_4_scalar

        fields_t_range = self.S.Probe(0,"Ex").getTimes()
        AM_tot = np.max(AM_full_int)

        figure = canvas.figure
        ax = figure.axes[0]
        ax.plot(fields_t_range/self.l0, AM_full_int,label="AM", ls="-", c=f"C{len(ax.get_lines())}")
        self.writeToFileCompa(self.sim_directory_path,"AM",np.vstack([fields_t_range,AM_full_int]).T)


        if is_compa and self.is_compa_sim_loaded:
            try:
                data_file = np.loadtxt(f"{self.compa_sim_directory_path}/data_AM.npy")
                data_t_range, data = data_file[:,0], data_file[:,1]
                ax.plot(data_t_range/self.l0, data,
                          label="AM", ls="--",color = f"C{len(ax.get_lines())}")
            except FileNotFoundError:
                print("USE EXPENSIVE AM COMPUTATION COMPA")
                self.call_ThreadGetAMIntegral_compa(self.compa_S)
                # raise
                # self.call_ThreadGetAMIntegral(self.compa_S)
        ax.legend()
        figure.suptitle(f"{self.sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_tot/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)
        if is_compa:
            figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_tot/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)

        canvas.draw()        
        ax.relim()            # Recompute the limits
        ax.autoscale_view()   
        figure.tight_layout()
        canvas.draw()

        return

    def onUpdateTabScalar(self, check_id, is_compa=False):
        print("onUpdateTabScalar")
        # print("onUpdateTabScalar",check_id, is_compa)
        if self.INIT_tabScalar == None or self.is_sim_loaded == False:
            # Popup().showError("Simulation not loaded")
            return
        if self.INIT_tabScalar:
            [check.setChecked(False) for check in self.scalar_check_list[1:]]
            [check.setChecked(False) for check in self.compa_scalar_check_list[1:]]
            self.ax0.cla()
            self.ax4_scalar.cla()
            self.ax0.grid()
            self.ax0.set_xlabel("t/t0",fontsize=14)
            self.ax4_scalar.grid()
            self.ax4_scalar.set_xlabel("t/t0",fontsize=14)

            print("===== INIT SCALAR =====")

        self.INIT_tabScalar = False

        # t0 = time.perf_counter()
        l0 = 2*pi
        
        if not is_compa:
            canvas = self.canvas_0
            boolList = [check.isChecked() for check in self.scalar_check_list]
        else:
            canvas = self.canvas_4_scalar
            boolList = [check.isChecked() for check in self.compa_scalar_check_list]

        self.scalar_t_range = self.S.Scalar("Uelm").getTimes()
        figure = canvas.figure
        ax = figure.axes[0]
        AM_max = np.NaN

        if boolList[check_id] == True: # Was False before and has been selected
            if self.scalar_names[check_id] == "Uelm/Utot":
                data = np.array(self.S.Scalar("Uelm").getData())/(np.array(self.S.Scalar("Utot").getData())+1e-12)
                ax.plot(self.scalar_t_range[10:]/l0, data[10:],
                              label=self.scalar_names[check_id], ls="-",color = f"C{len(ax.get_lines())}")

                if is_compa and self.is_compa_sim_loaded:
                    # data_file = np.loadtxt(f"{self.compa_sim_directory_path}/data_Uelm_over_Utot.npy")
                    # data_t_range, data = data_file[:,0], data_file[:,1]
                    data_compa = np.array(self.compa_S.Scalar("Uelm").getData())/(np.array(self.compa_S.Scalar("Utot").getData())+1e-12)
                    data_t_range_compa = self.compa_S.Scalar("Uelm").getTimes()
                    ax.plot(data_t_range_compa[10:]/l0, data_compa[10:],
                              label=f"{self.scalar_names[check_id]}", ls="--",color = f"C{len(ax.get_lines())}")

            elif self.scalar_names[check_id] =="α_abs":
                data = np.array(np.array(self.S.Scalar("Uelm").getData())/(np.array(self.S.Scalar("Utot").getData())+1e-12))
                dt_diag = self.scalar_t_range[1]/l0 - self.scalar_t_range[0]/l0 #/l0 bcs we want um^-1 as units
                alpha = scipy.signal.savgol_filter(data[20:], window_length=round(self.Llong/l0), polyorder=5, deriv=1)/dt_diag #interpolate and derivate
                ax.plot(self.scalar_t_range[20:]/l0, alpha*1000,
                              label=self.scalar_names[check_id], ls="-",color = f"C{len(ax.get_lines())}")
                alpha_theo = -0.5*self.ne/(self.Tp/2/l0)*1000
                ax.axhline(alpha_theo,ls="--",color="k",alpha=0.5, label=r"$\alpha_{abs}$ theo")
                if is_compa and self.is_compa_sim_loaded:
                    data_t_range_compa = self.compa_S.Scalar("Uelm").getTimes()
                    data = np.array(np.array(self.compa_S.Scalar("Uelm").getData())/(np.array(self.compa_S.Scalar("Utot").getData())+1e-12))
                    dt_diag = self.scalar_t_range[1]/l0 - self.scalar_t_range[0]/l0 #/l0 bcs we want um^-1 as units
                    alpha = scipy.signal.savgol_filter(data[20:], window_length=round(self.Llong/l0), polyorder=5, deriv=1)/dt_diag #interpolate and derivate
                    ax.plot(data_t_range_compa[20:]/l0, alpha*1000,
                                  label=self.scalar_names[check_id], ls="--",color = f"C{len(ax.get_lines())}")

            elif self.scalar_names[check_id] != "AM":
                data = np.array(self.S.Scalar(self.scalar_names[check_id]).getData())
                ax.plot(self.scalar_t_range/l0, data,
                              label=self.scalar_names[check_id], color = f"C{len(ax.get_lines())}")
                # self.writeToFileCompa(self.sim_directory_path, self.scalar_names[check_id],np.VStack([self.scalar_t_range, data]).T)
                if is_compa and self.is_compa_sim_loaded:
                    #data_file = np.loadtxt(f"{self.compa_sim_directory_path}/data_{self.scalar_names[check_id]}.npy")
                    #data_t_range, data = data_file[:,0], data_file[:,1]
                    data_compa = np.array(self.compa_S.Scalar(self.scalar_names[check_id]).getData())
                    data_t_range_compa = self.compa_S.Scalar("Uelm").getTimes()
                    ax.plot(data_t_range_compa/l0, data_compa,
                              label=f"{self.scalar_names[check_id]}", ls="--",color = f"C{len(ax.get_lines())}")
            else: #Angular Momentum
                try:
                    data_file_AM = np.loadtxt(f"{self.sim_directory_path}/data_AM.npy")
                    print(data_file_AM.shape)
                    data_t_range, AM_data = data_file_AM[:,0], data_file_AM[:,1]
                    ax.plot(data_t_range/l0, AM_data,
                              label=f"{self.scalar_names[check_id]}", ls="-",color = f"C{len(ax.get_lines())}")
                    # self.onUpdateTabScalar_AM(AM_data,is_compa=is_compa)
                        # self.onUpdateTabScalar_AM(AM_data_compa,is_compa=is_compa)
                except FileNotFoundError:
                    print("USE EXPENSIVE AM COMPUTATION")
                    self.call_ThreadGetAMIntegral(self.S, is_compa)
                if is_compa and self.is_compa_sim_loaded:
                    try:
                        data_file_AM_compa = np.loadtxt(f"{self.compa_sim_directory_path}/data_AM.npy")
                        data_t_range_compa, AM_data_compa = data_file_AM_compa[:,0], data_file_AM_compa[:,1]
                        ax.plot(data_t_range_compa/l0, AM_data_compa,
                                  label=f"{self.scalar_names[check_id]}", ls="--",color = f"C{len(ax.get_lines())}")
                    except FileNotFoundError:
                        print("USE EXPENSIVE AM COMPUTATION COMPA ONLY")
                        self.call_ThreadGetAMIntegral_compa(self.compa_S)
                # AM_max = np.max(AM_data)

                # return
        else:
            old_lines = ax.get_lines()

            for k, t in enumerate(ax.get_legend().get_texts()): # Delete lines
                if self.scalar_names[check_id] == t.get_text():
                    old_lines[k].remove()
            updated_lines = ax.get_lines()
            if len(updated_lines) > 0:
                for k in range(len(updated_lines)): # Recolor lines
                    ax.get_lines()[k].set_color(f"C{k}")
        ax.legend()
        ax.relim()            # Recompute the limits based on current data
        ax.autoscale_view()   # Apply the new limits
        # hfont = {'fontname':'Helvetica'}

        figure.suptitle(f"""{self.sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_max/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ""",fontsize=14)
        if is_compa:
            figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_max/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)

        figure.tight_layout()
        canvas.draw()
        # t1 = time.perf_counter()
        return

    def onUpdateTabFieldsFigure(self, fields_data_list):

        #=====================================
        # REMOVE ALL FIGURES --> NOT OPTIMAL
        #=====================================
        if len(self.figure_1.axes) !=0:
            for ax in self.figure_1.axes: ax.remove()

        self.fields_data_list = fields_data_list
        self.fields_image_list = []

        boolList = [check.isChecked() for check in self.fields_check_list]
        combo_box_index = self.sim_cut_direction_BOX.currentIndex()

        Naxis = min(sum(boolList),3)

        if Naxis != len(self.fields_data_list):
            # self.loading_LABEL.deleteLater()
            return

        time_idx = self.fields_time_SLIDER.sliderPosition()
        x_idx = self.fields_xcut_SLIDER.sliderPosition()


        t1 = time.perf_counter()
        k=0
        print("--------------")
        for i in range(len(self.fields_names)):
            if boolList[i]:
                if combo_box_index==0:
                    print(len(self.fields_data_list))
                    ax = self.figure_1.add_subplot(Naxis,1,k+1)
                    im = ax.imshow(self.fields_data_list[k][time_idx,:,:,self.fields_trans_mid_idx].T,cmap="RdBu", aspect="auto",
                                   extent=self.extentXY,origin='lower', interpolation="spline16")
                    ax.set_title(self.fields_names[i],rotation='vertical',x=-0.1,y=0.48)
                else:
                    ax = self.figure_1.add_subplot(1,Naxis,k+1)
                    im = ax.imshow(fields_data_list[k][time_idx,x_idx,:,:].T,cmap="RdBu", aspect="auto",
                                   extent=self.extentYZ,origin='lower', interpolation="spline16")
                    ax.set_title(self.fields_names[i])
                if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
                self.figure_1.colorbar(im, ax=ax,pad=0.01)
                self.fields_image_list.append(im)
                k+=1
        t2 = time.perf_counter()
        print("plot field",(t2-t1)*1000,"ms")

        byte_size_track = getsizeof(self.fields_data_list)+getsizeof(self.fields_image_list)
        print("Memory from FIELDS:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/psutil.virtual_memory().total,1),"%)")
        if combo_box_index==0:
            self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
        else:
            self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[x_idx]/self.l0:.2f}~\lambda$",**self.qss_plt_title)
        self.figure_1.tight_layout()
        self.figure_1.tight_layout()
        self.canvas_1.draw()

        # self.loading_LABEL.deleteLater()


    def onUpdateTabFields(self,check_id):
        boolList = [check.isChecked() for check in self.fields_check_list]

        if sum(boolList)==3:
            for check in self.fields_check_list:
                if not check.isChecked():
                    check.setEnabled(False)
        else:
            for check in self.fields_check_list:
                if not check.isChecked():
                    check.setEnabled(True)


        if self.INIT_tabFields == None or self.is_sim_loaded == False:
            # Popup().showError("Simulation not loaded")
            return
        if self.INIT_tabFields:
            print("===== INIT FIELDS TAB =====")
            # self.displayLoadingLabel(self.fields_groupBox)
            Ex_diag = self.S.Probe(0,"Ex")
            l0 = 2*pi
            # fields_shape = np.array(self.S.Probe(0,"Ex").getData()).astype(np.float32).shape

            self.fields_paxisX = Ex_diag.getAxis("axis1")[:,0]
            self.fields_paxisY = Ex_diag.getAxis("axis2")[:,1]-self.Ltrans/2
            self.fields_paxisZ = Ex_diag.getAxis("axis3")[:,2]-self.Ltrans/2
            self.extentYZ = [self.fields_paxisY[0]/self.l0,self.fields_paxisY[-1]/self.l0,self.fields_paxisZ[0]/self.l0,self.fields_paxisZ[-1]/self.l0]
            self.extentXY = [self.fields_paxisX[0]/self.l0,self.fields_paxisX[-1]/self.l0,self.fields_paxisY[0]/self.l0,self.fields_paxisY[-1]/self.l0]
            self.fields_t_range = Ex_diag.getTimes()

            del Ex_diag

            self.fields_trans_mid_idx = len(self.fields_paxisZ)//2
            self.fields_long_mid_idx = len(self.fields_paxisX)//2
            self.fields_time_SLIDER.setMaximum(len(self.fields_t_range)-1)
            self.fields_xcut_SLIDER.setMaximum(len(self.fields_paxisX)-1)

            self.fields_image_list = []
            self.fields_data_list = []


            self.fields_time_SLIDER.setValue(len(self.fields_t_range))
            self.fields_previous_xcut_SLIDER_value = self.fields_xcut_SLIDER.sliderPosition()

            self.fields_time_EDIT.setText(str(round(self.fields_t_range[-1]/l0,2)))
            self.fields_xcut_EDIT.setText(str(round(self.fields_paxisX[-1]/l0,2)))


            byte_size_track = getsizeof(self.fields_paxisX)+getsizeof(self.fields_paxisY)+getsizeof(self.fields_paxisZ)
            print("Memory from FIELDS:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/self.MEMORY().total,1),"%)")

            # self.loading_LABEL.deleteLater()
            self.INIT_tabFields = False
            app.processEvents()
            self.updateInfoLabel()

        l0 = 2*pi
        if check_id < 10: #CHECK_BOX UPDATE

            combo_box_index = self.sim_cut_direction_BOX.currentIndex()

            self.fields_image_list = []
            self.fields_data_list = []

            self.loadthread = class_threading.ThreadGetFieldsProbeData(boolList, self.fields_names, self.S, self.fields_t_range, self.fields_paxisX, self.fields_paxisY, self.fields_paxisZ)
            self.loadthread.finished.connect(self.onUpdateTabFieldsFigure)
            self.loadthread.start()


        elif check_id==200 and self.sim_cut_direction_BOX.currentIndex()==0:
            self.fields_xcut_SLIDER.setValue(self.fields_previous_xcut_SLIDER_value) #cannot change z slider if not in Transverse mode
            return

        elif check_id <= 110 or ((check_id==200 or check_id==201) and self.sim_cut_direction_BOX.currentIndex()==1): #SLIDER UPDATE
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
                xcut_edit_value = float(self.fields_xcut_EDIT.text())
                xcut_idx = np.where(abs(self.fields_paxisX/l0-xcut_edit_value)==np.min(abs(self.fields_paxisX/l0-xcut_edit_value)))[0][0]
                self.fields_xcut_SLIDER.setValue(xcut_idx)
            else:
                xcut_idx = self.fields_xcut_SLIDER.sliderPosition()
                self.fields_xcut_EDIT.setText(str(round(self.fields_paxisX[xcut_idx]/l0,2)))

            self.fields_previous_xcut_SLIDER_value = self.fields_xcut_SLIDER.sliderPosition()
            combo_box_index = self.sim_cut_direction_BOX.currentIndex()

            if combo_box_index==0:
                for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,:,:,self.fields_trans_mid_idx].T)
                        self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
                        if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
            else:
                for i,im in enumerate(self.fields_image_list):
                    im.set_data(self.fields_data_list[i][time_idx,xcut_idx,:,:].T)
                    if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
                    self.figure_1.suptitle(f"$t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[xcut_idx]/l0:.2f}~\lambda$",**self.qss_plt_title)
            self.canvas_1.draw()

        elif check_id == 1000:

            if self.loop_in_process: return

            self.loop_in_process = True

            combo_box_index = self.sim_cut_direction_BOX.currentIndex()
            xcut_idx = self.fields_xcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.fields_t_range)):
                self.fields_time_SLIDER.setValue(time_idx)
                if combo_box_index==0:
                    for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,:,:,self.fields_trans_mid_idx].T)
                        self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
                else:
                    for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,xcut_idx,:,:].T)
                        if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
                self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[xcut_idx]/l0:.2f}~\lambda$",**self.qss_plt_title)
                self.canvas_1.draw()
                time.sleep(0.05)
                app.processEvents()

            self.loop_in_process = False
        self.updateInfoLabel()
    
    def onUpdateTabIntensity(self,check_id):
        boolList = [check.isChecked() for check in self.intensity_check_list]
        if sum(boolList) <1: return
        l0 = 2*pi
        if self.INIT_tabIntensity == None or self.is_sim_loaded == False:
            return
        if self.INIT_tabIntensity and check_id < 10:
            print("===== INIT INTENSITY TAB =====")
            print(boolList)
            if len(self.figure_6.axes) !=0:
                for ax in self.figure_6.axes: ax.remove()
            if len(self.figure_6_time.axes) !=0:
                for ax in self.figure_6_time.axes: ax.remove()
                
            self.INIT_tabIntensity = False
            Ey = 0
            Ex = 0
            # x5_idx = np.where(np.abs(self.intensity_paxisX-5*l0)==np.min(np.abs(self.intensity_paxisX-5*l0)))[0][0]

            if boolList[1]:
                Ey_diag = self.S.Probe("Exy_intensity","Ey")
                self.E_diag_name_str = "Ey"
                self.intensity_diag_name_str = "|Ey|^2"
                Ey = np.array(Ey_diag.getData()).astype(np.float32)
                intensity_dx = Ey_diag.getAxis("axis1")[:,0][1] - Ey_diag.getAxis("axis1")[:,0][0]
                any_diag = Ey_diag
                try:
                    Ey_hd_diag = self.S.Probe("temp_env","Ey")
                    self.Ey_hd = np.array(Ey_hd_diag.getData())
                    try:
                        Ez_hd_diag = self.S.Probe("temp_env","Ez")
                        self.Ez_hd = np.array(Ez_hd_diag.getData())
                    except:
                        self.Ez_hd = self.Ey_hd*0 #if no Ez diag, set it to 0
                        
                    
                    abs_E_x0 = np.max(np.sqrt(self.Ey_hd[:,0,:,:]**2 + self.Ez_hd[:,0,:,:]**2),axis=(1,2))
                    self.intensity_t_range_hd = Ey_hd_diag.getTimes()
                except:
                    abs_E_x0 = np.max(np.abs(Ey[:,0,:,:]),axis=(1,2))
                    self.intensity_t_range_hd = Ey_diag.getTimes()
                    

            if boolList[0]:
                Ex_diag = self.S.Probe("Exy_intensity","Ex")
                self.E_diag_name_str = "Ex"
                self.intensity_diag_name_str = "|Ex|^2"
                Ex = np.array(Ex_diag.getData()).astype(np.float32)
                intensity_dx = Ex_diag.getAxis("axis1")[:,0][1] - Ex_diag.getAxis("axis1")[:,0][0]
                any_diag = Ex_diag
                abs_E_x0 = np.max(np.abs(Ex[:,0,:,:]),axis=(1,2))
                # Ex_hd = np.array(self.S.Probe("temp_env","Ex"))
                # abs_E_x0_hd = np.max(np.abs(Ex_hd[:,0,:,:]),axis=(1,2))

            average_over = 2*l0 #number of optical period
            intensity_cmap = "jet"
            
            W = round(average_over/intensity_dx)
            
            if "_AM" in self.sim_directory_name:
                track_r_center = 0
            else:
                track_r_center = self.Ltrans/2
            
            self.intensity_paxisX = any_diag.getAxis("axis1")[W:,0] #remove first W values
            self.intensity_paxisY = any_diag.getAxis("axis2")[:,1]-track_r_center
            self.intensity_paxisZ = any_diag.getAxis("axis3")[:,2]-track_r_center
            self.intensity_t_range = any_diag.getTimes()

            self.intensity_extentYZ = [self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0,
                                       self.intensity_paxisZ[0]/self.l0,self.intensity_paxisZ[-1]/self.l0]
            self.intensity_extentXY = [self.intensity_paxisX[0]/self.l0,self.intensity_paxisX[-1]/self.l0,
                                       self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0]
            self.intensity_extentTY = [self.intensity_t_range[0]/self.l0,self.intensity_t_range[-1]/self.l0,
                                       self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0]
            if boolList[1]: del Ey_diag
            if boolList[0]: del Ex_diag
            del any_diag

            self.intensity_trans_mid_idx = len(self.intensity_paxisZ)//2
            self.intensity_long_mid_idx = len(self.intensity_paxisX)//2
            self.intensity_time_SLIDER.setMaximum(len(self.intensity_t_range)-1)
            self.intensity_xcut_SLIDER.setMaximum(len(self.intensity_paxisX)-1)
            
            self.intensity_time_SLIDER.setValue(len(self.intensity_t_range)-1)
            self.intensity_xcut_SLIDER.setValue(len(self.intensity_paxisX)-1)
            
            self.intensity_time_EDIT.setText(str(round(self.intensity_t_range[-1]/l0,2)))
            self.intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[-1]/l0,2)))
            #COMPUTE INTENSITY 
            t0 = time.perf_counter()
            temp_Int = Ey**2 + Ex**2
            gc.collect()
            Ey_squared_cumsum = np.cumsum(temp_Int,axis=1)
            t1 = time.perf_counter()
            self.intensity_data = (Ey_squared_cumsum[:,W:] - Ey_squared_cumsum[:,:-W]) / W
            t2 = time.perf_counter()
            
            del Ey_squared_cumsum
            gc.collect()

            print("cumsum:",(t1-t0),"s | ","average:",(t2-t1),"s")
            
            max_intensity = np.max(self.intensity_data)
            
            #PLOT INTENSITY
            self.ax6_a = self.figure_6.add_subplot(1,2,1)
            self.ax6_b = self.figure_6.add_subplot(1,2,2)
            self.intensity_im_a = self.ax6_a.imshow(self.intensity_data[-1,:,:,self.intensity_trans_mid_idx].T,aspect="auto", 
                                                    extent=self.intensity_extentXY,vmax=max_intensity,cmap=intensity_cmap)
            self.intensity_im_b = self.ax6_b.imshow(self.intensity_data[-1,-1,:,:],aspect="auto", 
                                                    extent=self.intensity_extentYZ,vmax=max_intensity,cmap=intensity_cmap)
            
            self.figure_6.colorbar(self.intensity_im_a, ax =self.ax6_a,pad=0.01)
            self.figure_6.colorbar(self.intensity_im_b, ax =self.ax6_b,pad=0.01)
            
            self.intensity_line_x = self.ax6_a.axvline(x=self.intensity_paxisX[-1]/l0,color="r",ls="--",alpha=0.5)
            
            zR = 0.5*self.w0**2 #Rayleigh length
            self.waist_max_intensity = self.w0*sqrt(1+((self.intensity_paxisX-self.xfoc)/zR)**2)*sqrt(abs(self.l1)/2)
            """
            To convert from intensity to a0: a = np.sqrt(2*I)*np.exp(0.5), for |l| = 1
            """
            self.ax6_a.plot(self.intensity_paxisX/l0,self.waist_max_intensity/l0, color="k", ls="--")
            self.ax6_a.plot(self.intensity_paxisX/l0,-self.waist_max_intensity/l0, color="k", ls="--")
            self.circle_max_intensity = plt.Circle((0, 0), self.waist_max_intensity[-1], fill=False, ec="k",ls="--")
            self.ax6_b.add_patch(self.circle_max_intensity)
            
            self.figure_6.suptitle(f"{self.sim_directory_name} {self.intensity_diag_name_str} | $t={self.intensity_t_range[-1]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[-1]/l0:.2f}~\lambda$",**self.qss_plt_title)
            self.figure_6.tight_layout()
            self.canvas_6.draw()
            
            self.ax6_time = self.figure_6_time.add_subplot(1,2,1)
            self.ax6_time_temp_env = self.figure_6_time.add_subplot(1,2,2)
            
            pwr = 10
            c_super=sqrt(2)*special.gamma((pwr+1)/pwr)
            c_gauss = sqrt(pi/2)
            def sin2(t):
                return np.sin(pi*t/self.Tp)**2*(t<=self.Tp)
            def gauss(t):
                t_center = 1.25*self.Tp
                return np.exp(-((t-t_center)/(self.Tp*3/8/c_gauss))**2)
            # def superGauss(t):
            #     return np.exp(-((t-self.Tp)/(self.Tp/sqrt(2.)/2/c_super))**10)
            # temp_env = sin2
            t_range_smooth = np.arange(0,self.intensity_t_range_hd[-1],0.1)
            # print()
            self.ax6_time_temp_env.plot(self.intensity_t_range_hd/l0, abs_E_x0,".-",label="|E|(x=0)")
            # self.ax6_time_temp_env.plot(t_range_smooth/l0, self.a0*np.exp(-0.5*np.abs(self.l1))*sin2(t_range_smooth),"--",label="sin2")
            self.ax6_time_temp_env.plot(t_range_smooth/l0, self.a0*np.exp(-0.5*np.abs(self.l1))*gauss(t_range_smooth),"r--",label="gauss")
            # self.ax6_time_temp_env.plot(t_range_smooth/l0, self.a0*np.exp(-0.5)*superGauss(t_range_smooth),"--",label="superGauss")
            # self.ax6_time_temp_env.plot(self.intensity_t_range, abs_E_x5,".-",label="|E|(x=<x0>)")

            
            empirical_corr = 0.98 # To compensate for other effect on top of dis
            groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) # Formula: vg = c^2k/sqrt(wp^2+c^2k^2)
            laser_x_pos_range = np.max([groupe_velocity*self.intensity_t_range-self.Tp/2,self.intensity_t_range*0],axis=0)          
            self.intensity_data_time = np.empty((self.intensity_data.shape[0],self.intensity_data.shape[2]))
            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))

            for t,laser_x_pos in enumerate(laser_x_pos_range):
                xcut_idx = np.where(self.intensity_data[t,:,:,self.intensity_trans_mid_idx]==max_intensity[t])[0][0]  #Use maximum intensity
                if self.intensity_use_vg_CHECK.isChecked(): xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0] #Use group velocity
                self.intensity_data_time[t] = self.intensity_data[t,xcut_idx,:,self.intensity_trans_mid_idx]
            self.intensity_im_time = self.ax6_time.imshow(self.intensity_data_time.T, cmap="jet",aspect="auto",extent=self.intensity_extentTY)
            self.figure_6_time.colorbar(self.intensity_im_time, ax = self.ax6_time, pad=0.01)
            self.ax6_time.set_title(f"{self.sim_directory_name} {self.intensity_diag_name_str}\n(use group velocity: {self.intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)
            self.ax6_time.set_xlabel("t/t0")
            self.ax6_time_temp_env.set_title(f"{self.sim_directory_name} |{self.E_diag_name_str}|(t,x=0)",**self.qss_plt_title)
            self.ax6_time_temp_env.set_xlabel("t/t0")
            self.ax6_time_temp_env.legend()
            self.ax6_time_temp_env.grid()
            self.figure_6_time.tight_layout()
            self.canvas_6_time.draw()
            
           
        if check_id == 2000:
            if self.intensity_spatial_time_BOX.currentIndex() == 1:
                self.canvas_6.hide()
                self.plt_toolbar_6.hide()                
                self.canvas_6_time.show()
                self.plt_toolbar_6_time.show()                
            else:
                self.canvas_6.show()
                self.plt_toolbar_6.show()                
                self.canvas_6_time.hide()
                self.plt_toolbar_6_time.hide()
            return

        if check_id < 10:
            Ey = 0
            Ex = 0
            t0 = time.perf_counter()

            if boolList[1]:
                Ey_diag = self.S.Probe("Exy_intensity","Ey")
                self.intensity_diag_name_str = "|Ey|^2"
                Ey = np.array(Ey_diag.getData()).astype(np.float32)
                intensity_dx = Ey_diag.getAxis("axis1")[:,0][1] - Ey_diag.getAxis("axis1")[:,0][0]
                any_diag = Ey_diag

            if boolList[0]:
                Ex_diag = self.S.Probe("Exy_intensity","Ex")
                self.intensity_diag_name_str = "|Ex|^2"
                Ex = np.array(Ex_diag.getData()).astype(np.float32)
                intensity_dx = Ex_diag.getAxis("axis1")[:,0][1] - Ex_diag.getAxis("axis1")[:,0][0]
                any_diag = Ex_diag
            t1 = time.perf_counter()

            average_over = 2*l0 #number of optical period
            intensity_cmap = "jet"
            
            W = round(average_over/intensity_dx)
            #COMPUTE INTENSITY 
            temp_Int = Ey**2 + Ex**2
            # del Ey, Ex
            gc.collect()
            Ey_squared_cumsum = np.cumsum(temp_Int,axis=1)
            self.intensity_data = (Ey_squared_cumsum[:,W:] - Ey_squared_cumsum[:,:-W]) / W
            t2 = time.perf_counter()
            del Ey_squared_cumsum
            gc.collect()
            print("load Ex/Ey in RAM:",(t1-t0),"s | ","Compute intensity:",(t2-t1),"s")
            
            max_intensity = np.max(self.intensity_data)
            
            #PLOT INTENSITY
            time_idx = self.intensity_time_SLIDER.sliderPosition()
            xcut_idx = self.intensity_xcut_SLIDER.sliderPosition()

            self.intensity_im_a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.intensity_im_b.set_data(self.intensity_data[time_idx,xcut_idx,:,:])
            self.intensity_im_a.set_clim(vmax=max_intensity)
            self.intensity_im_b.set_clim(vmax=max_intensity)

            self.ax6_time.set_title(f"{self.sim_directory_name} {self.intensity_diag_name_str}\n(use group velocity: {self.intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)
            self.figure_6.tight_layout()
            self.canvas_6.draw()
            check_id = 5000 # Update time distribution too
                
        elif check_id <= 110 or check_id==200 or check_id==201: #SLIDER OR LINE_EDIT UPDATE
            self.timer = time.perf_counter()
            if check_id == 101: #QLineEdit changed
                time_edit_value = float(self.intensity_time_EDIT.text())
                time_idx = np.where(abs(self.intensity_t_range/l0-time_edit_value)==np.min(abs(self.intensity_t_range/l0-time_edit_value)))[0][0]
                self.intensity_time_SLIDER.setValue(time_idx)
                self.intensity_time_EDIT.setText(str(round(self.intensity_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.intensity_time_SLIDER.sliderPosition()
                self.intensity_time_EDIT.setText(str(round(self.intensity_t_range[time_idx]/l0,2)))

            if check_id == 201:#QSlider changed
                xcut_edit_value = float(self.intensity_xcut_EDIT.text())
                xcut_idx = np.where(abs(self.intensity_paxisX/l0-xcut_edit_value)==np.min(abs(self.intensity_paxisX/l0-xcut_edit_value)))[0][0]
                self.intensity_xcut_SLIDER.setValue(xcut_idx)
            else:
                xcut_idx = self.intensity_xcut_SLIDER.sliderPosition()
                self.intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[xcut_idx]/l0,2)))

            self.intensity_previous_xcut_SLIDER_value = self.intensity_xcut_SLIDER.sliderPosition()
            # combo_box_index = self.sim_cut_direction_BOX.currentIndex()
            
            if self.intensity_follow_laser_CHECK.isChecked():
                max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
                xcut_idx = np.where(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx]==max_intensity[time_idx])[0][0] #Use maximum intensity
                if self.intensity_use_vg_CHECK.isChecked(): 
                    empirical_corr = 0.98 #to compensate for other effect on top of dis
                    groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) #c^2k/sqrt(wp^2+c^2k^2)
                    laser_x_pos = max(groupe_velocity*self.intensity_t_range[time_idx]-self.Tp/2,0)
                    xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0]

                self.intensity_xcut_SLIDER.setValue(xcut_idx)
            
            self.intensity_im_a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.intensity_im_b.set_data(self.intensity_data[time_idx,xcut_idx,:,:])
            self.intensity_line_x.set_xdata(self.intensity_paxisX[xcut_idx]/l0)
            self.circle_max_intensity.set_radius(self.waist_max_intensity[xcut_idx]/l0)

            self.ax6_time.set_title(f"{self.sim_directory_name} {self.intensity_diag_name_str}\n(use group velocity: {self.intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)
            self.figure_6_time.tight_layout()
            self.canvas_6.draw()
        elif check_id == 1000:

            if self.loop_in_process: return

            self.loop_in_process = True

            xcut_idx = self.intensity_xcut_SLIDER.sliderPosition()
            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))

            for time_idx in range(len(self.intensity_t_range)):
                
                # empirical_corr = 0.98 #to compensate for other effect on top of dis
                # groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) #c^2k/sqrt(wp^2+c^2k^2)
                # laser_x_pos = max(groupe_velocity*self.intensity_t_range[time_idx]-self.Tp/2,0)
                # laser_x_pos_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0]
                laser_x_pos_idx = np.where(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx]==max_intensity[time_idx])[0][0] #Use maximum intensity
                
                self.intensity_time_SLIDER.setValue(time_idx)
                self.intensity_xcut_SLIDER.setValue(laser_x_pos_idx)
                self.intensity_time_EDIT.setText(str(round(self.intensity_t_range[time_idx]/l0,2)))
                self.intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[laser_x_pos_idx]/l0,2)))
                
                self.intensity_im_a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
                self.intensity_im_b.set_data(self.intensity_data[time_idx,laser_x_pos_idx,:,:])
                self.intensity_line_x.set_xdata(self.intensity_paxisX[laser_x_pos_idx]/l0)
                self.circle_max_intensity.set_radius(self.waist_max_intensity[xcut_idx]/l0)

                self.figure_6.suptitle(f"{self.sim_directory_name} {self.intensity_diag_name_str} | $t={self.intensity_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[xcut_idx]/l0:.2f}~\lambda$",**self.qss_plt_title)
                self.figure_6_time.tight_layout()
                self.canvas_6.draw()
                time.sleep(0.10)
                app.processEvents()

            self.loop_in_process = False
            
        if self.INIT_tabIntensity==False and check_id == 5000:
            max_intensity_vmax = np.max(self.intensity_data)
            empirical_corr = 0.98 # To compensate for other effect on top of dis
            groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) # Formula: vg = c^2k/sqrt(wp^2+c^2k^2)
            laser_x_pos_range = np.max([groupe_velocity*self.intensity_t_range-self.Tp/2,self.intensity_t_range*0],axis=0)          
            self.intensity_data_time = np.empty((self.intensity_data.shape[0],self.intensity_data.shape[2]))
            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            for t,laser_x_pos in enumerate(laser_x_pos_range):
                xcut_idx = np.where(self.intensity_data[t,:,:,self.intensity_trans_mid_idx]==max_intensity[t])[0][0] #Use maximum intensity
                if self.intensity_use_vg_CHECK.isChecked(): 
                    xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0] #Use group velocity
                self.intensity_data_time[t] = self.intensity_data[t,xcut_idx,:,self.intensity_trans_mid_idx]
            self.intensity_im_time.set_data(self.intensity_data_time.T)
            self.intensity_im_time.set_clim(vmax=max_intensity_vmax)
            self.ax6_time.set_title(f"{self.sim_directory_name} {self.intensity_diag_name_str}\n(use group velocity: {self.intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)
            self.figure_6_time.tight_layout()
            self.canvas_6_time.draw()
        self.updateInfoLabel()
    
    def onUpdateTabCompaIntensity(self,check_id):
        boolList = [check.isChecked() for check in self.compa_intensity_check_list]
        if sum(boolList) <1: return
        l0 = 2*pi
        if self.INIT_tabCompaIntensity == None or self.is_sim_loaded == False or self.is_compa_sim_loaded==False:
            return
        if self.INIT_tabCompaIntensity and check_id < 10:
            print("===== INIT INTENSITY TAB =====")
            print(boolList)
            t0 = time.perf_counter()
            self.INIT_tabCompaIntensity = False
            if len(self.figure_4_intensity.axes) !=0:
                for ax in self.figure_6.axes: ax.remove()
            if len(self.figure_4_intensity_time.axes) !=0:
                for ax in self.figure_6_time.axes: ax.remove()
            Ey, Ey_compa = 0,0
            Ex, Ex_compa = 0,0
            if boolList[1]:
                Ey_diag = self.S.Probe("Exy_intensity","Ey")
                self.compa_intensity_diag_name_str = "|Ey|^2"
                Ey = np.array(Ey_diag.getData()).astype(np.float32)
                Ey_diag_compa = self.compa_S.Probe("Exy_intensity","Ey")
                Ey_compa = np.array(Ey_diag_compa.getData()).astype(np.float32)
                intensity_dx = Ey_diag.getAxis("axis1")[:,0][1] - Ey_diag.getAxis("axis1")[:,0][0]
                any_diag = Ey_diag

            if boolList[0]:
                Ex_diag = self.S.Probe("Exy_intensity","Ex")
                self.compa_intensity_diag_name_str = "|Ex|^2"
                Ex = np.array(Ex_diag.getData()).astype(np.float32)
                Ex_diag_compa = self.compa_S.Probe("Exy_intensity","Ex")
                Ex_compa = np.array(Ex_diag_compa.getData()).astype(np.float32)
                intensity_dx = Ex_diag.getAxis("axis1")[:,0][1] - Ex_diag.getAxis("axis1")[:,0][0]
                any_diag = Ex_diag

            average_over = 2*l0 #number of optical period
            intensity_cmap = "jet"
            
            W = round(average_over/intensity_dx)
                        
            self.intensity_paxisX = any_diag.getAxis("axis1")[W:,0] #remove first W values
            if self.sim_geometry == "AMcylindrical":
                self.int_r_center = 0
            else:
                self.int_r_center = self.Ltrans/2
            self.intensity_paxisY = any_diag.getAxis("axis2")[:,1]-self.int_r_center
            self.intensity_paxisZ = any_diag.getAxis("axis3")[:,2]-self.int_r_center
            self.intensity_t_range = any_diag.getTimes()

            self.intensity_extentYZ = [self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0,
                                       self.intensity_paxisZ[0]/self.l0,self.intensity_paxisZ[-1]/self.l0]
            self.intensity_extentXY = [self.intensity_paxisX[0]/self.l0,self.intensity_paxisX[-1]/self.l0,
                                       self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0]
            self.intensity_extentTY = [self.intensity_t_range[0]/self.l0,self.intensity_t_range[-1]/self.l0,
                                       self.intensity_paxisY[0]/self.l0,self.intensity_paxisY[-1]/self.l0]
            if boolList[1]: del Ey_diag
            if boolList[0]: del Ex_diag
            del any_diag

            self.intensity_trans_mid_idx = len(self.intensity_paxisZ)//2
            self.compa_intensity_long_mid_idx = len(self.intensity_paxisX)//2
            self.compa_intensity_time_SLIDER.setMaximum(len(self.intensity_t_range)-1)
            self.compa_intensity_xcut_SLIDER.setMaximum(len(self.intensity_paxisX)-1)
            
            self.compa_intensity_time_SLIDER.setValue(len(self.intensity_t_range)-1)
            self.compa_intensity_xcut_SLIDER.setValue(len(self.intensity_paxisX)-1)
            
            self.compa_intensity_time_EDIT.setText(str(round(self.intensity_t_range[-1]/l0,2)))
            self.compa_intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[-1]/l0,2)))
            #COMPUTE INTENSITY 
            E_squared_cumsum = np.cumsum(Ey**2 + Ex**2,axis=1)
            E_compa_squared_cumsum = np.cumsum(Ey_compa**2 + Ex_compa**2,axis=1)
            self.intensity_data = (E_squared_cumsum[:,W:] - E_squared_cumsum[:,:-W]) / W
            self.intensity_data_compa = (E_compa_squared_cumsum[:,W:] - E_compa_squared_cumsum[:,:-W]) / W

            t1 = time.perf_counter()
            del E_squared_cumsum,E_compa_squared_cumsum
            gc.collect()
            print("Init INTENSITY COMPA:",(t1-t0),"s")
            
            max_intensity_vmax = np.max(self.intensity_data)
            max_intensity_vmax_compa = np.max(self.intensity_data_compa)
            
            #PLOT INTENSITY
            self.ax4_intensity_1a = self.figure_4_intensity.add_subplot(2,2,1)
            self.ax4_intensity_1b = self.figure_4_intensity.add_subplot(2,2,2)
            self.ax4_intensity_2a = self.figure_4_intensity.add_subplot(2,2,3)
            self.ax4_intensity_2b = self.figure_4_intensity.add_subplot(2,2,4)
            
            self.ax4_intensity_1time = self.figure_4_intensity_time.add_subplot(1,2,1)
            self.ax4_intensity_2time = self.figure_4_intensity_time.add_subplot(1,2,2)
            self.compa_intensity_im_1a = self.ax4_intensity_1a.imshow(self.intensity_data[-1,:,:,self.intensity_trans_mid_idx].T,aspect="auto", 
                                                    extent=self.intensity_extentXY,vmax=max_intensity_vmax,cmap=intensity_cmap)
            self.compa_intensity_im_1b = self.ax4_intensity_1b.imshow(self.intensity_data[-1,-1,:,:],aspect="auto", 
                                                    extent=self.intensity_extentYZ,vmax=max_intensity_vmax,cmap=intensity_cmap)
            
            self.compa_intensity_im_2a = self.ax4_intensity_2a.imshow(self.intensity_data_compa[-1,:,:,self.intensity_trans_mid_idx].T,aspect="auto", 
                                                    extent=self.intensity_extentXY,vmax=max_intensity_vmax_compa,cmap=intensity_cmap)
            self.compa_intensity_im_2b = self.ax4_intensity_2b.imshow(self.intensity_data_compa[-1,-1,:,:],aspect="auto", 
                                                    extent=self.intensity_extentYZ,vmax=max_intensity_vmax_compa,cmap=intensity_cmap)
            
            self.figure_4_intensity.colorbar(self.compa_intensity_im_1a, ax =self.ax4_intensity_1a,pad=0.01)
            self.figure_4_intensity.colorbar(self.compa_intensity_im_1b, ax =self.ax4_intensity_1b,pad=0.01)
            self.figure_4_intensity.colorbar(self.compa_intensity_im_2a, ax =self.ax4_intensity_2a,pad=0.01)
            self.figure_4_intensity.colorbar(self.compa_intensity_im_2b, ax =self.ax4_intensity_2b,pad=0.01)
            

            self.compa_intensity_line_x1 = self.ax4_intensity_1a.axvline(x=self.intensity_paxisX[-1]/l0,color="r",ls="--",alpha=0.5)
            self.compa_intensity_line_x2 = self.ax4_intensity_2a.axvline(x=self.intensity_paxisX[-1]/l0,color="r",ls="--",alpha=0.5)

            zR = 0.5*self.w0**2 #Rayleigh length
            self.waist_max_intensity = self.w0*sqrt(1+((self.intensity_paxisX-self.xfoc)/zR)**2)*sqrt(abs(self.l1)/2)
            self.waist_max_intensity_compa = self.compa_w0*sqrt(1+((self.intensity_paxisX-self.xfoc)/zR)**2)*sqrt(abs(self.compa_l1)/2)
            """
            To convert from intensity to a0: a = np.sqrt(2*I)*np.exp(0.5), for |l| = 1
            """
            self.ax4_intensity_1a.plot(self.intensity_paxisX/l0,self.waist_max_intensity/l0, color="k", ls="--")
            self.ax4_intensity_1a.plot(self.intensity_paxisX/l0,-self.waist_max_intensity/l0, color="k", ls="--")
            self.ax4_intensity_2a.plot(self.intensity_paxisX/l0,self.waist_max_intensity_compa/l0, color="k", ls="--")
            self.ax4_intensity_2a.plot(self.intensity_paxisX/l0,-self.waist_max_intensity_compa/l0, color="k", ls="--")
            self.circle_max_compa_intensity1 = plt.Circle((0, 0), self.waist_max_intensity[-1], fill=False, ec="k",ls="--")
            self.circle_max_compa_intensity2 = plt.Circle((0, 0), self.waist_max_intensity_compa[-1], fill=False, ec="k",ls="--")
            self.ax4_intensity_1b.add_patch(self.circle_max_compa_intensity1)
            self.ax4_intensity_2b.add_patch(self.circle_max_compa_intensity2)

            self.figure_4_intensity.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | $t={self.intensity_t_range[-1]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[-1]/l0:.2f}~\lambda$",**self.qss_plt_title)
            self.figure_4_intensity.tight_layout()
            self.canvas_4_intensity.draw()
            
            empirical_corr = 0.98 # To compensate for other effect on top of dispersion
            groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) # Formula: vg = c^2k/sqrt(wp^2+c^2k^2)
            laser_x_pos_range = np.max([groupe_velocity*self.intensity_t_range-self.Tp/2,self.intensity_t_range*0],axis=0)          
            self.intensity_data_time = np.empty((self.intensity_data.shape[0],self.intensity_data.shape[2]))
            self.intensity_data_time_compa = np.empty((self.intensity_data_compa.shape[0],self.intensity_data_compa.shape[2]))

            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            max_intensity_compa = np.max(self.intensity_data_compa[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))

            for t,laser_x_pos in enumerate(laser_x_pos_range):
                xcut_idx = np.where(self.intensity_data[t,:,:,self.intensity_trans_mid_idx]==max_intensity[t])[0][0] #Use maximum intensity
                xcut_idx_compa = np.where(self.intensity_data_compa[t,:,:,self.intensity_trans_mid_idx]==max_intensity_compa[t])[0][0] #Use maximum intensity
                if self.intensity_use_vg_CHECK.isChecked(): 
                    xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0] #Use group velocity
                    xcut_idx_compa = xcut_idx
                self.intensity_data_time[t] = self.intensity_data[t,xcut_idx,:,self.intensity_trans_mid_idx]
                self.intensity_data_time_compa[t] = self.intensity_data_compa[t,xcut_idx_compa,:,self.intensity_trans_mid_idx]
            
            self.compa_intensity_im_1time = self.ax4_intensity_1time.imshow(self.intensity_data_time.T, cmap="jet",aspect="auto",
                                                                            extent=self.intensity_extentTY,vmax=max_intensity_vmax)
            self.compa_intensity_im_2time = self.ax4_intensity_2time.imshow(self.intensity_data_time_compa.T, cmap="jet",aspect="auto",
                                                                            extent=self.intensity_extentTY,vmax=max_intensity_vmax_compa)

            self.figure_4_intensity_time.colorbar(self.compa_intensity_im_1time, ax = self.ax4_intensity_1time, pad=0.01)
            self.figure_4_intensity_time.colorbar(self.compa_intensity_im_2time, ax = self.ax4_intensity_2time, pad=0.01)

            self.figure_4_intensity_time.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | Time evolution of pulse center (use group velocity: {self.compa_intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)

            self.ax4_intensity_1time.set_xlabel("t/t0")
            self.ax4_intensity_2time.set_xlabel("t/t0")
            self.figure_4_intensity_time.tight_layout()
            self.canvas_4_intensity_time.draw()
    
        
        if check_id == 2000:
            if self.compa_intensity_spatial_time_BOX.currentIndex() == 1:
                self.canvas_4_intensity.hide()
                self.plt_toolbar_4_intensity.hide()                
                self.canvas_4_intensity_time.show()
                self.plt_toolbar_4_intensity_time.show()                

            else:
                self.canvas_4_intensity.show()
                self.plt_toolbar_4_intensity.show()                
                self.canvas_4_intensity_time.hide()
                self.plt_toolbar_4_intensity_time.hide()
            return

        if check_id < 10:
            Ey, Ey_compa = 0,0
            Ex, Ex_compa = 0,0
            t0 = time.perf_counter()

            if boolList[1]:
                Ey_diag = self.S.Probe("Exy_intensity","Ey")
                self.compa_intensity_diag_name_str = "|Ey|^2"
                Ey = np.array(Ey_diag.getData()).astype(np.float32)
                Ey_diag_compa = self.compa_S.Probe("Exy_intensity","Ey")
                Ey_compa = np.array(Ey_diag_compa.getData()).astype(np.float32)
                intensity_dx = Ey_diag.getAxis("axis1")[:,0][1] - Ey_diag.getAxis("axis1")[:,0][0]
                any_diag = Ey_diag

            if boolList[0]:
                Ex_diag = self.S.Probe("Exy_intensity","Ex")
                self.compa_intensity_diag_name_str = "|Ex|^2"
                Ex = np.array(Ex_diag.getData()).astype(np.float32)
                Ex_diag_compa = self.compa_S.Probe("Exy_intensity","Ex")
                Ex_compa = np.array(Ex_diag_compa.getData()).astype(np.float32)
                intensity_dx = Ex_diag.getAxis("axis1")[:,0][1] - Ex_diag.getAxis("axis1")[:,0][0]
                any_diag = Ex_diag
            t1 = time.perf_counter()

            average_over = 2*l0 #number of optical period
            intensity_cmap = "jet"
            
            W = round(average_over/intensity_dx)
            
            #COMPUTE INTENSITY 
            E_squared_cumsum = np.cumsum(Ey**2 + Ex**2,axis=1)
            E_compa_squared_cumsum = np.cumsum(Ey_compa**2 + Ex_compa**2,axis=1)
            self.intensity_data = (E_squared_cumsum[:,W:] - E_squared_cumsum[:,:-W]) / W
            self.intensity_data_compa = (E_compa_squared_cumsum[:,W:] - E_compa_squared_cumsum[:,:-W]) / W

            t2 = time.perf_counter()
            del E_squared_cumsum,E_compa_squared_cumsum
            gc.collect()
            print("load Ex/Ey in RAM:",(t1-t0),"s | ","Compute intensity:",(t2-t1),"s")
            
            max_intensity_vmax1 = np.max(self.intensity_data)
            max_intensity_vmax2 = np.max(self.intensity_data_compa)

            
            #PLOT INTENSITY
            time_idx = self.compa_intensity_time_SLIDER.sliderPosition()
            xcut_idx = self.compa_intensity_xcut_SLIDER.sliderPosition()

            self.compa_intensity_im_1a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.compa_intensity_im_1b.set_data(self.intensity_data[time_idx,xcut_idx,:,:])
            self.compa_intensity_im_1a.set_clim(vmax=max_intensity_vmax1)
            self.compa_intensity_im_1b.set_clim(vmax=max_intensity_vmax1)
            
            self.compa_intensity_im_2a.set_data(self.intensity_data_compa[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.compa_intensity_im_2b.set_data(self.intensity_data_compa[time_idx,xcut_idx,:,:])
            self.compa_intensity_im_2a.set_clim(vmax=max_intensity_vmax2)
            self.compa_intensity_im_2b.set_clim(vmax=max_intensity_vmax2)

            self.figure_4_intensity.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | $t={self.intensity_t_range[-1]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[-1]/l0:.2f}~\lambda$",**self.qss_plt_title)
            self.figure_4_intensity.tight_layout()
            self.canvas_4_intensity.draw()
            check_id = 5000 #Update time distribution
        
        elif check_id <= 110 or check_id==200 or check_id==201: #SLIDER OR LINE_EDIT UPDATE
            self.timer = time.perf_counter()
            if check_id == 101: #QLineEdit changed
                time_edit_value = float(self.compa_intensity_time_EDIT.text())
                time_idx = np.where(abs(self.intensity_t_range/l0-time_edit_value)==np.min(abs(self.intensity_t_range/l0-time_edit_value)))[0][0]
                self.compa_intensity_time_SLIDER.setValue(time_idx)
                self.compa_intensity_time_EDIT.setText(str(round(self.compa_intensity_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.compa_intensity_time_SLIDER.sliderPosition()
                self.compa_intensity_time_EDIT.setText(str(round(self.intensity_t_range[time_idx]/l0,2)))

            if check_id == 201:#QSlider changed
                xcut_edit_value = float(self.compa_intensity_xcut_EDIT.text())
                xcut_idx = np.where(abs(self.intensity_paxisX/l0-xcut_edit_value)==np.min(abs(self.intensity_paxisX/l0-xcut_edit_value)))[0][0]
                self.compa_intensity_xcut_SLIDER.setValue(xcut_idx)
            else:
                xcut_idx = self.compa_intensity_xcut_SLIDER.sliderPosition()
                self.compa_intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[xcut_idx]/l0,2)))
            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            max_intensity_compa = np.max(self.intensity_data_compa[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            if self.compa_intensity_follow_laser_CHECK.isChecked():
                # empirical_corr = 0.98 #to compensate for other effect on top of dis
                # groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) #c^2k/sqrt(wp^2+c^2k^2)
                # laser_x_pos = max(groupe_velocity*self.intensity_t_range[time_idx]-self.Tp/2,0)
                # xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0]
                
                xcut_idx = np.where(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx]==max_intensity[time_idx])[0][0] #Use maximum intensity
                xcut_idx_compa = np.where(self.intensity_data_compa[time_idx,:,:,self.intensity_trans_mid_idx]==max_intensity_compa[time_idx])[0][0] #Use maximum intensity
                
                self.compa_intensity_xcut_SLIDER.setValue(xcut_idx)
            
            self.compa_intensity_im_1a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.compa_intensity_im_1b.set_data(self.intensity_data[time_idx,xcut_idx,:,:])
            self.compa_intensity_im_2a.set_data(self.intensity_data_compa[time_idx,:,:,self.intensity_trans_mid_idx].T)
            self.compa_intensity_im_2b.set_data(self.intensity_data_compa[time_idx,xcut_idx,:,:])
            self.compa_intensity_line_x1.set_xdata(self.intensity_paxisX[xcut_idx]/l0)
            self.compa_intensity_line_x2.set_xdata(self.intensity_paxisX[xcut_idx_compa]/l0)
            self.circle_max_compa_intensity1.set_radius(self.waist_max_intensity[xcut_idx]/l0)
            self.circle_max_compa_intensity2.set_radius(self.waist_max_intensity_compa[xcut_idx]/l0)

            self.figure_4_intensity.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | $t={self.intensity_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[xcut_idx]/l0:.2f}~\lambda$",**self.qss_plt_title)

            self.canvas_4_intensity.draw()
        elif check_id == 1000:

            if self.loop_in_process: return

            self.loop_in_process = True

            xcut_idx = self.compa_intensity_xcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.intensity_t_range)):
                
                empirical_corr = 0.98 #to compensate for other effect on top of dis
                groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) #c^2k/sqrt(wp^2+c^2k^2)
                laser_x_pos = max(groupe_velocity*self.intensity_t_range[time_idx]-self.Tp/2,0)
                laser_x_pos_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0]
                
                self.compa_intensity_time_SLIDER.setValue(time_idx)
                self.compa_intensity_xcut_SLIDER.setValue(laser_x_pos_idx)
                self.compa_intensity_time_EDIT.setText(str(round(self.intensity_t_range[time_idx]/l0,2)))
                self.compa_intensity_xcut_EDIT.setText(str(round(self.intensity_paxisX[laser_x_pos_idx]/l0,2)))
              
                self.compa_intensity_im_1a.set_data(self.intensity_data[time_idx,:,:,self.intensity_trans_mid_idx].T)
                self.compa_intensity_im_1b.set_data(self.intensity_data[time_idx,laser_x_pos_idx,:,:])
                self.compa_intensity_im_2a.set_data(self.intensity_data_compa[time_idx,:,:,self.intensity_trans_mid_idx].T)
                self.compa_intensity_im_2b.set_data(self.intensity_data_compa[time_idx,laser_x_pos_idx,:,:])
                self.compa_intensity_line_x1.set_xdata(self.intensity_paxisX[laser_x_pos_idx]/l0)
                self.compa_intensity_line_x2.set_xdata(self.intensity_paxisX[laser_x_pos_idx]/l0)

                self.circle_max_compa_intensity1.set_radius(self.waist_max_intensity[xcut_idx]/l0)
                self.circle_max_compa_intensity2.set_radius(self.waist_max_intensity_compa[xcut_idx]/l0)

                self.figure_4_intensity.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | $t={self.intensity_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.intensity_paxisX[xcut_idx]/l0:.2f}~\lambda$",**self.qss_plt_title)
                self.canvas_4_intensity.draw()
                time.sleep(0.1)
                app.processEvents()

            self.loop_in_process = False
        if check_id == 5000:
            max_intensity_vmax1 = np.max(self.intensity_data)
            max_intensity_vmax2 = np.max(self.intensity_data_compa)

            empirical_corr = 0.98 # To compensate for other effect on top of dis
            groupe_velocity = empirical_corr*1/np.sqrt(self.ne+1) # Formula: vg = c^2k/sqrt(wp^2+c^2k^2)
            laser_x_pos_range = np.max([groupe_velocity*self.intensity_t_range-self.Tp/2,self.intensity_t_range*0],axis=0)          
            self.intensity_data_time = np.empty((self.intensity_data.shape[0],self.intensity_data.shape[2]))
            self.intensity_data_time_compa = np.empty((self.intensity_data_compa.shape[0],self.intensity_data_compa.shape[2]))

            max_intensity = np.max(self.intensity_data[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            max_intensity_compa = np.max(self.intensity_data_compa[:,:,:,self.intensity_trans_mid_idx],axis=(1,2))
            for t,laser_x_pos in enumerate(laser_x_pos_range):
                xcut_idx = np.where(self.intensity_data[t,:,:,self.intensity_trans_mid_idx]==max_intensity[t])[0][0] #Use maximum intensity
                xcut_idx_compa = np.where(self.intensity_data_compa[t,:,:,self.intensity_trans_mid_idx]==max_intensity_compa[t])[0][0] #Use maximum intensity
                if self.compa_intensity_use_vg_CHECK.isChecked(): 
                    xcut_idx = np.where(np.abs(self.intensity_paxisX-laser_x_pos) == np.min(np.abs(self.intensity_paxisX-laser_x_pos)))[0][0] #Use group velocity
                    xcut_idx_compa = xcut_idx
                self.intensity_data_time[t] = self.intensity_data[t,xcut_idx,:,self.intensity_trans_mid_idx]
                self.intensity_data_time_compa[t] = self.intensity_data_compa[t,xcut_idx_compa,:,self.intensity_trans_mid_idx]
            self.compa_intensity_im_1time.set_data(self.intensity_data_time.T)
            self.compa_intensity_im_2time.set_data(self.intensity_data_time_compa.T)
            self.compa_intensity_im_1time.set_clim(vmax=max_intensity_vmax1)
            self.compa_intensity_im_2time.set_clim(vmax=max_intensity_vmax2)
            self.figure_4_intensity_time.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} {self.compa_intensity_diag_name_str} | Time evolution of pulse center (use group velocity: {self.compa_intensity_use_vg_CHECK.isChecked()})",**self.qss_plt_title)
            self.figure_4_intensity_time.tight_layout()
            self.canvas_4_intensity_time.draw()
        self.updateInfoLabel()
        
    def onRemoveScalar(self):
        if not self.INIT_tabScalar: #IF TAB OPEN AND SIM LOADED
            gc.collect()
        self.updateInfoLabel()

    def onRemoveFields(self):
        if not self.INIT_tabFields: #IF TAB OPEN AND SIM LOADED
            del self.fields_image_list, self.fields_data_list,self.fields_paxisX,self.fields_paxisY,self.fields_paxisZ,
            self.extentXY,self.extentXY,self.fields_t_range,self.fields_trans_mid_idx,self.fields_long_mid_idx
            gc.collect()
        self.updateInfoLabel()

    def onRemoveTrack(self):
        if not self.INIT_tabTrack:
            del self.track_N, self.track_t_range,self.track_traj,self.x,self.y,self.z,self.px,self.py,self.pz,self.r,self.Lx_track
            gc.collect()
        self.updateInfoLabel()

    def onRemovePlasma(self):
        for currentIndex in range(self.programm_TABS.count()):
            if self.programm_TABS.tabText(currentIndex) == "PLASMA" or self.programm_TABS.tabText(currentIndex) == "COMPA": return
        print("no COMPA or PLASMA: allowed to remove plasma diags")
        if not self.INIT_tabPlasma: #IF TAB OPEN AND SIM LOADED
            del self.plasma_data_list, self.plasma_paxisX_long, self.plasma_paxisY_long, self.plasma_t_range, self.plasma_paxisY,
            self.plasma_paxisZ, self.plasma_paxisX_Bx, self.plasma_extentXY_long , self.plasma_extentYZ
            gc.collect()
        self.updateInfoLabel()
    
    def onRemoveIntensity(self):
        if self.INIT_tabIntensity == False:
            del self.intensity_data, self.intensity_data_time, self.intensity_im_a,self.intensity_im_b
            gc.collect()
        self.updateInfoLabel()
    
    def onRemoveCompa(self):
        if self.INIT_tabPlasma == False and self.INIT_tabPlasma is not None:
            del self.compa_plasma_paxisX_long,self.compa_plasma_paxisY_long,self.compa_plasma_t_range,self.compa_plasma_paxisY,
            self.compa_plasma_paxisZ,self.compa_plasma_paxisX_Bx, self.compa_plasma_extentXY_long, self.compa_plasma_extentYZ
            gc.collect()
        self.updateInfoLabel()

    def onRemoveBinning(self):
        if self.is_sim_loaded:
            del self.binning_t_range, self.compa_binning_t_range, self.binning_data_list,self.binning_image_list

            try:
                del self.compa_binning_image, self.compa_binning_data, self.compa_binning_image2, self.compa_binning_data2
            except: pass
        self.updateInfoLabel()

    def onRemoveTornado(self):
        gc.collect()
        self.updateInfoLabel()

    def onUpdateTabTrack(self, check_id):

        if self.INIT_tabFields == None or self.is_sim_loaded == False:
            # Popup().showError("Simulation not loaded")
            return
        l0 = 2*pi
        if self.INIT_tabTrack or check_id==-1: #if change of name reinit
            print("===== INIT TRACK TAB =====")

            track_name = self.track_file_BOX.currentText()
            
            # self.displayLoadingLabel()
            app.processEvents()
            try:
                T0 = self.S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
            except Exception:
                utils.Popup().showError("No TrackParticles diagnostic found")
                return

            self.track_N_tot = T0.nParticles
            self.track_t_range = T0.getTimes()
            self.track_traj = T0.getData()
            self.track_time_SLIDER.setMaximum(len(self.track_t_range)-1)
            self.track_time_SLIDER.setValue(len(self.track_t_range)-1)
            # self.extentXY = [-2*self.w0,2*self.w0,-2*self.w0,2*self.w0]

            del T0
            N_part = int(self.track_Npart_EDIT.text())
            self.x = self.track_traj["x"][:,::N_part]
            self.track_N = self.x.shape[1]
            
            if self.sim_geometry == "AMcylindrical":
                self.track_r_center = 0
            else:
                self.track_r_center = self.Ltrans/2
            
            self.y = self.track_traj["y"][:,::N_part] -self.track_r_center
            self.z = self.track_traj["z"][:,::N_part] -self.track_r_center
            self.py = self.track_traj["py"][:,::N_part]
            self.pz = self.track_traj["pz"][:,::N_part]
            self.px = self.track_traj["px"][:,::N_part]
            self.r = np.sqrt(self.y**2 + self.z**2)
            self.pr = (self.y*self.py + self.z*self.pz)/self.r

            self.Lx_track =  self.y*self.pz - self.z*self.py
            self.theta = np.arctan2(self.z,self.y)
            self.gamma = np.sqrt(1+self.px**2+self.py**2+self.pz**2)


            byte_size_track = getsizeof(self.Lx_track) + getsizeof(self.r)
            + getsizeof(self.x)+getsizeof(self.y) + getsizeof(self.z)
            + getsizeof(self.px)+getsizeof(self.py) + getsizeof(self.pz)
            del self.track_traj
            gc.collect()

            print("Memory from TRACK:",round(byte_size_track*10**-6,1),"MB (",round(byte_size_track*100/psutil.virtual_memory().total,1),"%)")

            self.INIT_tabTrack = False
            # self.loading_LABEL.deleteLater()
            app.processEvents()
            self.updateInfoLabel()

        if check_id <= 0:
            if len(self.figure_2.axes) !=0:
                for ax in self.figure_2.axes: ax.remove()
                for ax in self.figure_2_displace.axes: ax.remove()

            ax1,ax2 = self.figure_2.subplots(1,2)
            time0 = time.perf_counter()
            mean_coef = 5
            self.track_radial_distrib_im = ax1.scatter(self.r[0]/l0,self.Lx_track[-1],s=1,label="$L_x$")
            
            ax1.set_xlabel("$r/\lambda$")
            ax1.set_ylabel("$L_x$")
            a_range_r,MLx = self.averageAM(self.r[0], self.Lx_track[-1], 0.5)
            ax1.plot(a_range_r/l0, MLx*mean_coef,"r",label="5<$L_x$>",alpha=0.2)
            
            r_range = np.arange(0,2*self.w0,0.1)
            theta_range = np.arange(0,2*pi,0.01)
            R_grid, Theta_grid = np.meshgrid(r_range,theta_range)
            z_foc_lz = np.mean(self.x[0])
            Tint = 3/8*self.Tp
            Lx2_model = np.max(self.LxEpolar_V2_O3(R_grid,Theta_grid,z_foc_lz,self.w0,self.a0,Tint),axis=0)
            ax1.plot(r_range/l0,Lx2_model,"k--",alpha=1)
            ax1.plot(r_range/l0,-Lx2_model,"k--",alpha=1, label="Model $L_z^{NR}$")

            ax1.grid()
            ax1.legend()
            vmax = 1.25*np.nanstd(self.Lx_track[-1])

            self.track_trans_distrib_im = ax2.scatter(self.y[0]/l0,self.z[0]/l0,s=1, c=self.Lx_track[-1], vmin=-vmax,vmax=vmax, cmap="RdYlBu")
            self.figure_2.colorbar(self.track_trans_distrib_im,ax=ax2,pad=0.01)
            self.figure_2.suptitle(f"{self.sim_directory_name}: {track_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.2f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
            self.figure_2.tight_layout()
            self.canvas_2.draw()
            
            self.ax1_displace,self.ax2_displace,self.ax3_displace,self.ax4_displace = self.figure_2_displace.subplots(2,2).ravel()
            self.ax1_displace.set_title("x-x0")
            self.ax2_displace.set_title("r-r0")
            self.ax3_displace.set_title("theta-theta0")
            self.ax4_displace.set_title("pr")
            self.ax1_displace.grid()
            self.ax2_displace.grid()
            self.ax3_displace.grid()
            self.ax4_displace.grid()
            self.track_displace_x = self.ax1_displace.scatter(self.r[0]/l0,(self.x[-1]-self.x[0])/l0,s=1)
            self.track_displace_r = self.ax2_displace.scatter(self.r[0]/l0,(self.r[-1]-self.r[0]),s=1)
            self.track_displace_theta = self.ax3_displace.scatter(self.r[0]/l0,(self.theta[-1]-self.theta[0]),s=1)
            self.track_displace_pr = self.ax4_displace.scatter(self.r[0]/l0,self.pr[-1],s=1)
            x_pos = np.min(self.x[0])

            self.track_displace_pr_model, = self.ax4_displace.plot(r_range/l0, -self.a0**2/4*self.f_squared_prime(r_range,x_pos-self.xfoc)*3/8*self.Tp,"r--",label="Model")
            
            r_range_dr = np.arange(self.w0/sqrt(2),2*self.w0,0.1)
            
                
            pr_end = -self.a0**2/4*self.f_squared_prime(r_range_dr, x_pos-self.xfoc)*3/8*self.Tp
            self.track_displace_r_model, = self.ax2_displace.plot(r_range_dr/l0, pr_end*(self.track_t_range[-1]-self.Tp-x_pos),"r--",label="Model")
            
            
            # self.figure_2.colorbar(self.track_trans_distrib_im,ax=ax2,pad=0.01)
            self.figure_2_displace.suptitle(f"{self.sim_directory_name}: {track_name}| $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.1f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
            self.ax2_displace.legend()
            self.ax4_displace.legend()
            self.figure_2_displace.tight_layout()
            self.canvas_2_displace.draw()

            
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
            self.track_trans_distrib_im.set_array(self.Lx_track[time_idx])
            if self.track_update_offset_CHECK.isChecked():
                self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                self.track_radial_distrib_im.set_offsets(np.c_[self.r[time_idx]/l0,self.Lx_track[time_idx]])
            else:
                self.track_radial_distrib_im.set_offsets(np.c_[self.r[0]/l0,self.Lx_track[time_idx]])
            
            combo_box_index = self.track_pannel_BOX.currentIndex()
            track_name = self.track_file_BOX.currentText()
            if combo_box_index==1:
                r_range = np.arange(0,2*self.w0,0.1)

                self.track_displace_x.set_offsets(np.c_[self.r[0]/l0,(self.x[time_idx]-self.x[0])/l0])
                self.track_displace_r.set_offsets(np.c_[self.r[0]/l0,(self.r[time_idx]-self.r[0])])
                self.track_displace_theta.set_offsets(np.c_[self.r[0]/l0,(self.theta[time_idx]-self.theta[0])])
                self.track_displace_pr.set_offsets(np.c_[self.r[0]/l0,self.pr[time_idx]])
                self.track_displace_pr_model.set_ydata(-self.a0**2/4*self.f_squared_prime(r_range,0)*3/8*self.Tp)
                # self.track_displace_r_model.set_ydata(np.abs(-self.a0**2/4*self.f_squared_prime(r_range,0)*(self.track_t_range[time_idx])**2/2)/l0)
                self.figure_2_displace.suptitle(f"{self.sim_directory_name}: {track_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.1f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
                self.canvas_2_displace.draw()
                
            self.figure_2.suptitle(f"{self.sim_directory_name}: {track_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.1f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
            self.canvas_2.draw()

        elif check_id == 1000: #PLAY ANIMATION
            track_name = self.track_file_BOX.currentText()
            combo_box_index = self.track_file_BOX.currentIndex()
            if combo_box_index==0:
                anim_speed = 0.001 #high time resolution = higher refresh rate
                every_frame = 10
            if self.loop_in_process: return
            self.loop_in_process = True
            for time_idx in range(0,len(self.track_t_range),every_frame):
                self.track_time_SLIDER.setValue(time_idx)
                if self.track_update_offset_CHECK.isChecked(): self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                self.track_trans_distrib_im.set_array(self.Lx_track[time_idx])
                if self.track_update_offset_CHECK.isChecked():
                    self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                    self.track_radial_distrib_im.set_offsets(np.c_[self.r[time_idx]/l0,self.Lx_track[time_idx]])
                else:
                    self.track_radial_distrib_im.set_offsets(np.c_[self.r[0]/l0,self.Lx_track[time_idx]])

                self.figure_2.suptitle(f"{self.sim_directory_name}: {track_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.1f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
                self.canvas_2.draw()
                time.sleep(anim_speed)
                app.processEvents()
            self.loop_in_process = False
        
        elif check_id == 5000:
            combo_box_index = self.track_pannel_BOX.currentIndex()
            if combo_box_index==1:
                self.canvas_2.hide()
                self.plt_toolbar_2.hide()
                self.canvas_2_displace.show()
                self.plt_toolbar_2_displace.show()
            else:
                self.canvas_2.show()
                self.plt_toolbar_2.show()
                self.canvas_2_displace.hide()
                self.plt_toolbar_2_displace.hide()
            self.figure_2.tight_layout()
            self.figure_2_displace.tight_layout()
        self.updateInfoLabel()
        return
    
    
    def onUpdateTabCompaTrack(self,check_id):
        print(self.is_sim_loaded,self.is_compa_sim_loaded)
        if  self.is_sim_loaded == False or self.is_compa_sim_loaded==False: return
        
        if self.INIT_tabCompaTrack !=False: 
            self.INIT_tabCompaTrack=False
            track_name = self.track_file_BOX.currentText()
            app.processEvents()
            try:
                print("Open 1st track comparison")
                T0 = self.S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
            except Exception:
                utils.Popup().showError("No TrackParticles diagnostic found")
                return
            self.track_N_tot = T0.nParticles
            self.track_t_range = T0.getTimes()
            self.track_traj = T0.getData()
            self.compa_track_time_SLIDER.setMaximum(len(self.track_t_range)-1)
            self.compa_track_time_SLIDER.setValue(len(self.track_t_range)-1)
            del T0
            N_part = int(self.track_Npart_EDIT.text())
            self.x = self.track_traj["x"][:,::N_part]
            self.track_N = self.x.shape[1]
            
            if self.sim_geometry == "AMcylindrical":
                self.track_r_center = 0
            else:
                self.track_r_center = self.Ltrans/2
            
            self.y = self.track_traj["y"][:,::N_part] -self.track_r_center
            self.z = self.track_traj["z"][:,::N_part] -self.track_r_center
            self.py = self.track_traj["py"][:,::N_part]
            self.pz = self.track_traj["pz"][:,::N_part]
            self.px = self.track_traj["px"][:,::N_part]
            self.r = np.sqrt(self.y**2 + self.z**2)
            self.pr = (self.y*self.py + self.z*self.pz)/self.r
    
            self.Lx_track =  self.y*self.pz - self.z*self.py
            self.theta = np.arctan2(self.z,self.y)
            self.gamma = np.sqrt(1+self.px**2+self.py**2+self.pz**2)
            l0 = 2*pi
            
            # self.ax4_track = self.figure_4_track.add_subplot(1,1,1)

            self.compa_track_radial_distrib_im = self.ax4_track.scatter(self.r[0]/l0, self.Lx_track[-1],s=1,label=f"{self.sim_directory_name}")
            print("draw 1st track comparison")
            self.canvas_4_track.draw()
            
            try:
                print("Open 2nd track comparison")
                T0 = self.compa_S.TrackParticles(track_name, axes=["x","y","z","py","pz","px"])
            except Exception:
                utils.Popup().showError("No TrackParticles diagnostic found")
                return
            self.compa_track_N_tot = T0.nParticles
            self.compa_track_t_range = T0.getTimes()
            self.compa_track_traj = T0.getData()
            self.compa_track_time_SLIDER.setMaximum(len(self.track_t_range)-1)
            self.compa_track_time_SLIDER.setValue(len(self.track_t_range)-1)
            del T0
            N_part = int(self.compa_track_Npart_EDIT.text())
            self.x = self.track_traj["x"][:,::N_part]
            self.track_N = self.x.shape[1]
            
            if self.compa_sim_geometry == "AMcylindrical":
                self.compa_track_r_center = 0
            else:
                self.compa_track_r_center = self.Ltrans/2
            
            self.y2 = self.track_traj["y"][:,::N_part] -self.compa_track_r_center
            self.z2 = self.track_traj["z"][:,::N_part] -self.compa_track_r_center
            self.py2 = self.track_traj["py"][:,::N_part]
            self.pz2 = self.track_traj["pz"][:,::N_part]
            self.px2 = self.track_traj["px"][:,::N_part]
            self.r2 = np.sqrt(self.y**2 + self.z**2)
    
            self.Lx_track2 =  self.y2*self.pz2 - self.z2*self.py2
            
                
            self.compa_track_radial_distrib_im2 = self.ax4_track.scatter(self.r2[0]/l0, self.Lx_track2[-1],s=1,label=f"{self.compa_sim_directory_name}")
            self.ax4_track.grid()
            self.ax4_track.legend()
            
            self.canvas_4_track.draw()
            print("draw 1st track comparison")

        if check_id == 100 or check_id==101:#SLIDER UPDATE
            l0=2*pi

            if check_id == 101:
                time_edit = float(self.compa_track_time_EDIT.text())
                time_idx = np.where(abs(self.compa_track_t_range/l0-time_edit)==np.min(abs(self.compa_track_t_range/l0-time_edit)))[0][0]
                self.compa_track_time_SLIDER.setValue(time_idx)
                self.compa_track_time_EDIT.setText(str(round(self.compa_track_t_range[time_idx]/l0,2)))
            else:
                time_idx = self.compa_track_time_SLIDER.sliderPosition()
                self.compa_track_time_EDIT.setText(str(round(self.compa_track_t_range[time_idx]/l0,2)))

            self.compa_track_radial_distrib_im.set_offsets(np.c_[self.r[0]/l0,self.Lx_track[time_idx]])
            self.compa_track_radial_distrib_im2.set_offsets(np.c_[self.r2[0]/l0,self.Lx_track2[time_idx]])
                
            self.figure_4_track.suptitle(f"{self.sim_directory_name} vs {self.compa_sim_directory_name}: {track_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (dx=λ/{l0/self.S.namelist.dx:.0f}; a0={self.a0:.1f}; N={self.track_N/1000:.2f}k; <x0>={np.mean(self.x[0])/l0:.1f}λ)",**self.qss_plt_title)
            self.canvas_4_track.draw()
            
    
    
    def averageAM(self, X,Y,dr_av):
        M = []
        da = 0.04
        t0 = time.perf_counter()
        print("Computing average...",da)
        a_range = np.arange(0,np.nanmax(X)*1.0+da,da)
        M = np.empty(a_range.shape)
        for i,a in enumerate(a_range):
            mask = (X > a-dr_av/2) & (X < a+dr_av/2)
            M[i] = np.nanmean(Y[mask])
        t1 = time.perf_counter()
        print(f"...{(t1-t0):.0f} s")
        return a_range,M

    def savePlasmaData(self):
        """Call get Plasma Probe data for all plasma names to save the files to .npz """
        return
        self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, self.plasma_names[:2])
        self.loadthread.start()
        # self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, selected_plasma_names=self.plasma_names[5:])
        # self.loadthread.start()
        return

    def onUpdateTabPlasmaFigure(self, plasma_data_list_used_selected_plasma_names):
        if len(plasma_data_list_used_selected_plasma_names)==1:
            utils.Popup().showError(f"Could not find Plasma Diagnostic\n{plasma_data_list_used_selected_plasma_names}")
            return
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
        contains_av = sum(["_av" in name for name in selected_plasma_names]) != 0
        print("contains_av:",contains_av)

        if contains_av: #if contains averaged quantities, change slider range
            self.effective_plasma_t_range = self.av_plasma_t_range
        else:
            self.effective_plasma_t_range = self.plasma_t_range

        self.plasma_time_SLIDER.setMaximum(len(self.effective_plasma_t_range)-1)
        self.plasma_time_SLIDER.setValue(len(self.effective_plasma_t_range)-1)
        self.plasma_time_EDIT.setText(str(round(self.effective_plasma_t_range[-1]/l0,2)))

        ne = self.S.namelist.ne
        self.toTesla = 10709

        VMAX_Bx = 0.001*self.toTesla*self.a0*ne/0.01 #1 = 10709T
        vmax_ptheta = 0.005

        #=====================================
        # REMOVE ALL AXES --> NOT OPTIMAL !
        #=====================================
        if len(self.figure_3.axes) !=0:
            for ax in self.figure_3.axes: ax.remove()

        slider_time_idx = self.plasma_time_SLIDER.sliderPosition()
        x_idx = self.plasma_xcut_SLIDER.sliderPosition()
        k=0
        for i in range(len(self.plasma_names)):
            if boolList[i]:
                if only_trans:
                    ax = self.figure_3.add_subplot(1,Naxis,k+1)
                elif only_long:
                    ax = self.figure_3.add_subplot(Naxis,1,k+1)
                elif Naxis <= 2:
                    ax = self.figure_3.add_subplot(1,Naxis,k+1)
                else:
                    ax = self.figure_3.add_subplot(2,2,k+1)

                if "_av" not in self.plasma_names[i] and contains_av:
                    current_time = self.av_plasma_t_range[slider_time_idx]
                    new_time_idx = np.where(np.abs(current_time - self.plasma_t_range) == np.min(np.abs(current_time - self.plasma_t_range)))[0][0]
                    time_idx = new_time_idx
                else: time_idx = slider_time_idx


                if "Bx" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin, vmax = -VMAX_Bx, VMAX_Bx
                elif "pθ" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin, vmax = -vmax_ptheta,vmax_ptheta
                elif "ne" in self.plasma_names[i] or "ni" in self.plasma_names[i]:
                    cmap = "jet"
                    vmin, vmax = 0,3
                elif "Ekin" in self.plasma_names[i]:
                    cmap = "smilei"
                    vmin, vmax = 0, 0.1
                elif "jx" in self.plasma_names[i] or "px" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin, vmax = -0.01, 0.01
                else:
                    cmap = "RdYlBu"
                    vmin = -0.05*np.max(np.abs(self.plasma_data_list[k][time_idx]))
                    vmax =  0.05*np.max(np.abs(self.plasma_data_list[k][time_idx]))

                if "trans" in self.plasma_names[i]:
                    extent = self.plasma_extentYZ
                    data = self.plasma_data_list[k][time_idx,x_idx,:,:]
                else:
                    extent = self.plasma_extentXY_long
                    data = self.plasma_data_list[k][time_idx].T

                im = ax.imshow(data, aspect="auto",
                                origin="lower", cmap = cmap, extent=extent, vmin=vmin, vmax=vmax) #bwr, RdYlBu #
                ax.set_title(self.plasma_names[i])


                self.figure_3.colorbar(im, ax=ax,pad=0.01)
                self.plasma_image_list.append(im)
                k+=1
        self.figure_3.suptitle(f"{self.sim_directory_name} | $t={self.effective_plasma_t_range[slider_time_idx]/l0:.2f}~t_0$",**self.qss_plt_title)
        for w in range(10): #multiple tight_layout 
            self.figure_3.tight_layout()
            self.figure_3.tight_layout()
        self.figure_3.subplots_adjust(right = 1.0,bottom=0.047)

        self.canvas_3.draw()
        # self.loading_LABEL.deleteLater()

    def onUpdateTabPlasma(self, check_id):
        boolList = [check.isChecked() for check in self.plasma_check_list]
        if sum(boolList)==4:
            for check in self.plasma_check_list:
                if not check.isChecked():
                    check.setEnabled(False)
        else:
            for check in self.plasma_check_list:
                if not check.isChecked():
                    check.setEnabled(True)

        if self.INIT_tabPlasma == None or self.is_sim_loaded == False:
            # Popup().showError("Simulation not loaded")
            return
        if self.INIT_tabPlasma:
            l0 = 2*pi
            plasma_species_exist = "eon" in [s.name for s in self.S.namelist.Species]
            if not plasma_species_exist:
                self.error_msg = QtWidgets.QMessageBox()
                self.error_msg.setIcon(QtWidgets.QMessageBox.Critical)
                self.error_msg.setWindowTitle("Error")
                self.error_msg.setText("No Plasma (eon) Species found")
                self.error_msg.exec_()
                return

            print("===== INIT PLASMA =====")
            
            if len(self.figure_3.axes) !=0:
                for ax in self.figure_3.axes: ax.remove()

            Bx_long_diag = self.S.Probe(2,"Bx")
            self.plasma_paxisX_long = Bx_long_diag.getAxis("axis1")[:,0]
            self.plasma_paxisY_long = Bx_long_diag.getAxis("axis2")[:,1]
            self.plasma_t_range = Bx_long_diag.getTimes()
            try:
                self.av_plasma_t_range = self.S.ParticleBinning("2D_weight_av").getTimes()
            except:
                self.av_plasma_t_range = self.S.ParticleBinning("weight_av").getTimes()

            Bx_trans_diag = self.S.Probe("trans","Bx")
            self.plasma_paxisY = Bx_trans_diag.getAxis("axis2")[:,1]
            self.plasma_paxisZ = Bx_trans_diag.getAxis("axis3")[:,2]
            self.plasma_paxisX_Bx = Bx_trans_diag.getAxis("axis1")[:,0]

            self.plasma_extentXY_long = [self.plasma_paxisX_long[0]/l0,self.plasma_paxisX_long[-1]/l0,
                                  self.plasma_paxisY_long[0]/l0-self.Ltrans/l0/2,self.plasma_paxisY_long[-1]/l0-self.Ltrans/l0/2]
            self.plasma_extentYZ = [self.plasma_paxisY[0]/l0-self.Ltrans/l0/2,self.plasma_paxisY[-1]/l0-self.Ltrans/l0/2,
                             self.plasma_paxisZ[0]/l0-self.Ltrans/l0/2,self.plasma_paxisZ[-1]/l0-self.Ltrans/l0/2]


            self.plasma_time_SLIDER.setMaximum(len(self.plasma_t_range)-1)
            self.plasma_xcut_SLIDER.setMaximum(len(self.plasma_paxisX_Bx)-1)
            self.plasma_time_SLIDER.setValue(len(self.plasma_t_range)-1)
            self.plasma_xcut_SLIDER.setValue(len(self.plasma_paxisX_Bx)-3)
            self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[-1]/l0,2)))
            self.plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[-3]/l0,2)))

            self.plasma_image_list = []
            self.plasma_data_list = []

            self.INIT_tabPlasma = False
            app.processEvents()
            self.updateInfoLabel()

        l0 = 2*pi
        if check_id < 50: #CHECK_BOX UPDATE
            #=====================================
            # REMOVE ALL FIGURES --> NOT OPTIMAL
            #=====================================
            if len(self.figure_3.axes) !=0:
                for ax in self.figure_3.axes: ax.remove()

            selected_plasma_names = np.array(self.plasma_names)[boolList]

            self.plasma_image_list = []
            self.plasma_data_list = []

            self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, selected_plasma_names)
            self.loadthread.finished.connect(self.onUpdateTabPlasmaFigure)
            self.loadthread.start()

        elif check_id <= 210: #SLIDER UPDATE
            boolList = [check.isChecked() for check in self.plasma_check_list]
            selected_plasma_names = np.array(self.plasma_names)[boolList]
            contains_av = sum(["_av" in name for name in selected_plasma_names]) != 0
            slider_time_idx = self.plasma_time_SLIDER.sliderPosition()

            if check_id == 101: #QLineEdit time
                time_edit_value = float(self.plasma_time_EDIT.text())
                time_idx = np.where(abs(self.effective_plasma_t_range/l0-time_edit_value)==np.min(abs(self.effective_plasma_t_range/l0-time_edit_value)))[0][0]
                self.plasma_time_SLIDER.setValue(time_idx)
                self.plasma_time_EDIT.setText(str(round(self.effective_plasma_t_range[time_idx]/l0,2)))
            else:
                time_idx = slider_time_idx
                self.plasma_time_EDIT.setText(str(round(self.effective_plasma_t_range[time_idx]/l0,2)))

            if check_id == 201: #QLineEdit zcut
                xcut_edit_value = float(self.plasma_xcut_EDIT.text())
                xcut_idx = np.where(abs(self.plasma_paxisX_Bx/l0-xcut_edit_value)==np.min(abs(self.plasma_paxisX_Bx/l0-xcut_edit_value)))[0][0]
                self.plasma_xcut_SLIDER.setValue(xcut_idx)
                self.plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[xcut_idx]/self.l0,2)))
            else:
                xcut_idx = self.plasma_xcut_SLIDER.sliderPosition()
                self.plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[xcut_idx]/l0,2)))


            for i,im in enumerate(self.plasma_image_list):
                if "_av" not in selected_plasma_names[i] and contains_av:
                    current_time = self.av_plasma_t_range[time_idx]
                    new_time_idx = np.where(np.abs(current_time - self.plasma_t_range) == np.min(np.abs(current_time - self.plasma_t_range)))[0][0]
                    time_idx = new_time_idx
                else: time_idx = self.plasma_time_SLIDER.sliderPosition()

                if "_trans" in selected_plasma_names[i]:
                    im.set_data(self.plasma_data_list[i][time_idx,xcut_idx,:,:])
                    self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($x={self.plasma_paxisX_Bx[xcut_idx]/l0:.1f}~\lambda$)")
                else:
                    im.set_data(self.plasma_data_list[i][time_idx].T)

            self.figure_3.suptitle(f"{self.sim_directory_name} | $t={self.effective_plasma_t_range[slider_time_idx]/l0:.2f}~t_0$",**self.qss_plt_title)
            self.canvas_3.draw()

        elif check_id == 1000 :
            if self.loop_in_process: return
            self.loop_in_process = True
            boolList = [check.isChecked() for check in self.plasma_check_list]
            selected_plasma_names = np.array(self.plasma_names)[boolList]
            contains_av = sum(["_av" in name for name in selected_plasma_names]) != 0
            xcut_idx = self.plasma_xcut_SLIDER.sliderPosition()
            for slider_time_idx in range(len(self.effective_plasma_t_range)):
                self.plasma_time_SLIDER.setValue(slider_time_idx)
                self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[slider_time_idx]/self.l0,2)))


                for i,im in enumerate(self.plasma_image_list):
                    if "_av" not in selected_plasma_names[i] and contains_av:
                        current_time = self.av_plasma_t_range[time_idx]
                        new_time_idx = np.where(np.abs(current_time - self.plasma_t_range) == np.min(np.abs(current_time - self.plasma_t_range)))[0][0]
                        time_idx = new_time_idx
                    else: time_idx = slider_time_idx

                    if "_trans" in selected_plasma_names[i]:
                        im.set_data(self.plasma_data_list[i][time_idx,xcut_idx,:,:].T)
                        self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($x={self.plasma_paxisX_Bx[xcut_idx]/l0:.1f}~\lambda$)")
                    else:
                        im.set_data(self.plasma_data_list[i][time_idx,:,:].T)

                self.figure_3.suptitle(f"{self.sim_directory_name} | t={self.effective_plasma_t_range[slider_time_idx]/self.l0:.2f}$~t_0$",**self.qss_plt_title)
                self.canvas_3.draw()
                time.sleep(0.01)
                app.processEvents()

            self.loop_in_process = False


    def call_compa_ThreadGetPlasmaProbeData(self, check_id):
        if not self.is_sim_loaded:
            # Popup().showError("Simulation not loaded")
            return
        boolList = [check.isChecked() for check in self.compa_plasma_check_list]

        self.figure_4_plasma.clf()
        self.ax4_plasma1 = self.figure_4_plasma.add_subplot(1,2,1)
        self.ax4_plasma1.set_title(self.sim_directory_name,**self.qss_plt_title)
        self.ax4_plasma2 = self.figure_4_plasma.add_subplot(1,2,2)
        self.ax4_plasma2.set_title(self.compa_sim_directory_name,**self.qss_plt_title)
        selected_compa_plasma_names = np.array(self.plasma_names)[boolList]

        self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, selected_compa_plasma_names)
        self.loadthread.finished.connect(partial(self.onUpdateTabCompaPlasmaFigure, is_compa=False))
        if self.is_compa_sim_loaded: self.loadthread.finished.connect(partial(self.call_compa2_ThreadGetPlasmaProbeData, check_id))
        self.loadthread.start()
        return

    def call_compa2_ThreadGetPlasmaProbeData(self, check_id):
        boolList = [check.isChecked() for check in self.compa_plasma_check_list]
        selected_compa_plasma_names = np.array(self.plasma_names)[boolList]
        self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.compa_S, selected_compa_plasma_names)
        self.loadthread.finished.connect(partial(self.onUpdateTabCompaPlasmaFigure, is_compa=True))
        self.loadthread.start()

    def onUpdateTabCompaPlasmaFigure(self, plasma_data_list_used_selected_plasma_names, is_compa=False):
        compa_plasma_data_list, used_selected_plasma_names = plasma_data_list_used_selected_plasma_names
        boolList = [check.isChecked() for check in self.compa_plasma_check_list]
        selected_plasma_names = np.array(self.plasma_names)[boolList]

        print(selected_plasma_names,used_selected_plasma_names)
        if not np.array_equal(selected_plasma_names, used_selected_plasma_names):
            print("compa plasma get data list DISCARDED")
            return

        self.compa_plasma_data = compa_plasma_data_list[0]
        self.selected_plasma_name = selected_plasma_names[0] #Only 1 requested element for compa

        ne = self.S.namelist.ne
        VMAX_Bx = 0.001*self.toTesla*self.a0*ne/0.01 #1 = 10709T
        vmax_ptheta = 0.005

        
        if "_av" in self.selected_plasma_name: #if contains averaged quantities, change slider range
            self.effective_compa_plasma_t_range = self.av_plasma_t_range
        else:
            self.effective_compa_plasma_t_range = self.plasma_t_range
        self.compa_plasma_time_SLIDER.setMaximum(len(self.effective_compa_plasma_t_range)-1)
        self.compa_plasma_time_SLIDER.setValue(len(self.effective_compa_plasma_t_range)-1)
        self.compa_plasma_time_EDIT.setText(str(round(self.effective_compa_plasma_t_range[-1]/self.l0,2)))
        time_idx = self.compa_plasma_time_SLIDER.sliderPosition()
        x_idx = self.compa_plasma_xcut_SLIDER.sliderPosition()
        
        if "Bx" in self.selected_plasma_name:
            cmap = "RdYlBu"
            vmin, vmax = -VMAX_Bx, VMAX_Bx
        elif "pθ" in self.selected_plasma_name:
            cmap = "RdYlBu"
            vmin, vmax = -vmax_ptheta,vmax_ptheta
        elif "ne" in self.selected_plasma_name or "ni" in self.selected_plasma_name:
            cmap = "jet"
            vmin, vmax = 0,3
        elif "Ekin" in self.selected_plasma_name:
            cmap = "smilei"
            vmin, vmax = 0, 0.1
        elif "jx" in self.selected_plasma_name or "px" in self.selected_plasma_name:
            cmap = "RdYlBu"
            vmin, vmax = -0.01, 0.01
        else:
            cmap = "RdYlBu"
            vmin = -0.075*np.max(np.abs(self.compa_plasma_data[time_idx]))
            vmax =  0.075*np.max(np.abs(self.compa_plasma_data[time_idx]))

        print("== DRAW IMSHOW FOR COMPA == ")

                
        if "trans" in self.selected_plasma_name:
            extent = self.plasma_extentYZ
            data = self.compa_plasma_data[time_idx,x_idx,:,:]
        else:
            extent = self.plasma_extentXY_long
            data = self.compa_plasma_data[time_idx].T

        if "trans" in self.selected_plasma_name:
            extent = self.plasma_extentYZ
            if is_compa: extent = self.compa_plasma_extentYZ
            data = self.compa_plasma_data[time_idx,x_idx,:,:]
        else:
            extent = self.plasma_extentXY_long
            if is_compa: extent = self.compa_plasma_extentXY_long
            data = self.compa_plasma_data[time_idx].T


        if is_compa:
            self.compa_plasma_data2 = self.compa_plasma_data
            # self.ax4_plasma2.cla()
            self.compa_plasma_im2 = self.ax4_plasma2.imshow(data,cmap=cmap, vmin=vmin, vmax=vmax, extent = extent, aspect="auto")
            self.cbar_4_plasma2 = self.figure_4_plasma.colorbar(self.compa_plasma_im2, ax=self.ax4_plasma2, pad=0.01)
            # self.cbar_4_plasma2.update_normal(self.compa_plasma_im2)
        else:
            self.compa_plasma_data1 = self.compa_plasma_data
            # self.ax4_plasma1.cla()
            self.compa_plasma_im1 = self.ax4_plasma1.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent = extent, aspect="auto")
            self.cbar_4_plasma1 = self.figure_4_plasma.colorbar(self.compa_plasma_im1, ax=self.ax4_plasma1, pad=0.01)
            # self.cbar_4_plasma1.update_normal(self.compa_plasma_im1)


        self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.effective_compa_plasma_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
        self.figure_4_plasma.tight_layout()
        self.canvas_4_plasma.draw()
        return

    def onUpdateTabCompaPlasma(self, check_id):
        if self.INIT_tabPlasma == None or self.is_sim_loaded == False or self.is_compa_sim_loaded==False:
            # Popup().showError("Simulation not loaded")
            return
        if check_id < 50:
            l0 = 2*pi
            Bx_long_diag = self.S.Probe("long","Bx")
            self.plasma_paxisX_long = Bx_long_diag.getAxis("axis1")[:,0]
            self.plasma_paxisY_long = Bx_long_diag.getAxis("axis2")[:,1]
            self.plasma_t_range = Bx_long_diag.getTimes()

            Bx_trans_diag = self.S.Probe("trans","Bx")
            self.plasma_paxisY = Bx_trans_diag.getAxis("axis2")[:,1]
            self.plasma_paxisZ = Bx_trans_diag.getAxis("axis3")[:,2]
            self.plasma_paxisX_Bx = Bx_trans_diag.getAxis("axis1")[:,0]

            self.plasma_extentXY_long = [self.plasma_paxisX_long[0]/l0,self.plasma_paxisX_long[-1]/l0,
                                  self.plasma_paxisY_long[0]/l0-self.Ltrans/l0/2,self.plasma_paxisY_long[-1]/l0-self.Ltrans/l0/2]
            self.plasma_extentYZ = [self.plasma_paxisY[0]/l0-self.Ltrans/l0/2,self.plasma_paxisY[-1]/l0-self.Ltrans/l0/2,
                             self.plasma_paxisZ[0]/l0-self.Ltrans/l0/2,self.plasma_paxisZ[-1]/l0-self.Ltrans/l0/2]

            compa_Bx_long_diag = self.compa_S.Probe(2,"Bx")
            self.compa_plasma_paxisX_long = compa_Bx_long_diag.getAxis("axis1")[:,0]
            self.compa_plasma_paxisY_long = compa_Bx_long_diag.getAxis("axis2")[:,1]
            self.compa_plasma_t_range = compa_Bx_long_diag.getTimes()

            compa_Bx_trans_diag = self.compa_S.Probe(1,"Bx")
            self.compa_plasma_paxisY = compa_Bx_trans_diag.getAxis("axis2")[:,1]
            self.compa_plasma_paxisZ = compa_Bx_trans_diag.getAxis("axis3")[:,2]
            self.compa_plasma_paxisX_Bx = compa_Bx_trans_diag.getAxis("axis1")[:,0]

            compa_Ltrans = self.compa_S.namelist.Ltrans

            self.compa_plasma_extentXY_long = [self.compa_plasma_paxisX_long[0]/l0,self.compa_plasma_paxisX_long[-1]/l0,
                                  self.compa_plasma_paxisY_long[0]/l0-compa_Ltrans/l0/2,self.compa_plasma_paxisY_long[-1]/l0-compa_Ltrans/l0/2]
            self.compa_plasma_extentYZ = [self.compa_plasma_paxisY[0]/l0-compa_Ltrans/l0/2,self.compa_plasma_paxisY[-1]/l0-compa_Ltrans/l0/2,
                             self.compa_plasma_paxisZ[0]/l0-compa_Ltrans/l0/2,self.compa_plasma_paxisZ[-1]/l0-compa_Ltrans/l0/2]

            self.av_plasma_t_range = self.S.ParticleBinning("2D_weight_av").getTimes()
            self.compa_av_plasma_t_range = self.compa_S.ParticleBinning("2D_weight_av").getTimes()

            if self.compa_plasma_t_range[-1] < self.plasma_t_range[-1]:
                self.effective_compa_plasma_t_range = self.compa_plasma_t_range
            else:
                self.effective_compa_plasma_t_range = self.plasma_t_range
                
            self.compa_plasma_time_SLIDER.setMaximum(len(self.effective_compa_plasma_t_range)-1)
            self.compa_plasma_xcut_SLIDER.setMaximum(len(self.plasma_paxisX_Bx)-1)
            self.compa_plasma_time_SLIDER.setValue(len(self.effective_compa_plasma_t_range)-1)
            self.compa_plasma_xcut_SLIDER.setValue(len(self.plasma_paxisX_Bx)-3)
            self.compa_plasma_time_EDIT.setText(str(round(self.effective_compa_plasma_t_range[-1]/l0,2)))
            self.compa_plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[-3]/l0,2)))

            self.plasma_image_list = []
            self.plasma_data_list = []

            self.compa_plasma_data1 = None
            self.compa_plasma_data2 = None
            # self.cbar_4_plasma1 = None
            # self.cbar_4_plasma2 = None

            self.INIT_tabPlasma = False
            self.INIT_tabCompaPlasma = False
            app.processEvents()
            self.updateInfoLabel()

            print("==== INIT PLASMA VAR FOR COMPA ====")

            self.call_compa_ThreadGetPlasmaProbeData(check_id)

        elif check_id <= 210: #SLIDER UPDATE
            if check_id == 101: #QLineEdit time
                time_edit_value = float(self.compa_plasma_time_EDIT.text())
                time_idx = np.where(abs(self.plasma_t_range/self.l0-time_edit_value)==np.min(abs(self.plasma_t_range/self.l0-time_edit_value)))[0][0]
                self.compa_plasma_time_SLIDER.setValue(time_idx)
                self.compa_plasma_time_EDIT.setText(str(round(self.effective_compa_plasma_t_range[time_idx]/self.l0,2)))
            else:
                time_idx = self.compa_plasma_time_SLIDER.sliderPosition()
                self.compa_plasma_time_EDIT.setText(str(round(self.effective_compa_plasma_t_range[time_idx]/self.l0,2)))

            if check_id == 201: #QLineEdit zcut

                xcut_edit_value = float(self.compa_plasma_xcut_EDIT.text())
                xcut_idx = np.where(abs(self.plasma_paxisX_Bx/self.l0-xcut_edit_value)==np.min(abs(self.plasma_paxisX_Bx/self.l0-xcut_edit_value)))[0][0]
                self.compa_plasma_xcut_SLIDER.setValue(xcut_idx)
                self.compa_plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[xcut_idx]/self.l0,2)))
            else:
                xcut_idx = self.compa_plasma_xcut_SLIDER.sliderPosition()
                self.compa_plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[xcut_idx]/self.l0,2)))

            if "trans" in self.selected_plasma_name:
                data1 = self.compa_plasma_data1[time_idx,xcut_idx,:,:]
                data2 = self.compa_plasma_data2[time_idx,xcut_idx,:,:]
                self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.effective_compa_plasma_t_range[time_idx]/self.l0:.2f}~t_0$ ($x={self.plasma_paxisX_Bx[xcut_idx]/self.l0:.1f}~\lambda$)",**self.qss_plt_title)
            else:
                data1 = self.compa_plasma_data1[time_idx].T
                data2 = self.compa_plasma_data2[time_idx].T
                self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.effective_compa_plasma_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)

            self.compa_plasma_im1.set_data(data1)
            self.compa_plasma_im2.set_data(data2)
            self.canvas_4_plasma.draw()
            return

        elif check_id == 1000: #Play button
            if self.loop_in_process: return

            self.loop_in_process = True

            xcut_idx = self.compa_plasma_xcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.effective_compa_plasma_t_range)):
                self.compa_plasma_time_SLIDER.setValue(time_idx)
                self.compa_plasma_time_EDIT.setText(str(round(self.effective_compa_plasma_t_range[time_idx]/self.l0,2)))


                if "trans" in self.selected_plasma_name:
                    data1 = self.compa_plasma_data1[time_idx,xcut_idx,:,:]
                    data2 = self.compa_plasma_data2[time_idx,xcut_idx,:,:]
                    self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.effective_compa_plasma_t_range[time_idx]/self.l0:.2f}~t_0$ ($x={self.plasma_paxisX_Bx[xcut_idx]/self.l0:.1f}~\lambda$)",**self.qss_plt_title)
                else:
                    data1 = self.compa_plasma_data1[time_idx].T
                    data2 = self.compa_plasma_data2[time_idx].T
                    self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.effective_compa_plasma_t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
                self.compa_plasma_im1.set_data(data1)
                self.compa_plasma_im2.set_data(data2)
                self.canvas_4_plasma.draw()
                time.sleep(0.01)
                app.processEvents()

            self.loop_in_process = False
            return

    def onUpdateTabCompa(self, box_idx):
        if box_idx==0: #scalar
            self.compa_scalar_groupBox.show()
            self.canvas_4_scalar.show()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
            self.compa_intensity_groupBox.hide()
            self.canvas_4_intensity.hide()
            self.canvas_4_intensity_time.hide()
            self.compa_track_groupBox.hide()
            self.canvas_4_track.hide()
        elif box_idx==1: #plasma
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.show()
            self.canvas_4_plasma.show()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
            self.compa_intensity_groupBox.hide()
            self.canvas_4_intensity.hide()
            self.canvas_4_intensity_time.hide()
            self.compa_track_groupBox.hide()
            self.canvas_4_track.hide()
        elif box_idx==2: #binning
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.show()
            self.canvas_4_binning.show()
            self.compa_intensity_groupBox.hide()
            self.canvas_4_intensity.hide()
            self.canvas_4_intensity_time.hide()
            self.compa_track_groupBox.hide()
            self.canvas_4_track.hide()
        elif box_idx==3: #intensity
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
            self.compa_intensity_groupBox.show()
            self.canvas_4_intensity.show()
            self.compa_track_groupBox.hide()
            self.canvas_4_track.hide()
            # self.canvas_4_intensity_time.show()
        else:
            self.compa_track_groupBox.show()
            self.canvas_4_track.show()
            
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
            self.compa_intensity_groupBox.hide()
            self.canvas_4_intensity.hide()
        return


    def onUpdateTabBinning(self, id, is_compa=False):
        if not self.is_sim_loaded:
            # Popup().showError("Simulation not loaded")
            return
        if is_compa and not self.is_compa_sim_loaded:
            utils.Popup().showError("2nd simulation not loaded")
            return

        if not is_compa:
            canvas = self.canvas_5
            diag_name = self.binning_diag_name_EDIT.text()
            time_slider = self.binning_time_SLIDER
        else:
            canvas = self.canvas_4_binning
            diag_name = self.compa_binning_diag_name_EDIT.text()
            time_slider = self.compa_binning_time_SLIDER

        figure = canvas.figure

        if id == 0:
            figure.clf()

            diag_name_list = diag_name.replace(" ", "").split(",")
            if is_compa: diag_name_list = diag_name_list[0:1] #cannot use multiple diag when comparing
            print("Binning diag:",diag_name_list, "| is_compa:",is_compa)
            ax = figure.add_subplot(1,1,1) #Assume single ax figure

            binning_image_list = []
            binning_data_list = []

            for diag_name in diag_name_list:

                if diag_name == "weight_r":
                    diag = self.S.ParticleBinning("2D_weight_av")
                    x_range = np.array(diag.getAxis("x"))
                    y_range = np.array(diag.getAxis("y"))-self.Ltrans/2
                    t_range = diag.getTimes()
                    time_slider.setMaximum(len(t_range)-1)
                    time_slider.setValue(len(t_range)-1)
                    time_idx = -1
                    idx_x = round(len(x_range)*0.25)

                    binning_data = np.mean(np.array(diag.getData()),axis=-1)
                    binning_data = np.mean(binning_data[:,:-idx_x],axis=1)/self.ne

                    binning_image, = ax.plot(y_range/self.l0,binning_data[-1], label=diag_name)
                    ax.set_xlabel("$y/\lambda$")
                    ax.set_ylabel("$n_e/n_c$")
                    self.binning_t_range = t_range

                    if is_compa:
                        diag2 = self.compa_S.ParticleBinning("2D_weight_av")
                        binning_data2 = np.mean(np.array(diag2.getData()),axis=-1)
                        x_range2 = np.array(diag2.getAxis("x"))
                        idx_x2 = round(len(x_range2)*0.25)
                        y_range2 = np.array(diag2.getAxis("y"))-self.compa_S.namelist.Ltrans/2
                        binning_data2 = np.mean(binning_data2[:,:-idx_x2],axis=1)/self.ne
                        binning_image2, =  ax.plot(y_range2/self.l0,binning_data2[-1], label=diag_name+"_compa")
                        self.compa_binning_t_range = t_range
                    break


                try:
                    diag = self.S.ParticleBinning(diag_name)
                    binning_data = np.array(diag.getData())
                    binning_data_list.append(binning_data)
                    # if self.binning_log_CHECK.isChecked():
                    #     data = np.log10(binning_data)
                    if is_compa:
                        diag2 = self.compa_S.ParticleBinning(diag_name)
                        binning_data2 = np.array(diag2.getData())
                except IndexError:
                    utils.Popup().showError(f'No ParticleBinning diagnostic "{diag_name}" found')
                    return

                t_range = diag.getTimes()
                time_slider.setMaximum(len(t_range)-1)
                time_slider.setValue(len(t_range)-1)
                time_idx = -1
                self.binning_t_range = t_range
                if is_compa:
                    self.compa_binning_t_range = t_range

                if binning_data.ndim == 1: # function of time only
                    binning_image, = ax.plot(t_range/self.l0,binning_data, label=diag_name)
                    binning_image_list.append(binning_image)
                    ax.set_xlabel("t/t0")
                    if is_compa: binning_image2, = ax.plot(t_range/self.l0,binning_data2, label=diag_name+"_compa")
                    if self.binning_log_CHECK.isChecked(): ax.set_yscale("log")
                elif binning_data.ndim == 2:
                    x_range = np.array(diag.getAxis(diag_name))
                    if diag_name == "px_av": x_range = np.array(diag.getAxis("px"))
                    if diag_name == "ptheta": x_range = np.array(diag.getAxis("user_function0"))

                    if diag_name == "ekin":
                        binning_image, = ax.plot(x_range,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.set_xlabel("Ekin")
                        if is_compa: binning_image2, = ax.plot(x_range,binning_data2[time_idx], label=diag_name+"_compa")

                    elif diag_name=="Lx_x" or diag_name=="Lx_x_av" or diag_name=="Lx_r" or diag_name=="Lx_r_ion":
                        ax.set_xlabel("$x/\lambda$")
                        ax.set_ylabel("$L_x$")

                        if diag_name=="Lx_r" or diag_name=="Lx_r_ion":
                            x_range = diag.getAxis("user_function0")
                            idx = round(0.3*binning_data.shape[0]) #Average over 30 - 70% of the range to remove transiant effects
                            ax.set_xlabel("$r/\lambda$")
                            m = np.nanmean(binning_data[idx:-idx],axis=0)
                            std = np.nanstd(binning_data[idx:-idx],axis=0)
                            if not is_compa:
                                ax.fill_between(x_range/self.l0, m-std, m+std, color="gray",alpha=0.25)
                                ax.plot(x_range/self.l0, m, "k--", label=diag_name+" time average")
                                
                            if is_compa:
                                x_range2 = np.array(diag2.getAxis("user_function0"))
                                m2 = np.nanmean(binning_data2[idx:-idx],axis=0)
                                ax.plot(x_range/self.l0, m, label=diag_name+" time average")
                                ax.plot(x_range2/self.l0, m2, label=diag_name+"_compa time average")
                                binning_image = None #No time slider as we used time average
                                binning_image2 = None
                                break
                        else: # =Lx_x
                            x_range = np.array(diag.getAxis("x"))
                        binning_image, = ax.plot(x_range/self.l0,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        if is_compa: 
                            if diag_name!="Lx_r": x_range2 = np.array(diag2.getAxis("x"))
                            binning_image2, = ax.plot(x_range2/self.l0,binning_data2[time_idx], label=diag_name+"_compa")

                    else:
                        # x_range = np.array(diag.getAxis(diag_name))
                        binning_image, = ax.plot(x_range,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        if is_compa: binning_image2, = ax.plot(x_range,binning_data2[time_idx], label=diag_name+"_compa")
                        ax.set_xlabel(diag_name)
                        ax.set_ylabel("weight")
                    if self.binning_log_CHECK.isChecked(): ax.set_yscale("log")
                elif binning_data.ndim == 3:
                    if self.binning_log_CHECK.isChecked():
                        data = np.log10(binning_data)
                        if is_compa: data2 = np.log10(binning_data2)
                    else:
                        data = binning_data
                        if is_compa: data2 = binning_data2
                    if is_compa:
                        figure.clf() #dont use single ax figure
                        ax = figure.add_subplot(1,2,1)
                        ax2 = figure.add_subplot(1,2,2)
                        ax.set_title(self.sim_directory_name,**self.qss_plt_title)
                        ax2.set_title(self.compa_sim_directory_name,**self.qss_plt_title)

                    if diag_name =="phase_space":
                        x_range  = diag.getAxis("x")
                        px_range = diag.getAxis("px")
                        extent = [x_range[0]/self.l0,x_range[-1]/self.l0,px_range[0],px_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: 
                            x_range2 = diag2.getAxis("x")
                            extent2 = [x_range2[0]/self.l0,x_range2[-1]/self.l0,px_range[0],px_range[-1]]
                            binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent2, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("$x/\lambda$")
                        ax.set_ylabel("px")
                    elif diag_name =="phase_space_p":
                        vy_range  = diag.getAxis("py")
                        vz_range = diag.getAxis("pz")
                        extent = [vy_range[0],vy_range[-1],vz_range[0],vz_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("vy")
                        ax.set_ylabel("vz")
                    elif diag_name =="phase_space_Lx" or diag_name =="phase_space_Lx_raw":
                        x_range  = diag.getAxis("x")
                        px_range = diag.getAxis("user_function0")
                        extent = [x_range[0]/self.l0,x_range[-1]/self.l0,px_range[0],px_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: 
                            x_range2  = diag2.getAxis("x")
                            extent2 = [x_range2[0]/self.l0,x_range2[-1]/self.l0,px_range[0],px_range[-1]]
                            binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent2, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("$x/\lambda$")
                        ax.set_ylabel("Lx")
                    elif diag_name =="phase_space_Lx_r" or diag_name =="phase_space_Lx_r_zoom":
                        r_range  = diag.getAxis("user_function0")
                        Lx_range = diag.getAxis("user_function1")
                        extent = [r_range[0]/self.l0,r_range[-1]/self.l0,Lx_range[0],Lx_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: 
                            r_range2  = diag2.getAxis("user_function0")
                            Lx_range2 = diag2.getAxis("user_function1")
                            extent2 = [r_range2[0]/self.l0,r_range2[-1]/self.l0,Lx_range2[0],Lx_range2[-1]]
                            binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent2, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("$r/\lambda$")
                        ax.set_ylabel("Lx")

                    self.binning_colorbar = figure.colorbar(binning_image, ax=ax, pad=0.01)
                    if is_compa: self.binning_colorbar2 = figure.colorbar(binning_image2, ax=ax2, pad=0.01)
                    break

            self.binning_data_list = binning_data_list
            self.binning_image_list = binning_image_list

            if is_compa:
                self.compa_binning_image = binning_image
                self.compa_binning_data = binning_data
                self.compa_binning_image2 = binning_image2
                self.compa_binning_data2 = binning_data2
            else:
                self.binning_image = binning_image
                self.binning_data = binning_data

            for ax in figure.axes:
                if ax.get_label()!='<colorbar>':
                    ax.grid()
                    ax.legend()
                    ax.relim()            # Recompute the limits
                    ax.autoscale_view()   # Apply the new limits
            figure.suptitle(f"{self.sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
            if is_compa:
                figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name}| $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)

            figure.tight_layout()
            canvas.draw()
            return

        elif id == 100:
            if is_compa:
                binning_image = self.compa_binning_image
                binning_data = self.compa_binning_data

                binning_image2 = self.compa_binning_image2
                binning_data2 = self.compa_binning_data2

                time_idx = self.compa_binning_time_SLIDER.sliderPosition()
                t_range = self.compa_binning_t_range
                data = binning_data[time_idx]
                data2 = binning_data2[time_idx]
            else:
                binning_image = self.binning_image
                binning_data = self.binning_data
                time_idx = self.binning_time_SLIDER.sliderPosition()
                t_range = self.binning_t_range
                data = binning_data[time_idx]

            if binning_image is None or (is_compa and binning_image2 is None): #For Lx_r, no time update!
                return
            if binning_data.ndim == 1:
                if self.binning_log_CHECK.isChecked():
                    binning_image.axes.set_yscale("log")
                    if is_compa: binning_image2.axes.set_yscale("log")
                    if is_compa: figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
                    else:figure.suptitle(f"{self.sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f}~t_0$",**self.qss_plt_title)
                    canvas.draw()
                else:
                    binning_image.axes.set_yscale("linear")
                    if is_compa: binning_image2.axes.set_yscale("linear")
                    if is_compa: figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} | t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
                    else:figure.suptitle(f"{self.sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
                    canvas.draw()
                return

            elif binning_data.ndim == 2:
                binning_image.set_ydata(data)
                if is_compa: binning_image2.set_ydata(data2)

                if self.binning_log_CHECK.isChecked():
                    binning_image.axes.set_yscale("log")
                    if is_compa: binning_image2.axes.set_yscale("log")
                else:
                    binning_image.axes.set_yscale("linear")
                    if is_compa: binning_image2.axes.set_yscale("linear")

            elif binning_data.ndim == 3:
                if self.binning_log_CHECK.isChecked():
                    binning_image.set_data(np.log10(data).T)
                    if is_compa: binning_image2.set_data(np.log10(data2).T)
                else:
                    binning_image.set_data(data.T)
                    if is_compa: binning_image2.set_data(data2.T)

            for ax in figure.axes:
                if ax.get_label()!='<colorbar>':
                    ax.relim()            # Recompute the limits
                    ax.autoscale_view()   # Apply the new limits
        else:
            raise Exception('id for Binning invalid')

        if is_compa: figure.suptitle(f"{self.sim_directory_name} VS {self.compa_sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
        else:figure.suptitle(f"{self.sim_directory_name} | $t = {t_range[time_idx]/self.l0:.2f} ~t_0$",**self.qss_plt_title)
        canvas.draw()
        return

    def onCloseProgressBar(self, sim_id_int):
        print("REMOVE PROGRESS BAR LAYOUT", sim_id_int)
        print("finished sim:",self.finished_sim_hist)
        self.finished_sim_hist.remove(sim_id_int)
        layout_to_del = self.layout_progress_bar_dict[str(sim_id_int)]
        for i in range(self.layoutTornado.count()):
            layout_progressBar = self.layoutTornado.itemAt(i)
            if layout_progressBar == layout_to_del:
                self.deleteLayout(self.layoutTornado, i)

    def async_onUpdateTabTornado(self, download_trnd_json = True):
        """
        asynchronous function called PERIODICALLY
        """
        sim_json_name = "simulations_info.json"
        self.tornado_refresh_BUTTON.setStyleSheet("") 
        with open(sim_json_name) as f:
            self.sim_dict = json.load(f)

        OLD_NB_SIM_RUNNING = len(self.running_sim_hist)
        CURRENT_NB_SIM_RUNNING = len(self.sim_dict) - 1 #-1 for datetime

        print("TORNADO sim running: ",list(self.sim_dict)[:-1])
        #================================
        # CHECK FOR FINISHED SIMULATIONS
        #================================
        print("previous sim dict:",list(self.previous_sim_dict)[:-1],"\n")
        if (CURRENT_NB_SIM_RUNNING <= OLD_NB_SIM_RUNNING) and (list(self.sim_dict) != list(self.running_sim_hist)): #AT LEAST ONE SIMULATION HAS FINISHED
            running_sim_hist = np.copy(self.running_sim_hist)
            for n,old_sim_id_int in enumerate(running_sim_hist):
                print("running sim loop:",n,old_sim_id_int)
                if str(old_sim_id_int) not in list(self.sim_dict): #this simulation has finished
                    print(old_sim_id_int,"not in", list(self.sim_dict))
                    try:
                        finished_sim_path = self.previous_sim_dict[str(old_sim_id_int)]["job_full_path"]
                        finished_sim_name = self.previous_sim_dict[str(old_sim_id_int)]["job_full_name"]
                        print(finished_sim_path,"download is available ! \a") #\a
                        utils.Popup().showToast('Tornado download is available', finished_sim_name)
                    except KeyError:
                        print("/!\ KeyError",old_sim_id_int)
                    self.finished_sim_hist.append(old_sim_id_int)
                    self.running_sim_hist.remove(old_sim_id_int)
                    self.can_download_sim_dict[int(old_sim_id_int)] = finished_sim_path

                    layout = self.layout_progress_bar_dict[str(old_sim_id_int)]
                    progress_bar = layout.itemAt(2).widget()
                    ETA_LABEL = layout.itemAt(3).widget()
                    dl_sim_BUTTON = layout.itemAt(6).widget()

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
                    close_BUTTON.clicked.connect(partial(self.onCloseProgressBar,old_sim_id_int))
                    layout.addWidget(close_BUTTON)
                else:
                    finished_sim_path = self.sim_dict[str(old_sim_id_int)]["job_full_path"]
                    self.can_download_sim_dict[int(old_sim_id_int)] = finished_sim_path #Still allow to be downloaded !
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
            sim_expected_time = sim["expected_time"].rjust(5)

            sim_name = sim["job_full_name"][:-3]
            sim_nodes = int(sim["NODES"])
            sim_push_time = sim["push_time"]
            diag_id = sim["diag_id"]
            sim_params = sim["sim_params"]

            if (sim_id_int not in self.running_sim_hist) and (sim_id_int not in self.finished_sim_hist):

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA,sim_expected_time, sim_push_time, diag_id,sim_params)
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
                ETA_label.setText(f"<b>{sim_ETA}</b><br>{sim_expected_time}")
                push_time_label = self.layout_progress_bar_dict[sim_id].itemAt(4).widget()
                push_time_str_SI = str(sim_push_time)+"ns"
                if sim_push_time >= 10_000:
                    push_time_str_SI = f"{self.printSI(sim_push_time*10**-9,'s',ndeci=2):}"
                    push_time_label.setStyleSheet("color: red")
                push_time_label.setText(push_time_str_SI)

        #Update label with Update datetime
        sim_datetime = self.sim_dict["datetime"]
        self.tornado_last_update_LABEL.setText(f"Last updated: {sim_datetime}")

        self.previous_sim_dict = self.sim_dict
        app.processEvents()
        return

    def call_ThreadDownloadSimJSON(self):
        
        # return
        # print(self.tornado_refresh_BUTTON.styleSheet())
        self.tornado_refresh_BUTTON.setStyleSheet("background-color : #D22B2B	") 
        
        self.loadthread = class_threading.ThreadDownloadSimJSON("/sps3/jeremy/LULI/simulations_info.json", os.environ["SMILEI_QT"])
        self.loadthread.finished.connect(self.async_onUpdateTabTornado)
        self.loadthread.start()
        return

    def call_ThreadDownloadSimData(self,sim_id):
        job_full_path = self.can_download_sim_dict[int(sim_id)]
        print("===========================")
        print("downloading request for", sim_id,job_full_path)

        host = "llrlsi-gw.in2p3.fr"
        user = "jeremy"
        with open(f"{os.environ['SMILEI_QT']}\\..\\..\\tornado_pwdfile.txt",'r') as f: pwd_crypt = f.read()
        pwd = utils.encrypt(pwd_crypt,-2041000*2-1)
        remote_path = "/sps3/jeremy/LULI/"
        ssh_key_filepath = r"C:\Users\Jeremy\.ssh\id_rsa.pub"
        remote_path = "/sps3/jeremy/LULI/"
        remote_client = paramiko_SSH_SCP_class.RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
        res = remote_client.execute_commands([f"du {job_full_path} -b"]) #gives size in bytes (1kb = 1024 bytes)
        total_size_du_b = int(res[0].split()[0])#gives size in bytes (1kb = 1024 bytes)

        self.download_prc_hist_dict[sim_id] = [] #create prc history list for sim_id
        tornado_download_TIMER = QtCore.QTimer()
        refresh_time_s = 2 #s
        tornado_download_TIMER.setInterval(int(refresh_time_s*1000)) #in ms
        tornado_download_TIMER.timeout.connect(partial(self.onUpdateDownloadBar,total_size_du_b, job_full_path, sim_id))
        tornado_download_TIMER.start()

        self.loadthread = class_threading.ThreadDownloadSimData(job_full_path)
        self.loadthread.finished.connect(partial(self.onDownloadSimDataFinished,sim_id, tornado_download_TIMER))
        self.loadthread.start()

        layout = self.layout_progress_bar_dict[str(sim_id)]
        # progress_bar = layout.itemAt(2).widget()
        ETA_label = layout.itemAt(3).widget()
        dl_sim_BUTTON = layout.itemAt(6).widget()
        dl_sim_BUTTON.setStyleSheet("border-color: orange")
        dl_sim_BUTTON.setEnabled(False)
        close_sim_BUTTON = layout.itemAt(7).widget()
        close_sim_BUTTON.setEnabled(False)
        ETA_label.setText("DL")
        return
    def onUpdateDownloadBar(self, total_size_du_b, job_full_path, sim_id):
        general_folder_name = job_full_path[27:]
        local_folder = os.environ["SMILEI_CLUSTER"]
        local_sim_path = f"{local_folder}\\{general_folder_name}"

        size = sum([os.path.getsize(f"{local_sim_path}\{f}") for f in os.listdir(local_sim_path)])
        
        print(size/1048576/1024,"GB /",total_size_du_b/1048576/1024,"GB")
        prc_exact = size/(total_size_du_b)*100
        prc = round(prc_exact)
        
        self.download_prc_hist_dict[sim_id].append(prc_exact) #append prc to download hist
        
        layout = self.layout_progress_bar_dict[str(sim_id)]
        progress_bar = layout.itemAt(2).widget()
        progress_bar.setValue(prc)
        
        if len(self.download_prc_hist_dict[sim_id]) > 4:
            prc_diff_4_period = (self.download_prc_hist_dict[sim_id][-1] - self.download_prc_hist_dict[sim_id][-5])
            download_speed_prc_per_s = prc_diff_4_period/(4*2) #speed in %/s (refresh_time_s = 2s in the above function)
            
            total_size_MB = total_size_du_b/1048576 #total size in MB
            download_speed_MB_per_s = download_speed_prc_per_s/100*total_size_MB
            print(f"tornado download {sim_id}: {prc_exact:.1f} % ({download_speed_MB_per_s:.0f} MB/s)")
            
            ETA_label = layout.itemAt(3).widget()
            ETA_label.setText(f"{download_speed_MB_per_s:.0f} MB/s")
        else:
            print(f"tornado download {sim_id}: {prc_exact:.1f} %")

        self.updateInfoLabel()
        return

    def onDownloadSimDataFinished(self,sim_id, tornado_download_TIMER):
        tornado_download_TIMER.stop()
        layout = self.layout_progress_bar_dict[str(sim_id)]
        progress_bar = layout.itemAt(2).widget()
        ETA_label = layout.itemAt(3).widget()
        dl_sim_BUTTON = layout.itemAt(6).widget()
        close_sim_BUTTON = layout.itemAt(7).widget()
        dl_sim_BUTTON.setStyleSheet("border-color: green")
        dl_sim_BUTTON.setEnabled(False)
        close_sim_BUTTON.setEnabled(True)
        # pixmap = QtGui.QPixmap(os.environ["SMILEI_QT"]+"\\Ressources\\green_check_icon.jpg")
        # pixmap = pixmap.scaled(ETA_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation);

        ETA_label.setText("OK") #green check

        ETA_label.setStyleSheet("background-color: #80ef80")
        progress_bar.setStyleSheet(self.qss_progressBar_DOWNLOADED)
        del self.download_prc_hist_dict[sim_id] #delete prc history once download finished
        
        print(sim_id,"download is finished ! \a") #\a
        utils.Popup().showToast('Tornado download is finished', sim_id,ToastPreset.INFORMATION)

        return


    def onInitTabTornado(self):
        if self.INIT_tabTornado == None: return
        if self.INIT_tabTornado:

            self.tornado_update_TIMER = QtCore.QTimer()
            refresh_time_min = 20 #minute
            self.tornado_update_TIMER.setInterval(int(refresh_time_min*60*1000)) #in ms
            # self.tornado_update_TIMER.timeout.connect(self.async_onUpdateTabTornado)
            self.tornado_update_TIMER.timeout.connect(self.call_ThreadDownloadSimJSON)
            self.tornado_update_TIMER.start()

            sim_json_name = "simulations_info.json"
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
            self.download_prc_hist_dict = {}

            for sim_id in self.sim_dict:
                if sim_id == "datetime": continue #only open sim data and not metadata (located at the end of dict)
                sim = self.sim_dict[sim_id]
                sim_progress = sim["progress"]*100
                sim_ETA = sim["ETA"].rjust(5)
                sim_expected_time = sim["expected_time"].rjust(5)

                sim_name = sim["job_full_name"][:-3]
                sim_nodes = int(sim["NODES"])
                sim_push_time = sim["push_time"]
                diag_id = sim["diag_id"]
                sim_params = sim["sim_params"]

                self.running_sim_hist.append(int(sim_id))

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA, sim_expected_time, sim_push_time, diag_id,sim_params)

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

    def createLayoutProgressBar(self, sim_id, sim_progress, sim_name, sim_nodes, sim_ETA, sim_expected_time, sim_push_time, diag_id,sim_params):
        layoutProgressBar = QtWidgets.QHBoxLayout()

        tornado_PROGRESS_BAR = QtWidgets.QProgressBar(maximum=100)
        tornado_PROGRESS_BAR.setValue(round(sim_progress))
        tornado_PROGRESS_BAR.setFont(QFont('Arial', 15))
        tornado_PROGRESS_BAR.setAlignment(QtCore.Qt.AlignCenter)

        custom_bold_FONT = QtGui.QFont("Courier New", 14,QFont.Bold)
        custom_FONT = QtGui.QFont("Courier New", 14)
        custom_small_FONT = QtGui.QFont("Courier New", 12)

        sim_name_LABEL = QtWidgets.QLabel(f"[{sim_id}] {sim_name}")
        sim_name_LABEL.setFont(custom_bold_FONT)
        sim_name_LABEL.setMinimumWidth(420) #450 FOR LAPTOP
        sim_name_LABEL.setStyleSheet("background-color: lightblue")
        sim_name_LABEL.setWordWrap(True)
        a0, Tp, w0, dx, description = sim_params
        
        sim_name_prop_dict = self.get_sim_name_properties(sim_name)
        a0_name, Tp_name = sim_name_prop_dict["a"], sim_name_prop_dict["Tp"], 
        
        if sim_name_prop_dict["dx"] is not None: #if dx is not in the name of the sim
            dx_name = self.l0/sim_name_prop_dict["dx"]
            dx_neg_cond = (dx_name != dx)
        else:
            dx_neg_cond = False
            
        if sim_name_prop_dict["Tp"] is not None: #if dx is not in the name of the sim
            Tp_name = sim_name_prop_dict["Tp"]
            Tp_neg_cond = (round(Tp_name*self.l0,2) != round(Tp,2))
        else:
            Tp_neg_cond = False
        
        if (a0_name != a0) or Tp_neg_cond or dx_neg_cond:
            sim_name_LABEL.setStyleSheet("background-color: red")

        
        l0=2*pi
        sim_name_LABEL.setToolTip(f"a0={a0}; Tp={Tp/l0:.0f}; w0={w0/l0:.1f}; dx={l0/dx:.0f}\n{description}")
        # sim_name_LABEL.setAlignment(QtCore.Qt.AlignCenter)

        sim_node_LABEL = QtWidgets.QLabel(f"NDS:{sim_nodes}")
        sim_node_LABEL.setFont(custom_bold_FONT)
        sim_node_LABEL.setStyleSheet("background-color: lightblue")

        ETA_LABEL = QtWidgets.QLabel(f"<b>{sim_ETA}</b><br>{sim_expected_time}")
        ETA_LABEL.setFont(custom_FONT)
        ETA_LABEL.setStyleSheet("background-color: lightblue")
        # ETA_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        ETA_LABEL.setMinimumWidth(75)

        push_time_str_SI = str(sim_push_time)+"ns"
        push_time_LABEL = QtWidgets.QLabel(push_time_str_SI)
        if sim_push_time >= 10_000:
            push_time_str_SI = f"{self.printSI(sim_push_time*10**-9,'s',ndeci=2):}"
            push_time_LABEL.setStyleSheet("color: red")
        push_time_LABEL.setText(push_time_str_SI)
        push_time_LABEL.setFont(custom_small_FONT)
        push_time_LABEL.setMinimumWidth(75)

        diag_id_LABEL = QtWidgets.QLabel("D"+str(diag_id))
        diag_id_LABEL.setFont(custom_small_FONT)
        diag_id_LABEL.setMinimumWidth(65)

        dl_sim_BUTTON = QtWidgets.QPushButton()
        dl_sim_BUTTON.setIcon(QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\download_button.png"))
        dl_sim_BUTTON.setFixedSize(35,35)
        dl_sim_BUTTON.setIconSize(QtCore.QSize(25, 25))

        dl_sim_BUTTON.clicked.connect(partial(self.call_ThreadDownloadSimData, sim_id))

        layoutProgressBar.addWidget(sim_name_LABEL)
        layoutProgressBar.addWidget(sim_node_LABEL)
        layoutProgressBar.addWidget(tornado_PROGRESS_BAR)
        layoutProgressBar.addWidget(ETA_LABEL)
        layoutProgressBar.addWidget(push_time_LABEL)
        layoutProgressBar.addWidget(diag_id_LABEL)
        layoutProgressBar.addWidget(dl_sim_BUTTON)

        layoutProgressBar.setContentsMargins(25,20,25,20) #left top right bottom
        return layoutProgressBar


    def call_ThreadUpdateVerlet(self):
              
        # print(" == call_ThreadUpdateVerlet == ")
        # print("CALL:",self.verlet_POS.shape)
        self.loadthread = class_threading.ThreadUpdateVerlet(self.verlet_POS,self.verlet_OLD_POS,self.verlet_circles_size, self.verlet_mouse_pos)
        self.loadthread.finished.connect(self.onUpdateVerlet)
        self.loadthread.start()
        return
    
    def onVerletSpawn(self):
        circle_size = np.random.randint(3,5)
        x0,y0 = np.random.uniform(0.3*self.verlet_window, 0.7*self.verlet_window), 0.8*self.verlet_window
        circle = patches.Circle((x0, y0), 
                                circle_size, fc=np.random.choice(self.verlet_circles_colors),ec="k")
        
        # print("b",self.verlet_POS.shape)
        self.verlet_POS = np.vstack([self.verlet_POS,np.array([x0,y0])])
        # print("a",self.verlet_POS.shape)
        self.verlet_OLD_POS = np.vstack([self.verlet_OLD_POS,np.array([[x0,y0]])])
        self.verlet_circles.append(circle)

        self.verlet_circles_size.append(circle_size)
        self.ax_verlet.add_patch(circle)
        
        # self.verlet_circles_size = self.verlet_circles_size[1:]
        # self.verlet_circles[0].remove()
        # self.verlet_circles = self.verlet_circles[1:]
        # self.verlet_circles_size = self.verlet_circles_size[1:]
        
            
    def onUpdateVerlet(self, POS_OLD_POS):
        # print("-- onUpdateVerlet --")
        self.verlet_POS, self.verlet_OLD_POS = POS_OLD_POS
        # print("onUpdateVerlet:",self.verlet_POS.shape)
        for i in range(len(self.verlet_POS)):
            self.verlet_circles[i].center = self.verlet_POS[i]
        self.canvas_verlet.draw()
        return
    
    def onMouseMoveVerlet(self, event):
        if event.xdata is not None and event.ydata is not None:
            self.verlet_mouse_pos = np.array([event.xdata, event.ydata])
            # print("onMouseMove:",self.verlet_mouse_pos)
            # print(f"Mouse position in data coordinates: ({event.xdata:.2f}, {event.ydata:.2f})")
    
    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.key() == 178:  # Keycode for ²
            if self.is_verlet_timer_active:
                self.verlet_update_TIMER.stop()
                self.verlet_spawn_TIMER.stop()
                
                for c in self.verlet_circles: #hide circles when verlet stopped 
                    c.set_visible(False)
                self.canvas_verlet.draw()
                
                print("Verlet Timer Deactivated")
            else:
                self.verlet_update_TIMER.start(self.verlet_update_interval)
                self.verlet_spawn_TIMER.start(self.verlet_spawn_interval)
                print("Verlet Timer Activated")
                
                for c in self.verlet_circles: #hide circles when verlet stopped 
                    c.set_visible(True)
                self.canvas_verlet.draw()

            self.is_verlet_timer_active = not self.is_verlet_timer_active  # Toggle state
            



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
        print("job_full_path:",job_full_path)
        self.loadthread = class_threading.ThreadDownloadSimData(job_full_path)
        self.loadthread.start()
        # self.downloadSimData(job_full_path) #"_NEW_PLASMA_/new_plasma_LG_optic_ne0.01_dx12/")

    def printSI(self,x,baseunit,ndeci=2):
        prefix="yzafpnµm kMGTPEZY"
        shift=decimal.Decimal('1E24')
        d=(decimal.Decimal(str(x))*shift).normalize()
        m,e=d.to_eng_string().split('E')
        return m[:4] + " " + prefix[int(e)//3] + baseunit
    
    def LxEpolar(self,r,Theta,z,w0,a0,Tint):
        expr = -(1 / (2 * (w0**4 + 4 * z**2)**5)) * exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * Tint * w0**6 * (
            -8 * r**2 * z * (4 * r**4 - 12 * w0**6 + w0**8 - 48 * w0**2 * z**2 + 
            4 * r**2 * (-4 * w0**2 + w0**4 - 4 * z**2) + 8 * w0**4 * (3 + z**2) + 16 * z**2 * (6 + z**2)) * cos(2 * Theta) - 
            4 * r**2 * (4 * r**6 + 10 * w0**8 - w0**10 + 32 * w0**4 * z**2 - 32 * z**4 + 
            4 * r**4 * (-7 * w0**2 + w0**4 - 4 * z**2) - 8 * w0**6 * (3 + z**2) - 16 * w0**2 * z**2 * (6 + z**2) + 
            r**2 * (-16 * w0**6 + w0**8 - 32 * w0**2 * z**2 + 8 * w0**4 * (7 + z**2) + 16 * z**2 * (10 + z**2))) * sin(2 * Theta))
        return expr
    def Ftheta_V2_O3(self,r,theta,z, w0, a0):
        return (2 * np.exp(-((2 * r**2 * w0**2) / (w0**4 + 4 * z**2))) * a0**2 * r * w0**6 * 
                (2 * z * np.cos(2 * theta) + (r**2 - w0**2) * np.sin(2 * theta))) / (w0**4 + 4 * z**2)**3

    def Ftheta_V2_O5(self,r,theta,z, w0, a0):
        numerator = (
            2 * a0**2 * r * w0**6 * np.exp(-2 * r**2 * w0**2 / (w0**4 + 4 * z**2)) *
            (
                2 * z * np.cos(2 * theta) * (
                    4 * r**4 - 4 * r**2 * (w0**4 + 4 * w0**2 - 4 * z**2) +
                    (w0**4 + 4 * z**2) * (w0**4 + 12 * w0**2 + 4 * z**2 + 24)
                ) +
                np.sin(2 * theta) * (
                    4 * r**6 - 4 * r**4 * (w0**4 + 7 * w0**2 - 4 * z**2) +
                    r**2 * (
                        8 * (w0**4 + 4 * w0**2 + 20) * z**2 +
                        (w0**4 + 16 * w0**2 + 56) * w0**4 + 16 * z**4
                    ) -
                    (w0**4 + 4 * z**2) * (
                        4 * (w0**2 - 2) * z**2 +
                        (w0**2 + 4) * (w0**2 + 6) * w0**2
                    )
                )
            )
        )
        denominator = (w0**4 + 4 * z**2)**5
        
        expression = numerator / denominator
        return expression
    def LxEpolar_V2_O3(self,r,theta,z,w0,a0,Tint):
        return sqrt(1+(self.f(r,z)*a0)**2+ 1/4*(self.f(r,z)*a0)**4) * Tint*r*self.Ftheta_V2_O3(r,theta,z, w0, a0)

    def LxEpolar_V2_O5(self,r,theta,z,w0,a0,Tint):
        return sqrt(1+(self.f(r,z)*a0)**2+ 1/4*(self.f(r,z)*a0)**4) * Tint*r*self.Ftheta_V2_O5(r,theta,z, w0,a0)
    
    def w(self,z):
        zR = 0.5*self.w0**2
        return self.w0*np.sqrt(1+(z/zR)**2)
    def f(self,r,z):
        return (r*sqrt(2)/self.w(z))**abs(self.l1)*np.exp(-(r/self.w(z))**2)
    def f_prime(self,r,z):
        C_lp = np.sqrt(1/math.factorial(abs(self.l1)))
        return C_lp/self.w(z)**3 * exp(-(r/self.w(z))**2) * (r/self.w(z))**(abs(self.l1)-1) * (-2*r**2+self.w(z)**2*abs(self.l1))
    def f_squared_prime(self,r,z):
        return 2*self.w0**2/(self.w(z)**2*r) * self.f(r,z)**2*(abs(self.l1)-2*(r/self.w0)**2+ 4*(z**2/self.w0**4))

    def get_sim_name_properties(self, sim_name):
        patterns = {
         "a": r"a(\d+)",
         "Tp": r"Tp(\d+)",
         "dx": r"dx(\d+)"
         }
        results = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, sim_name)
            if match:
                results[key] = int(match.group(1))
            else:
                results[key] = None
                # None if the key is not found in the string
                
        return results
        

class ProxyStyle(QtWidgets.QProxyStyle):
    """Overwrite the QSlider: left click place the cursor at cursor position"""
    def styleHint(self, hint, opt=None, widget=None, returnData=None):
        res = super().styleHint(hint, opt, widget, returnData)
        if hint == self.SH_Slider_AbsoluteSetButtons:
            res |= QtCore.Qt.LeftButton
        return res

if __name__ == '__main__':
    os.chdir(os.environ["SMILEI_QT"])
    
    myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    # QtWidgets.QApplication.setAttribute(QtCore.Qt.Auto)
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough);

    qdarktheme.enable_hi_dpi()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(ProxyStyle()) #Apply slider style

    pixmap = QtGui.QPixmap(os.environ["SMILEI_QT"]+'\\Ressources\\Smilei_GUI_logo_V3.png')
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    # time.sleep(25)

    main = MainWindow()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())