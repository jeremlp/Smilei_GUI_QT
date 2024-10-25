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

import log_dialog
import IPython_dialog
import memory_dialog
import paramiko_SSH_SCP_class
import class_threading

import subprocess
import json
from pathlib import Path
from functools import partial
# from win11toast import toast
from pyqttoast import ToastPreset
from utils import Popup, encrypt
import decimal

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
        self.DISK = psutil.disk_usage
        self.SCRIPT_VERSION ='0.11.2 - Plasma Averaged'
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
        self.actionOpenMemory = QtWidgets.QAction("Open Memory Graph",self)

        self.actionDiagScalar = QtWidgets.QAction("Scalar",self)
        self.actionDiagFields = QtWidgets.QAction("Fields",self)
        self.actionDiagTrack = QtWidgets.QAction("Track",self)
        self.actionDiagPlasma = QtWidgets.QAction("Plasma",self)
        self.actionDiagPlasma = QtWidgets.QAction("Plasma",self)
        self.actionDiagBinning = QtWidgets.QAction("Binning",self)
        self.actionDiagCompa = QtWidgets.QAction("Comparison",self)
        self.actionTornado = QtWidgets.QAction("Tornado",self)

        self.actionDiagScalar.setCheckable(True)
        self.actionDiagFields.setCheckable(True)
        self.actionDiagTrack.setCheckable(True)
        self.actionDiagPlasma.setCheckable(True)
        self.actionDiagBinning.setCheckable(True)
        self.actionDiagCompa.setCheckable(True)

        self.actionTornado.setCheckable(True)

        self.fileMenu.addAction(self.actionOpenSim)
        self.fileMenu.addAction(self.actionOpenLogs)
        self.fileMenu.addAction(self.actionOpenIPython)
        self.fileMenu.addAction(self.actionOpenMemory)
        self.menuBar.addAction(self.fileMenu.menuAction())

        self.editMenu.addAction(self.actionDiagScalar)
        self.editMenu.addAction(self.actionDiagFields)
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

        self.run_time_LABEL = QtWidgets.QLabel("")
        self.run_time_LABEL.setFont(self.medium_bold_FONT)
        layoutRunTime = self.creatPara("Run time :", self.run_time_LABEL)
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
        self.toolBar_height = 50 #45 for Tower
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
        self.scalar_groupBox.setFixedHeight(110)
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
        self.fields_groupBox.setFixedHeight(210)
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

        self.track_file_BOX = QtWidgets.QComboBox()
        self.track_file_BOX.addItem("track_eon")
        self.track_file_BOX.addItem("track_eon_full")
        self.track_file_BOX.addItem("track_eon_dense")

        layoutTabSettingsTrackFile = QtWidgets.QHBoxLayout()
        self.track_Npart_EDIT = QtWidgets.QLineEdit("10")
        self.track_Npart_EDIT.setValidator(self.int_validator)
        self.track_Npart_EDIT.setMaximumWidth(45)

        self.track_update_offset_CHECK = QtWidgets.QCheckBox("Update offsets")

        layoutNpart = self.creatPara("Npart=", self.track_Npart_EDIT,adjust_label=True)

        layoutTabSettingsTrackFile.addLayout(layoutNpart)
        layoutTabSettingsTrackFile.addWidget(self.track_file_BOX)
        layoutTabSettingsTrackFile.addWidget(self.track_update_offset_CHECK)

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

        self.track_groupBox = QtWidgets.QGroupBox("Track Particles Diagnostic")
        self.track_groupBox.setFixedHeight(150)
        self.track_groupBox.setLayout(layoutTabSettings)

        self.layoutTrack = QtWidgets.QVBoxLayout()
        self.layoutTrack.addWidget(self.track_groupBox)
        self.layoutTrack.addWidget(self.canvas_2)

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
        self.plasma_names = ["Bx","Bx_av","Bx_trans","ne","ne_av","ne_trans","Lx_av","Lx_trans","jx_av","jx_trans","pθ_av","pθ_trans", "Jx","Jx_trans","Jθ", "Jθ_trans","Rho", "Rho_trans","Ekin", "Ekin_trans"]
        self.plasma_check_list = []
        N_plasma = len(self.plasma_names)
        print("len plasma:", N_plasma)
        for i, name in enumerate(self.plasma_names):
            plasma_CHECK = QtWidgets.QCheckBox(name)
            self.plasma_check_list.append(plasma_CHECK)

            # if i%2==0 and i>0:
                # separator1 = QtWidgets.QFrame()
                # separator1.setFrameShape(QtWidgets.QFrame.VLine)
                # separator1.setLineWidth(1)
                # layoutTabSettingsCheck.addWidget(separator1)
            print(i, i%(N_plasma//2))
            layoutTabSettingsCheck.addWidget(plasma_CHECK, int(i>=len(self.plasma_names)//2),i%(N_plasma//2))

        # layoutTabSettingsCheck.addStretch(50)
        layoutTabSettingsCheck.setContentsMargins(0, 0, 0, 0)


        self.plasma_check_list[0].setChecked(True)
        # self.Ey_CHECK.setChecked(True)
        self.plasma_check_list[3].setChecked(True)

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
        self.plasma_groupBox.setFixedHeight(210)
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
        self.figure_4_scalar.tight_layout()

        #-------------- MAIN Groupbox -----------------
        self.compa_load_sim_BUTTON = QtWidgets.QPushButton('Open Comparison')
        self.compa_load_sim_BUTTON.setFixedWidth(150)
        self.compa_load_status_LABEL = QtWidgets.QLabel("")
        self.compa_load_status_LABEL.setStyleSheet("color: black")
        self.compa_load_status_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        self.compa_load_status_LABEL.setFont(self.medium_bold_FONT)
        self.compa_sim_directory_name_LABEL = QtWidgets.QLabel("")
        self.compa_sim_directory_name_LABEL.setFont(self.medium_bold_FONT)
        self.compa_sim_directory_name_LABEL.adjustSize()
        self.compa_groupBox = QtWidgets.QGroupBox("Settings")
        self.compa_groupBox.setFixedHeight(110)
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
        boxLayout_settings.addWidget(self.diag_type_BOX)
        self.compa_groupBox.setLayout(boxLayout_settings)


        #-------------- COMPA SCALAR Groupbox -----------------
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
        self.compa_scalar_groupBox.setFixedHeight(110)
        self.compa_scalar_groupBox.setLayout(layoutTabSettingsCompaScalar)

        #-------------- COMPA BINNING Groupbox -----------------
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
        self.compa_binning_groupBox.setFixedHeight(150)
        self.compa_binning_groupBox.setLayout(layoutTabSettingsCompaBinning)

        # self.layoutCompaBinning = QtWidgets.QVBoxLayout()
        # self.layoutCompaBinning.addWidget(self.compa_binning_groupBox)
        # self.layoulayoutCompaBinningtBinning.addWidget(self.canvas_4_binning)
        # self.compa_binning_Widget = QtWidgets.QWidget()
        # self.compa_binning_Widget.setLayout(self.layoutCompaBinning)


        #-------------- COMPA PLASMA Groupbox -----------------
        layoutCompaTabSettingsCheck = QtWidgets.QGridLayout()
        self.compa_plasma_check_list = []
        for i,name in enumerate(self.plasma_names):
            compa_plasma_RADIO = QtWidgets.QRadioButton(name)
            self.compa_plasma_check_list.append(compa_plasma_RADIO)

            # if i%2==0 and i>0:
            #     separator1 = QtWidgets.QFrame()
            #     separator1.setFrameShape(QtWidgets.QFrame.VLine)
            #     separator1.setLineWidth(1)
                # layoutCompaTabSettingsCheck.addWidget(separator1)
            layoutCompaTabSettingsCheck.addWidget(compa_plasma_RADIO, int(i>=N_plasma//2),i%(N_plasma//2))

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
        self.compa_plasma_groupBox.setFixedHeight(210)
        self.compa_plasma_groupBox.setLayout(layoutTabSettingsCompaPlasma)

        self.layoutCompa = QtWidgets.QVBoxLayout()
        self.layoutCompa.addWidget(self.compa_groupBox)
        self.layoutCompa.addWidget(self.compa_scalar_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_scalar)

        self.compa_plasma_groupBox.hide()
        self.canvas_4_plasma.hide()

        self.compa_binning_groupBox.hide()
        self.canvas_4_binning.hide()

        self.layoutCompa.addWidget(self.compa_binning_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_binning)

        self.layoutCompa.addWidget(self.compa_plasma_groupBox)
        self.layoutCompa.addWidget(self.canvas_4_plasma)

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
        self.binning_groupBox.setFixedHeight(150)
        self.binning_groupBox.setLayout(layoutTabSettingsBinning)

        self.layoutBinning = QtWidgets.QVBoxLayout()
        self.layoutBinning.addWidget(self.binning_groupBox)
        self.layoutBinning.addWidget(self.canvas_5)
        self.binning_Widget = QtWidgets.QWidget()
        self.binning_Widget.setLayout(self.layoutBinning)


        #---------------------------------------------------------------------
        # TAB 6 TORNADO
        #---------------------------------------------------------------------
        self.tornado_Widget = QtWidgets.QWidget()

        self.layoutTornado = QtWidgets.QVBoxLayout()
        self.tornado_groupBox = QtWidgets.QGroupBox("Infos")
        self.tornado_groupBox.setFixedHeight(75)

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
        self.SMILEI_ICON_LABEL = QtGui.QIcon(os.environ["SMILEI_QT"]+"\\Ressources\\smilei_gui_svg_v3.png") #CUSTOM GUI LOGO

        self.smilei_icon_BUTTON = QtWidgets.QPushButton(self.programm_TABS)
        self.smilei_icon_BUTTON.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.smilei_icon_BUTTON.setIcon(self.SMILEI_ICON_LABEL)
        self.smilei_icon_BUTTON.setIconSize(QtCore.QSize(int(logo_width/1.5),int(logo_width*3/8/1.5))) # 800 x 300
        # self.smilei_icon_BUTTON.setIconSize(QtCore.QSize(10,10)) # 800 x 300

        self.smilei_icon_BUTTON.setStyleSheet("QPushButton { border: none; }")

        # self.smilei_icon_BUTTON.setGeometry(int((self.programm_TABS.width()-100)/2), int((self.programm_TABS.height()-100)/2) , 100, 100)

        layoutSmileiLogo = QtWidgets.QHBoxLayout()
        layoutSmileiLogo.addWidget(self.smilei_icon_BUTTON)
        self.programm_TABS.setLayout(layoutSmileiLogo)


        # print(self.programm_TABS.geometry().bottomRight())
        # print(self.smilei_icon_BUTTON.geometry().bottomRight())
        # p = self.programm_TABS.geometry().bottomRight()/2 #- self.smilei_icon_BUTTON.geometry().bottomRight()
        # print(p)

        # self.smilei_icon_BUTTON.move(p)
        # self.smilei_icon_BUTTON.hide()

        # layoutTabsAndLeft.addWidget(self.smilei_icon_BUTTON)
        # self.programm_TABS.ad
        # self.programm_TABS.hide()
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


        self.tornado_refresh_BUTTON.clicked.connect(self.call_ThreadDownloadSimJSON)

        #Open and Close Tabs
        self.actionDiagScalar.toggled.connect(lambda: self.onMenuTabs("SCALAR"))
        self.actionDiagFields.toggled.connect(lambda: self.onMenuTabs("FIELDS"))
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

        self.memory_update_TIMER = QtCore.QTimer()
        self.memory_update_TIMER.setInterval(5000) #in ms
        self.memory_update_TIMER.timeout.connect(self.updateInfoLabel)
        self.memory_update_TIMER.start()

        # self.reset.clicked.connect(self.onReset)
        # self.interpolation.currentIndexChanged.connect(self.plot)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)
        self.setWindowTitle("Smilei IFE GUI XDIR")


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
            text = f"""import os\nimport sys\nimport numpy as np\nimport matplotlib.pyplot as plt\nmodule_dir_happi = 'C:/Users/jerem/Smilei'\nsys.path.insert(0, module_dir_happi)\nimport happi\nS = happi.Open('{self.sim_directory_path}')\nl0=2*np.pi\n"""
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
        self.logs_DIALOG = log_dialog.LogsDialog(self.spyder_default_stdout,app)
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

    def updateInfoLabel(self):
        mem_prc = self.MEMORY().used*100/self.MEMORY().total
        if mem_prc > 85:
            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION}  |  Memory: {self.MEMORY().used*100/self.MEMORY().total:.0f}% | Storage: {self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30):.1f} Go | {self.COPY_RIGHT}")

            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | <font color='red'>Memory: {mem_prc:.0f}%</font> | Storage: {self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30):.1f} Go | {self.COPY_RIGHT}")
        elif mem_prc > 75:
            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | <font color='orange'>Memory: {mem_prc:.0f}%</font> | Storage: {self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30):.1f} Go | {self.COPY_RIGHT}")
        else:
            self.general_info_LABEL.setText(f"Version: {self.SCRIPT_VERSION} | Memory: {mem_prc:.0f}% | Storage: {self.DISK(os.environ['SMILEI_CLUSTER']).free/(2**30):.1f} Go | {self.COPY_RIGHT}")

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
        self.a0 = self.S.namelist.a0
        self.Tp = self.S.namelist.Tp
        self.dx = self.S.namelist.dx
        self.Ltrans = self.S.namelist.Ltrans
        self.Llong = self.S.namelist.Llong
        self.tsim = self.S.namelist.tsim
        self.l1 = self.S.namelist.l1
        self.eps = self.S.namelist.eps
        self.ne = self.S.namelist.ne
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

        run_time, push_time = self.getSimRunTime(self.S._results_path[0])
        NODES = self.S.namelist.smilei_mpi_size//2

        self.run_time_LABEL.setText(f"{(run_time/60)//60:.0f}h{(run_time/60)%60:0>2.0f} | {NODES} nds ({push_time:.0f} ns)")


        self.intensity_SI = (self.a0/0.85)**2 *10**18 #W/cm^2

        self.power_SI = self.intensity_SI * pi*(self.w0/l0*10**-4)**2/2

        me = 9.1093837*10**-31
        e = 1.60217663*10**-19
        self.c = 299792458
        eps0 = 8.854*10**-12
        self.toTesla = 10709
        self.wr = 2*pi*self.c/1e-6
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

        self.intensity_SI_LABEL.setText(f"{'%.1E' % decimal.Decimal(str(self.intensity_SI))} W/cm²")
        self.power_SI_LABEL.setText(f"{self.printSI(self.power_SI,'W',ndeci=2):}")
        self.energy_SI_LABEL.setText(f"{self.energy_SI:.2f} mJ")
        self.Tp_SI_LABEL.setText(f"{self.Tp_SI:.0f} fs")

        self.nppc_LABEL.setText(f"{self.S.namelist.nppc_plasma}")
        self.density_LABEL.setText(f"{self.S.namelist.ne} nc")

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

        self.updateInfoLabel()
        if self.actionDiagScalar.isChecked(): self.onUpdateTabScalar(0)
        if self.actionDiagFields.isChecked(): self.onUpdateTabFields(-1)
        if self.actionDiagTrack.isChecked(): self.onUpdateTabTrack(-1)
        if self.actionDiagPlasma.isChecked(): self.onUpdateTabPlasma(-1)
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
        return
    """
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
    """
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
        if tab_name == "COMPA":
            self.actionDiagCompa.setChecked(False)
            self.onRemoveCompa()
            self.onRemovePlasma()
        # print(self.programm_TABS.count())
        if self.programm_TABS.count() ==0:
            self.smilei_icon_BUTTON.show()
        return


    def onMenuTabs(self, tab_name):
        if tab_name == "SCALAR":
            if not self.actionDiagScalar.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "SCALAR":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveScalar()
            else:
                self.programm_TABS.addTab(self.scalar_Widget,"SCALAR")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
                # self.smilei_icon_BUTTON.deleteLater()
                self.INIT_tabScalar = True
                app.processEvents()
                self.onUpdateTabScalar(0)

        if tab_name == "FIELDS":
            if not self.actionDiagFields.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "FIELDS":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveFields()
            else:
                self.programm_TABS.addTab(self.fields_Widget,"FIELDS")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
                self.INIT_tabFields = True
                app.processEvents()
                self.onUpdateTabFields(0)

        if tab_name == "TRACK":
            if not self.actionDiagTrack.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TRACK":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveTrack()
            else:
                self.programm_TABS.addTab(self.track_Widget,"TRACK")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
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
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemovePlasma()
            else:
                self.smilei_icon_BUTTON.hide()
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
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveCompa()
            else:
                self.programm_TABS.addTab(self.compa_Widget,"COMPA")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
                if self.INIT_tabCompa != None: self.INIT_tabCompa = True
                self.INIT_tabCompa = True
                app.processEvents()
                self.onUpdateTabCompa(0)

        if tab_name == "BINNING":
            if not self.actionDiagBinning.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "BINNING":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveBinning()
            else:
                self.programm_TABS.addTab(self.binning_Widget,"BINNING")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
                # if self.INIT_tabCompa != None: self.INIT_tabCompa = True
                # self.INIT_tabCompa = True
                app.processEvents()
                # self.onUpdateTabBinning()

        if tab_name == "TORNADO":
            if not self.actionTornado.isChecked():
                for currentIndex in range(self.programm_TABS.count()):
                    if self.programm_TABS.tabText(currentIndex) == "TORNADO":
                        self.programm_TABS.removeTab(currentIndex)
                        if self.programm_TABS.count() ==0: self.smilei_icon_BUTTON.show()
                        self.onRemoveTornado()
            else:
                self.programm_TABS.addTab(self.tornado_Widget,"TORNADO")
                self.programm_TABS.setCurrentIndex(self.programm_TABS.count()-1)
                self.programm_TABS.show()
                self.smilei_icon_BUTTON.hide()
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
        self.writeToFileCompa(self.compa_sim_directory_path,"AM",np.vstack([fields_t_range,AM_full_int_compa]).T)
        ax.legend()
        ax.relim()            # Recompute the limits based on current data
        ax.autoscale_view()   # Apply the new limits
        figure.tight_layout()
        canvas.draw()


    def onUpdateTabScalar_AM(self, AM_full_int, is_compa=False):
        # print(AM_full_int.shape, is_compa)
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
        # hfont = {'fontname':'Helvetica'}
        # print(AM_tot)

        # print(f"AM/U={AM_tot/self.Uelm_tot_max:.2f}")
        # print(f"Scalar time plot \nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_tot/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ")
        figure.suptitle(f"{self.sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_tot/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)
        if is_compa:
            figure.suptitle(f"{self.sim_directory_name} vs {self.compa_sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_tot/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)

        ax.relim()            # Recompute the limits based on current data
        ax.autoscale_view()   # Apply the new limits
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

        # print("Scalar boolList:",boolList, "| is_compa:",is_compa)

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
                data = np.gradient(np.array(np.array(self.S.Scalar("Uelm").getData())/(np.array(self.S.Scalar("Utot").getData())+1e-12)),self.scalar_t_range/l0)
                ax.plot(self.scalar_t_range/l0, data*1000,
                              label=self.scalar_names[check_id], ls="-",color = f"C{len(ax.get_lines())}")
                alpha_theo = -0.5*self.ne/(self.Tp/2/l0)*1000
                ax.axhline(alpha_theo,ls="--",color="k",alpha=0.5, label=r"$\alpha_{abs}$ theo")
                if is_compa and self.is_compa_sim_loaded:
                    data_t_range_compa = self.compa_S.Scalar("Uelm").getTimes()
                    data_compa =  np.gradient(np.array(self.compa_S.Scalar("Uelm").getData())/(np.array(self.compa_S.Scalar("Utot").getData())+1e-12),data_t_range_compa/l0)
                    ax.plot(data_t_range_compa/l0, data_compa*1000,
                                  label=self.scalar_names[check_id], ls="--",color = f"C{len(ax.get_lines())}")

            elif self.scalar_names[check_id] != "AM":
                data = np.array(self.S.Scalar(self.scalar_names[check_id]).getData())
                ax.plot(self.scalar_t_range/l0, data,
                              label=self.scalar_names[check_id], color = f"C{len(ax.get_lines())}")
                # self.writeToFileCompa(self.sim_directory_path, self.scalar_names[check_id],np.vstack([self.scalar_t_range, data]).T)
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
                AM_max = np.max(AM_data)

                # return
        else:
            # print("\n==================")
            old_lines = ax.get_lines()
            # print("old lines:",old_lines)

            for k, t in enumerate(ax.get_legend().get_texts()): # Delete lines
                # print(k,t.get_text())
                if self.scalar_names[check_id] == t.get_text():
                    old_lines[k].remove()
            updated_lines = ax.get_lines()
            # print("new lines:",updated_lines)
            if len(updated_lines) > 0:
                for k in range(len(updated_lines)): # Recolor lines
                    ax.get_lines()[k].set_color(f"C{k}")
        ax.legend()
        ax.relim()            # Recompute the limits based on current data
        ax.autoscale_view()   # Apply the new limits
        # hfont = {'fontname':'Helvetica'}

        figure.suptitle(f"""{self.sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_max/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ""",fontsize=14)
        if is_compa:
            figure.suptitle(f"{self.sim_directory_name} vs {self.compa_sim_directory_name}\nMAX: $Utot={self.Utot_tot_max*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_max*self.KNL3*1000:.2f}$ mJ; $Ukin={self.Ukin_tot_max*self.KNL3*1000:.2f}$ mJ;  AM/U={AM_max/self.Uelm_tot_max:.2f}\nEND: $Utot={self.Utot_tot_end*self.KNL3*1000:.2f}$ mJ;  $Uelm={self.Uelm_tot_end*self.KNL3*1000:.2f}$ mJ;  $Ukin={self.Ukin_tot_end*self.KNL3*1000:.2f}$ mJ",fontsize=14)

        figure.tight_layout()
        canvas.draw()
        # t1 = time.perf_counter()
        # print("Scalar update:",(t1-t0)*1000,"ms")
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
            self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$")
        else:
            self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[x_idx]/self.l0:.2f}~\lambda$")
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
            # print(len(self.fields_image_list))

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
                        self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$")
                        if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
            else:
                for i,im in enumerate(self.fields_image_list):
                    im.set_data(self.fields_data_list[i][time_idx,xcut_idx,:,:].T)
                    if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
                    self.figure_1.suptitle(f"$t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[xcut_idx]/l0:.2f}~\lambda$")
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
                        self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$")
                else:
                    for i,im in enumerate(self.fields_image_list):
                        im.set_data(self.fields_data_list[i][time_idx,xcut_idx,:,:].T)
                        if self.fields_use_autoscale_CHECK.isChecked(): im.autoscale()
                self.figure_1.suptitle(f"{self.sim_directory_name} | $t={self.fields_t_range[time_idx]/self.l0:.2f}~t_0$ ; $x={self.fields_paxisX[xcut_idx]/l0:.2f}~\lambda$")
                self.canvas_1.draw()
                time.sleep(0.05)
                app.processEvents()

            self.loop_in_process = False
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

    def onRemoveCompa(self):
        if not self.INIT_tabPlasma == True and self.INIT_tabPlasma is not None:
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
                Popup().showError("No TrackParticles diagnostic found")
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

            self.y = self.track_traj["y"][:,::N_part]-self.Ltrans/2
            self.z = self.track_traj["z"][:,::N_part] -self.Ltrans/2
            self.py = self.track_traj["py"][:,::N_part]
            self.pz = self.track_traj["pz"][:,::N_part]
            self.px = self.track_traj["px"][:,::N_part]
            self.r = np.sqrt(self.y**2 + self.z**2)
            self.Lx_track =  self.y*self.pz - self.z*self.py

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

            ax1,ax2 = self.figure_2.subplots(1,2)
            time0 = time.perf_counter()
            mean_coef = 5
            self.track_radial_distrib_im = ax1.scatter(self.r[0]/l0,self.Lx_track[-1],s=1,label="$L_x$")
            ax1.set_xlabel("$r/\lambda$")
            ax1.set_ylabel("$L_x$")
            a_range_r,MLx = self.averageAM(self.r[0], self.Lx_track[-1], 0.5)
            ax1.plot(a_range_r/l0, MLx*mean_coef,"r",label="5<$L_x$>",alpha=0.2)
            ax1.grid()
            ax1.legend()
            # im = ax2.imshow(Lz_interp,extent=extent_interp,cmap="RdYlBu")
            vmax = 1.25*np.nanstd(self.Lx_track[-1])
            # ax2.scatter(self.y[0],self.z[0],s=1)
            self.track_trans_distrib_im = ax2.scatter(self.y[0]/l0,self.z[0]/l0,s=1, c=self.Lx_track[-1], vmin=-vmax,vmax=vmax, cmap="RdYlBu")
            self.figure_2.colorbar(self.track_trans_distrib_im,ax=ax2,pad=0.01)
            self.figure_2.suptitle(f"{self.sim_directory_name} | $t={self.track_t_range[-1]/self.l0:.2f}~t_0$ (N={self.track_N/1000:.2f}k)")
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
            self.track_trans_distrib_im.set_array(self.Lx_track[time_idx])
            if self.track_update_offset_CHECK.isChecked():
                self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                self.track_radial_distrib_im.set_offsets(np.c_[self.r[time_idx]/l0,self.Lx_track[time_idx]])
            else:
                self.track_radial_distrib_im.set_offsets(np.c_[self.r[0]/l0,self.Lx_track[time_idx]])
            self.figure_2.suptitle(f"{self.sim_directory_name} | $t={self.track_t_range[time_idx]/self.l0:.2f}~t_0$ (N={self.track_N/1000:.2f}k)")
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
                if self.track_update_offset_CHECK.isChecked(): self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                self.track_trans_distrib_im.set_array(self.Lx_track[time_idx])
                if self.track_update_offset_CHECK.isChecked():
                    self.track_trans_distrib_im.set_offsets(np.c_[self.y[time_idx]/l0,self.z[time_idx]/l0])
                    self.track_radial_distrib_im.set_offsets(np.c_[self.r[time_idx]/l0,self.Lx_track[time_idx]])
                else:
                    self.track_radial_distrib_im.set_offsets(np.c_[self.r[0]/l0,self.Lx_track[time_idx]])

                self.figure_2.suptitle(f"{self.sim_directory_name} | $t={self.track_t_range[time_idx]/self.l0:.2f}~t_0$ (N={self.track_N/1000:.2f}k)")
                self.canvas_2.draw()
                # print('drawn')
                time.sleep(anim_speed)
                app.processEvents()
            self.loop_in_process = False
        self.updateInfoLabel()


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

    def savePlasmaData(self):
        """Call get Plasma Probe data for all plasma names to save the files to .npz """
        return
        self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, self.plasma_names[:2])
        self.loadthread.start()
        # self.loadthread = class_threading.ThreadGetPlasmaProbeData(self.S, selected_plasma_names=self.plasma_names[5:])
        # self.loadthread.start()
        return

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
                    print(slider_time_idx, new_time_idx)
                    time_idx = new_time_idx
                else: time_idx = slider_time_idx


                if "Bx" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin = -VMAX_Bx
                    vmax =  VMAX_Bx
                elif "pθ" in self.plasma_names[i]:
                    cmap = "RdYlBu"
                    vmin = -vmax_ptheta
                    vmax =  vmax_ptheta
                elif "ne" in self.plasma_names[i]:
                    cmap = "jet"
                    vmin = 0
                    vmax = 3
                elif "Ekin" in self.plasma_names[i]:
                    cmap = "smilei"
                    vmin = 0
                    vmax = 0.1
                else:
                    cmap = "RdYlBu"
                    vmin = -0.05*np.max(np.abs(self.plasma_data_list[k][time_idx]))
                    vmax =  0.05*np.max(np.abs(self.plasma_data_list[k][time_idx]))

                if "trans" in self.plasma_names[i]:
                    extent = self.plasma_extentYZ
                    data = self.plasma_data_list[k][time_idx,x_idx,:,:]
                else:
                    extent = self.plasma_extentXY_long
                    print(self.plasma_data_list[k].shape)
                    data = self.plasma_data_list[k][time_idx].T

                im = ax.imshow(data, aspect="auto",
                                origin="lower", cmap = cmap, extent=extent, vmin=vmin, vmax=vmax) #bwr, RdYlBu #
                ax.set_title(self.plasma_names[i])

                self.figure_3.colorbar(im, ax=ax,pad=0.01)
                self.plasma_image_list.append(im)
                k+=1
        self.figure_3.suptitle(f"{self.sim_directory_name} | $t={self.effective_plasma_t_range[slider_time_idx]/l0:.2f}~t_0$")
        for w in range(10):
            self.figure_3.tight_layout()
            self.figure_3.tight_layout()
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

            print("===== INIT FIELDS PLASMA =====")

            Bx_long_diag = self.S.Probe(2,"Bx")
            self.plasma_paxisX_long = Bx_long_diag.getAxis("axis1")[:,0]
            self.plasma_paxisY_long = Bx_long_diag.getAxis("axis2")[:,1]
            self.plasma_t_range = Bx_long_diag.getTimes()
            self.av_plasma_t_range = self.S.ParticleBinning("weight_av").getTimes()


            Bx_trans_diag = self.S.Probe(1,"Bx")
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
        if check_id < 20: #CHECK_BOX UPDATE
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
                    print(time_idx, new_time_idx)
                    time_idx = new_time_idx
                else: time_idx = self.plasma_time_SLIDER.sliderPosition()

                if "_trans" in selected_plasma_names[i]:
                    im.set_data(self.plasma_data_list[i][time_idx,xcut_idx,:,:])
                    self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($x={self.plasma_paxisX_Bx[xcut_idx]/l0:.1f}~\lambda$)")
                else:
                    im.set_data(self.plasma_data_list[i][time_idx].T)

            self.figure_3.suptitle(f"{self.sim_directory_name} | $t={self.effective_plasma_t_range[slider_time_idx]/l0:.2f}~t_0$")
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
                        print(time_idx, new_time_idx)
                        time_idx = new_time_idx
                    else: time_idx = slider_time_idx

                    if "_trans" in selected_plasma_names[i]:
                        im.set_data(self.plasma_data_list[i][time_idx,xcut_idx,:,:].T)
                        self.figure_3.axes[i*2].set_title(f"{selected_plasma_names[i]} ($x={self.plasma_paxisX_Bx[xcut_idx]/l0:.1f}~\lambda$)")
                    else:
                        im.set_data(self.plasma_data_list[i][time_idx,:,:].T)

                self.figure_3.suptitle(f"{self.sim_directory_name} | t={self.effective_plasma_t_range[slider_time_idx]/self.l0:.2f}$~t_0$")
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
        self.ax4_plasma1.set_title(self.sim_directory_name)
        self.ax4_plasma2 = self.figure_4_plasma.add_subplot(1,2,2)
        self.ax4_plasma2.set_title(self.compa_sim_directory_name)
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

        # if not self.is_compa_sim_loaded: return
        time_idx = self.compa_plasma_time_SLIDER.sliderPosition()
        x_idx = self.compa_plasma_xcut_SLIDER.sliderPosition()

        if "Bx" in self.selected_plasma_name:
            cmap = "RdYlBu"
            vmin = -VMAX_Bx
            vmax =  VMAX_Bx
        elif "pθ" in self.selected_plasma_name:
            cmap = "RdYlBu"
            vmin = -vmax_ptheta
            vmax =  vmax_ptheta
        elif "ne" in self.selected_plasma_name:
            cmap = "jet"
            vmin = 0
            vmax = 3
        elif "Ekin" in self.selected_plasma_name:
            cmap = "smilei"
            vmin = 0
            vmax = 0.1
        else:
            cmap = "RdYlBu"
            vmin = -0.1*np.max(np.abs(self.compa_plasma_data[time_idx]))
            vmax =  0.1*np.max(np.abs(self.compa_plasma_data[time_idx]))

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


        self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.plasma_t_range[time_idx]/self.l0:.2f}~t_0$")
        self.figure_4_plasma.tight_layout()
        self.canvas_4_plasma.draw()
        return

    def onUpdateTabCompaPlasma(self, check_id):
        if self.INIT_tabPlasma == None or self.is_sim_loaded == False:
            # Popup().showError("Simulation not loaded")
            return
        if check_id < 20:
            l0 = 2*pi
            Bx_long_diag = self.S.Probe(2,"Bx")
            self.plasma_paxisX_long = Bx_long_diag.getAxis("axis1")[:,0]
            self.plasma_paxisY_long = Bx_long_diag.getAxis("axis2")[:,1]
            self.plasma_t_range = Bx_long_diag.getTimes()

            Bx_trans_diag = self.S.Probe(1,"Bx")
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


            if self.compa_plasma_t_range[-1] < self.plasma_t_range[-1]:
                t_range = self.compa_plasma_t_range
            else:
                t_range = self.plasma_t_range

            # self.plasma_time_SLIDER.setMaximum(len(self.plasma_t_range)-1)
            # self.plasma_xcut_SLIDER.setMaximum(len(self.plasma_paxisX_Bx)-1)
            # self.plasma_time_SLIDER.setValue(len(self.plasma_t_range)-1)
            # self.plasma_xcut_SLIDER.setValue(len(self.plasma_paxisX_Bx)-3)
            # self.plasma_time_EDIT.setText(str(round(self.plasma_t_range[-1]/l0,2)))
            # self.plasma_xcut_EDIT.setText(str(round(self.plasma_paxisX_Bx[-3]/l0,2)))

            self.compa_plasma_time_SLIDER.setMaximum(len(t_range)-1)
            self.compa_plasma_xcut_SLIDER.setMaximum(len(self.plasma_paxisX_Bx)-1)
            self.compa_plasma_time_SLIDER.setValue(len(t_range)-1)
            self.compa_plasma_xcut_SLIDER.setValue(len(self.plasma_paxisX_Bx)-3)
            self.compa_plasma_time_EDIT.setText(str(round(t_range[-1]/l0,2)))
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

        if check_id < 20:
            self.call_compa_ThreadGetPlasmaProbeData(check_id)

        elif check_id <= 210: #SLIDER UPDATE
            if check_id == 101: #QLineEdit time
                time_edit_value = float(self.compa_plasma_time_EDIT.text())
                time_idx = np.where(abs(self.plasma_t_range/self.l0-time_edit_value)==np.min(abs(self.plasma_t_range/self.l0-time_edit_value)))[0][0]
                self.compa_plasma_time_SLIDER.setValue(time_idx)
                self.compa_plasma_time_EDIT.setText(str(round(self.plasma_t_range[time_idx]/self.l0,2)))
            else:
                time_idx = self.compa_plasma_time_SLIDER.sliderPosition()
                self.compa_plasma_time_EDIT.setText(str(round(self.plasma_t_range[time_idx]/self.l0,2)))

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
                self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.plasma_t_range[time_idx]/self.l0:.2f}~t_0$ ($x={self.plasma_paxisX_Bx[xcut_idx]/self.l0:.1f}~\lambda$)")
            else:
                data1 = self.compa_plasma_data1[time_idx].T
                data2 = self.compa_plasma_data2[time_idx].T
                self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.plasma_t_range[time_idx]/self.l0:.2f}~t_0$")

            self.compa_plasma_im1.set_data(data1)
            self.compa_plasma_im2.set_data(data2)
            self.canvas_4_plasma.draw()
            return

        elif check_id == 1000: #Play button
            if self.loop_in_process: return

            self.loop_in_process = True

            xcut_idx = self.compa_plasma_xcut_SLIDER.sliderPosition()
            for time_idx in range(len(self.plasma_t_range)):
                self.compa_plasma_time_SLIDER.setValue(time_idx)
                self.compa_plasma_time_EDIT.setText(str(round(self.plasma_t_range[time_idx]/self.l0,2)))


                if "trans" in self.selected_plasma_name:
                    data1 = self.compa_plasma_data1[time_idx,xcut_idx,:,:]
                    data2 = self.compa_plasma_data2[time_idx,xcut_idx,:,:]
                    self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.plasma_t_range[time_idx]/self.l0:.2f}~t_0$ ($x={self.plasma_paxisX_Bx[xcut_idx]/self.l0:.1f}~\lambda$)")
                else:
                    data1 = self.compa_plasma_data1[time_idx].T
                    data2 = self.compa_plasma_data2[time_idx].T
                    self.figure_4_plasma.suptitle(f"{self.selected_plasma_name} at $t={self.plasma_t_range[time_idx]/self.l0:.2f}~t_0$")
                self.compa_plasma_im1.set_data(data1)
                self.compa_plasma_im2.set_data(data2)
                self.canvas_4_plasma.draw()
                time.sleep(0.01)
                app.processEvents()

            self.loop_in_process = False
            return

    def onUpdateTabCompa(self, box_idx):
        if box_idx==0:
            self.compa_scalar_groupBox.show()
            self.canvas_4_scalar.show()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
        elif box_idx==1:
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.show()
            self.canvas_4_plasma.show()
            self.compa_binning_groupBox.hide()
            self.canvas_4_binning.hide()
        else:
            self.compa_scalar_groupBox.hide()
            self.canvas_4_scalar.hide()
            self.compa_plasma_groupBox.hide()
            self.canvas_4_plasma.hide()
            self.compa_binning_groupBox.show()
            self.canvas_4_binning.show()
        return


    def onUpdateTabBinning(self, id, is_compa=False):
        if not self.is_sim_loaded:
            # Popup().showError("Simulation not loaded")
            return
        if is_compa and not self.is_compa_sim_loaded:
            Popup().showError("2nd simulation not loaded")
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
                    diag = self.S.ParticleBinning("weight_av")
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
                        diag2 = self.compa_S.ParticleBinning("weight_av")
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
                    Popup().showError(f'No ParticleBinning diagnostic "{diag_name}" found')
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
                    if diag_name == "ekin":
                        binning_image, = ax.plot(x_range,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.set_xlabel("Ekin")
                        if is_compa: binning_image2, = ax.plot(x_range,binning_data2[time_idx], label=diag_name+"_compa")

                    elif diag_name=="Lx_x" or diag_name=="Lx_x_av" or diag_name=="Lx_r":
                        x_range = np.array(diag.getAxis("x"))
                        ax.set_xlabel("$x/\lambda$")

                        if diag_name=="Lx_r":
                            x_range = diag.getAxis("user_function0")
                            idx = round(0.3*binning_data.shape[0]) #Average over 30 - 70% of the range to remove transiant effects
                            ax.set_xlabel("$r/\lambda$")
                            m = np.nanmean(binning_data[idx:-idx],axis=0)
                            std = np.nanstd(binning_data[idx:-idx],axis=0)
                            ax.fill_between(x_range/self.l0, m-std, m+std, color="gray",alpha=0.25)
                            ax.plot(x_range/self.l0,m, "k--",label=diag_name+" time average")
                            if is_compa:
                                m = np.nanmean(binning_data2[idx:-idx],axis=0)
                                ax.plot(x_range/self.l0, m, "-.", color="slategray",label=diag_name+"_compa time average")
                        ax.set_ylabel("$L_x$")
                        binning_image, = ax.plot(x_range/self.l0,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        if is_compa: binning_image2, = ax.plot(x_range/self.l0,binning_data2[time_idx], label=diag_name+"_compa")

                    else:
                        binning_image, = ax.plot(x_range/self.l0,binning_data[time_idx], label=diag_name)
                        binning_image_list.append(binning_image)
                        if is_compa: binning_image2, = ax.plot(x_range/self.l0,binning_data2[time_idx], label=diag_name+"_compa")
                        ax.set_xlabel(diag_name)
                        ax.set_ylabel("weight")
                        print(diag_name)
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
                        ax.set_title(self.sim_directory_name)
                        ax2.set_title(self.compa_sim_directory_name)

                    if diag_name =="phase_space":
                        x_range  = diag.getAxis("x")
                        px_range = diag.getAxis("px")
                        extent = [x_range[0]/self.l0,x_range[-1]/self.l0,px_range[0],px_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("$x/\lambda$")
                        ax.set_ylabel("px")
                    elif diag_name =="phase_space_v":
                        vy_range  = diag.getAxis("vy")
                        vz_range = diag.getAxis("vz")
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
                        if is_compa: binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        ax.set_xlabel("$x/\lambda$")
                        ax.set_ylabel("Lx")
                    elif diag_name =="phase_space_Lx_r" or diag_name =="phase_space_Lx_r_zoom":
                        r_range  = diag.getAxis("user_function0")
                        Lx_range = diag.getAxis("user_function1")
                        extent = [r_range[0]/self.l0,r_range[-1]/self.l0,Lx_range[0],Lx_range[-1]]
                        binning_image = ax.imshow(data[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
                        if is_compa: binning_image2 = ax2.imshow(data2[time_idx].T, extent=extent, cmap="smilei",aspect="auto", origin="lower")
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
            figure.suptitle(f"{self.sim_directory_name} | t = {t_range[time_idx]/self.l0:.2f} $t_0$")
            if is_compa:
                figure.suptitle(f"{self.sim_directory_name} vs {self.compa_sim_directory_name}| t = {t_range[time_idx]/self.l0:.2f} $t_0$")

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

            # print(t_range)

            if binning_data.ndim == 1:
                if self.binning_log_CHECK.isChecked():
                    binning_image.axes.set_yscale("log")
                    if is_compa: binning_image2.axes.set_yscale("log")
                    canvas.draw()
                else:
                    binning_image.axes.set_yscale("linear")
                    if is_compa: binning_image2.axes.set_yscale("linear")
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

        figure.suptitle(f"{self.sim_directory_name} | t = {t_range[time_idx]/self.l0:.2f} $t_0$")
        canvas.draw()
        return




    def onCloseProgressBar(self, sim_id_int):
        print("REMOVE PROGRESS BAR LAYOUT", sim_id_int)
        print("finished sim:",self.finished_sim_hist)
        self.finished_sim_hist.remove(sim_id_int)
        layout_to_del = self.layout_progress_bar_dict[str(sim_id_int)]
        # print("to del:",layout_to_del)
        for i in range(self.layoutTornado.count()):
            layout_progressBar = self.layoutTornado.itemAt(i)
            # print(i,layout_progressBar)
            if layout_progressBar == layout_to_del:
                # print("delete:",layout_progressBar)
                self.deleteLayout(self.layoutTornado, i)

    def async_onUpdateTabTornado(self, download_trnd_json = True):
        # print("async check")
        """
        asynchronous function called PERIODICALLY
        """
        sim_json_name = "simulations_info.json"
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
                print(n,old_sim_id_int)
                if str(old_sim_id_int) not in list(self.sim_dict): #this simulation has finished
                    print(old_sim_id_int,"not in", list(self.sim_dict))
                    finished_sim_path = self.previous_sim_dict[str(old_sim_id_int)]["job_full_path"]
                    finished_sim_name = self.previous_sim_dict[str(old_sim_id_int)]["job_full_name"]
                    print(finished_sim_path,"download is available ! \a") #\a
                    Popup().showToast('Tornado download is available', finished_sim_name)

                    self.finished_sim_hist.append(old_sim_id_int)
                    self.running_sim_hist.remove(old_sim_id_int)
                    self.can_download_sim_dict[int(old_sim_id_int)] = finished_sim_path

                    layout = self.layout_progress_bar_dict[str(old_sim_id_int)]
                    progress_bar = layout.itemAt(2).widget()
                    ETA_LABEL = layout.itemAt(3).widget()
                    dl_sim_BUTTON = layout.itemAt(5).widget()

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
            sim_name = sim["job_full_name"][:-3]
            sim_nodes = int(sim["NODES"])
            sim_push_time = sim["push_time"]


            if (sim_id_int not in self.running_sim_hist) and (sim_id_int not in self.finished_sim_hist):

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA, sim_push_time)
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
                push_time_label = self.layout_progress_bar_dict[sim_id].itemAt(4).widget()
                if sim_push_time > 10_000:
                    push_time_label.setStyleSheet("color: red")
                push_time_label.setText(str(sim_push_time)+"ns")

        #Update label with Update datetime
        sim_datetime = self.sim_dict["datetime"]
        self.tornado_last_update_LABEL.setText(f"Last updated: {sim_datetime}")

        self.previous_sim_dict = self.sim_dict
        app.processEvents()
        return

    def call_ThreadDownloadSimJSON(self):
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
        with open('../tornado_pwdfile.txt', 'r') as f: pwd_crypt = f.read()
        pwd = encrypt(pwd_crypt,-2041000*2-1)
        remote_path = "/sps3/jeremy/LULI/"
        ssh_key_filepath = r"C:\Users\jerem\.ssh\id_rsa.pub"
        remote_path = "/sps3/jeremy/LULI/"
        remote_client = paramiko_SSH_SCP_class.RemoteClient(host,user,pwd,ssh_key_filepath,remote_path)
        res = remote_client.execute_commands([f"du {job_full_path}"])
        total_size = int(res[0].split()[0])

        tornado_download_TIMER = QtCore.QTimer()
        refresh_time_s = 2 #s
        tornado_download_TIMER.setInterval(int(refresh_time_s*1000)) #in ms
        tornado_download_TIMER.timeout.connect(partial(self.onUpdateDownloadBar,total_size, job_full_path, sim_id))
        tornado_download_TIMER.start()

        self.loadthread = class_threading.ThreadDownloadSimData(job_full_path)
        self.loadthread.finished.connect(partial(self.onDownloadSimDataFinished,sim_id, tornado_download_TIMER))
        self.loadthread.start()

        layout = self.layout_progress_bar_dict[str(sim_id)]
        # progress_bar = layout.itemAt(2).widget()
        ETA_label = layout.itemAt(3).widget()
        dl_sim_BUTTON = layout.itemAt(5).widget()
        dl_sim_BUTTON.setStyleSheet("border-color: orange")
        dl_sim_BUTTON.setEnabled(False)
        close_sim_BUTTON = layout.itemAt(6).widget()
        close_sim_BUTTON.setEnabled(False)
        ETA_label.setText("DL")
        return
    def onUpdateDownloadBar(self, total_size, job_full_path, sim_id):
        general_folder_name = job_full_path[18:]
        local_folder = os.environ["SMILEI_CLUSTER"]
        local_sim_path = f"{local_folder}\\{general_folder_name}"

        size = sum([os.path.getsize(f"{local_sim_path}\{f}") for f in os.listdir(local_sim_path)])
        prc = round(size/(total_size*1024)*100)
        print(f"tornado download {sim_id}:",prc,"%")
        layout = self.layout_progress_bar_dict[str(sim_id)]
        progress_bar = layout.itemAt(2).widget()
        progress_bar.setValue(prc)
        self.updateInfoLabel()
        return

    def onDownloadSimDataFinished(self,sim_id, tornado_download_TIMER):
        tornado_download_TIMER.stop()
        layout = self.layout_progress_bar_dict[str(sim_id)]
        progress_bar = layout.itemAt(2).widget()
        ETA_label = layout.itemAt(3).widget()
        dl_sim_BUTTON = layout.itemAt(5).widget()
        close_sim_BUTTON = layout.itemAt(6).widget()
        dl_sim_BUTTON.setStyleSheet("border-color: green")
        dl_sim_BUTTON.setEnabled(False)
        close_sim_BUTTON.setEnabled(True)
        pixmap = QtGui.QPixmap(os.environ["SMILEI_QT"]+"\\Ressources\\green_check_icon.jpg")
        pixmap = pixmap.scaled(ETA_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation);

        ETA_label.setPixmap(pixmap) #green check

        ETA_label.setStyleSheet("background-color: #80ef80")
        progress_bar.setStyleSheet(self.qss_progressBar_DOWNLOADED)

        print(sim_id,"download is finished ! \a") #\a
        Popup().showToast('Tornado download is finished', sim_id,ToastPreset.INFORMATION)

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

            for sim_id in self.sim_dict:
                if sim_id == "datetime": continue #only open sim data and not metadata (located at the end of dict)
                sim = self.sim_dict[sim_id]
                sim_progress = sim["progress"]*100
                sim_ETA = sim["ETA"].rjust(5)
                sim_name = sim["job_full_name"][:-3]
                sim_nodes = int(sim["NODES"])
                sim_push_time = sim["push_time"]

                self.running_sim_hist.append(int(sim_id))

                layoutProgressBar = self.createLayoutProgressBar(sim_id, sim_progress, sim_name, sim_nodes, sim_ETA, sim_push_time)

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

    def createLayoutProgressBar(self, sim_id, sim_progress, sim_name, sim_nodes, sim_ETA, sim_push_time):
        layoutProgressBar = QtWidgets.QHBoxLayout()

        tornado_PROGRESS_BAR = QtWidgets.QProgressBar(maximum=100)
        tornado_PROGRESS_BAR.setValue(round(sim_progress))
        tornado_PROGRESS_BAR.setFont(QFont('Arial', 15))
        tornado_PROGRESS_BAR.setAlignment(QtCore.Qt.AlignCenter)

        custom_bold_FONT = QtGui.QFont("Courier New", 14,QFont.Bold)
        custom_FONT = QtGui.QFont("Courier New", 12)

        sim_name_LABEL = QtWidgets.QLabel(f"[{sim_id}] {sim_name}")
        sim_name_LABEL.setFont(custom_bold_FONT)
        sim_name_LABEL.setMinimumWidth(475) #450 FOR LAPTOP
        sim_name_LABEL.setStyleSheet("background-color: lightblue")
        sim_name_LABEL.setWordWrap(True)
        sim_name_LABEL.setAlignment(QtCore.Qt.AlignCenter)

        sim_node_LABEL = QtWidgets.QLabel(f"NDS:{sim_nodes}")
        sim_node_LABEL.setFont(custom_bold_FONT)
        sim_node_LABEL.setStyleSheet("background-color: lightblue")

        ETA_LABEL = QtWidgets.QLabel(sim_ETA)
        ETA_LABEL.setFont(custom_bold_FONT)
        ETA_LABEL.setStyleSheet("background-color: lightblue")
        # ETA_LABEL.setAlignment(QtCore.Qt.AlignCenter)
        ETA_LABEL.setMinimumWidth(75)

        push_time_LABEL = QtWidgets.QLabel(str(sim_push_time)+"ns")
        if sim_push_time > 10_000:
            push_time_LABEL.setStyleSheet("color: red")
        push_time_LABEL.setFont(custom_FONT)
        push_time_LABEL.setMinimumWidth(75)

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
        self.loadthread = class_threading.ThreadDownloadSimData(job_full_path)
        self.loadthread.start()
        # self.downloadSimData(job_full_path) #"_NEW_PLASMA_/new_plasma_LG_optic_ne0.01_dx12/")

    # def showToast(self,msg1,msg2=None, preset=ToastPreset.SUCCESS):
    #     toast = Toast(self)
    #     toast.setDuration(10000)  # Hide after 10 seconds
    #     toast.setTitle(msg1)
    #     toast.setText(msg2)
    #     toast.applyPreset(preset)  # Apply style preset
    #     toast.setBorderRadius(2)  # Default: 0

    #     toast.show()
    # def showError(self, message):
    #     self.error_msg = QtWidgets.QMessageBox()
    #     self.error_msg.setIcon(QtWidgets.QMessageBox.Critical)
    #     self.error_msg.setWindowTitle("Error")
    #     self.error_msg.setText(message)
    #     self.error_msg.exec_()

    def printSI(self,x,baseunit,ndeci=2):
        prefix="yzafpnµm kMGTPEZY"
        shift=decimal.Decimal('1E24')
        d=(decimal.Decimal(str(x))*shift).normalize()
        m,e=d.to_eng_string().split('E')
        return m[:4] + " " + prefix[int(e)//3] + baseunit

    def getSimRunTime(self, sim_path):
        with open(sim_path+"\\log") as f:
            text = f.readlines()
            for i, line in enumerate(text):
                if "push time [ns]" in line:
                    pt = int(np.mean([int(text[i+n].split()[-1]) for n in range(1,40)]))
                    print("-----------------")
                if "Time_in_time_loop" in line:
                    run_time = float(line.split()[1])
        return run_time, pt


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

    pixmap = QtGui.QPixmap(os.environ["SMILEI_QT"]+'\\Ressources\\Smilei_GUI_logo_V3.png')
    splash = QtWidgets.QSplashScreen(pixmap)
    splash.show()
    app.processEvents()
    # time.sleep(25)

    main = MainWindow()
    main.show()
    splash.finish(main)
    sys.exit(app.exec_())