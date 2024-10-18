# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:07:19 2024

@author: jerem
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QFont
import sys
import numpy as np
import time
import io

import ctypes
import psutil
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from matplotlib.figure import Figure
import datetime
import matplotlib
class MemoryDialog(QtWidgets.QMainWindow):
    def __init__(self, main_app):
        super(MemoryDialog, self).__init__()
        self.setGeometry(50, 50, 600, 400)
        self.setWindowTitle("Memory Graph")

        myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self.setWindowIcon(QtGui.QIcon('D:/JLP/CMI/_MASTER 2_/_STAGE_LULI_/SMILEI_QT_GUI/log_icon.png'))
        self.home()

        self.main_app = main_app

        self.MEMORY = psutil.virtual_memory

    def home(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plt_toolbar = NavigationToolbar(self.canvas)
        self.plt_toolbar.setFixedHeight(35)
        self.ax = self.figure.add_subplot(1,1,1)
        self.ax.grid()
        self.ax.set_xlabel("t (s)")
        self.ax.set_ylabel("Mem (%)")
        self.line, = self.ax.plot([],[],"-")
        self.figure.tight_layout()

        self.time_list = []
        self.mem_list = []

        self.update_ms = 200

        self.counter = 0


        layout = QtWidgets.QVBoxLayout(w)

        layout.addWidget(self.plt_toolbar)
        layout.addWidget(self.canvas)

        self.memory_update_TIMER = QtCore.QTimer()
        self.memory_update_TIMER.setInterval(self.update_ms) #in ms
        self.memory_update_TIMER.timeout.connect(self.updateGraph)
        self.memory_update_TIMER.start()

        self.show()


    def updateGraph(self):
        # t0 = time.perf_counter()
        mem = self.MEMORY().used*100/self.MEMORY().total
        self.counter += 1
        self.time_list.append(self.counter*self.update_ms/1000)
        self.mem_list.append(mem)
        if len(self.mem_list)*self.update_ms/1000/60 > 2: # 2min
            self.mem_list = self.mem_list[1:]
            self.time_list = self.time_list[1:]


        self.line.set_data(self.time_list,self.mem_list)
        self.ax.relim()            # Recompute the limits based on current data
        self.ax.autoscale_view()   # Apply the new limits
        self.canvas.draw()
        # t1 = time.perf_counter()
        # print((t1-t0)*1000,"ms")


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    GUI = MemoryDialog(app)
    sys.exit(app.exec_())