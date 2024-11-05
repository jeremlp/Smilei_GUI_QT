# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:34:14 2024

@author: jerem
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui

import log_dialog
import IPython_dialog
import memory_dialog
import tree_dialog



class ToolsDialog(QtWidgets.QMainWindow):
    def __init__(self, main_app):
        super().__init__()

        # Set up the main widget and layout
        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        
        self.main_app = main_app

        # Create a grid layout
        grid_layout = QtWidgets.QGridLayout()
        central_widget.setLayout(grid_layout)

        # Add 4 widgets to the grid layout (2x2)
        
        widget1 = tree_dialog.TreeDialog(self)
        widget1.setMinimumWidth(1000)
        widget2 = memory_dialog.MemoryDialog(self)
        widget3 = IPython_dialog.IPythonDialog(self)
        self.spyder_default_stdout = sys.stdout

        widget4 = log_dialog.LogsDialog(self.spyder_default_stdout, self.main_app)

        # Add widgets to the layout
        grid_layout.addWidget(widget1, 0, 0)  # Row 0, Column 0
        grid_layout.addWidget(widget2, 0, 1)  # Row 0, Column 1
        grid_layout.addWidget(widget3, 1, 0)  # Row 1, Column 0
        grid_layout.addWidget(widget4, 1, 1)  # Row 1, Column 1

        # Set the window title and dimensions
        self.setWindowTitle("Grid Layout Example")
        self.setGeometry(100, 100, int(1000*16/9), 1000)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    
    main = ToolsDialog(app)
    main.show()
    sys.exit(app.exec_())
