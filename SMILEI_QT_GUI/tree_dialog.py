import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QIcon, QAction
import matplotlib.pyplot as plt

import numpy as np


SMILEI_CLUSTER = os.environ["SMILEI_CLUSTER"]

import generate_diag_id
import utils

def open_folder(path):
    dic = {}
    
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    name = path.split("\\")[-1]
    is_sim_folder = "laser_propagation_3d.py" in onlyfiles

    for folder in onlyfolders:
        folder_path = os.path.join(path, folder)
        dic[folder] = open_folder(folder_path)
    
    if is_sim_folder:
        diag_id = generate_diag_id.get_diag_id(f"{path}\laser_propagation_3d.py")
        run_time, push_time = utils.getSimRunTime(path)
        run_time_format = f"{(run_time/60)//60:.0f}h{(run_time/60)%60:0>2.0f}"
        
        try:
            with open(f'{path}\log') as f:
                lines = f.readlines()
            for l in lines:
                if "assumes a global number of" in l:
                    l_list = l.split()
                    for i,c in enumerate(l_list):
                        if c == "of":
                            nodes = int(l_list[i+1])//24//2
                            break
        except FileNotFoundError:
            nodes = "NA"
        
        dic['DIAG_ID'] = diag_id
        dic['RUN_TIME'] = run_time_format
        dic['NODES'] = nodes
    
    return dic


        
# r = open_folder(SMILEI_CLUSTER)
    
# azeaezaezaze


import os
data_folders = {}
for root, dirs, files in os.walk(os.environ["SMILEI_CLUSTER"],topdown=True):
    for name in dirs:
        path = os.path.join(root, name)
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        is_sim_folder = "laser_propagation_3d.py" in onlyfiles
        if is_sim_folder:
            data_folders[name] = []

class TreeDialog(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('Smilei TreeView')
        self.setGeometry(100, 100, 1000, 700)


        
        self.data = open_folder(SMILEI_CLUSTER)
        self.matplotlib_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.diag_color_map = {}
        self.color_index = 0  # To track the color index
    
    
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["SIMULATION FOLDER", "DIAG ID","RUN_TIME", "NODES"])
                
        self.setStyleSheet("""
            QHeaderView::section {
                background-color: #1a73e8; /* Blue background for header */
                color: white; /* White text color */
                font-weight: bold; /* Bold text */
                padding: 10px; /* Padding for spacing */
                border: 1px solid #cccccc; /* Light gray border */
                border-radius: 5px; /* Rounded corners */
            }
            QTreeWidget {
                background-color: #f0f0f0; /* Light gray background for tree */
                alternate-background-color: #ebebf0;
                color: #333; /* Dark text color */
            }
            QTreeWidget::item { margin: 5px; }
 
            
        """)

        font = QtGui.QFont("Arial", 12)  # You can change "Arial" to any preferred font
        self.tree.setFont(font)
        self.tree.setColumnWidth(0, 600)  # Set a fixed width for the "Name" column
        # self.tree.setColumnWidth(1, 150)  # Set a fixed width for the "Diag_id" column
        # self.tree.setColumnWidth(2, 150)  # Set a fixed width for the "NDS" column

        self.tree.setAlternatingRowColors(True)  # Alternate row colors for better readability
        # self.tree.setStyleSheet("QTreeWidget::item { margin: 5px; }")  # Adds spacing within each item
        

        # Set header alignment to center
        self.tree.headerItem().setTextAlignment(0, Qt.AlignCenter)
        self.tree.headerItem().setTextAlignment(1, Qt.AlignCenter)
        self.tree.headerItem().setTextAlignment(2, Qt.AlignCenter)

        
        # Remove the grid lines
        # tree.setRootIsDecorated(False)
        self.tree.setHeaderHidden(False)
       
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        self.populate_tree(self.data)
        
        self.tree.resizeColumnToContents(0)
        
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.tree)
        central_widget.setLayout(layout)

        # self.setCentralWidget(self.tree)
        
    def populate_tree(self,data, parent=None):

        for key, value in data.items():
            if isinstance(value, dict):# If the value is a dictionary
                diag_id = value.get('DIAG_ID', "")
                run_time = value.get('RUN_TIME', "")
                nodes = value.get('NODES', "")
                item = QTreeWidgetItem([key,str(diag_id), str(run_time), str(nodes)])

                if diag_id not in self.diag_color_map and diag_id != '':
                    self.diag_color_map[diag_id] = QtGui.QColor(self.matplotlib_colors[self.color_index])
                    self.color_index += 1   # Cycle through colors
                if diag_id !='':
                    item.setForeground(1, self.diag_color_map[diag_id])
                # item.setText(diag_id_item)
                # item.setText(run_time_item)
                item.setTextAlignment(1, Qt.AlignCenter)  # Center align the DIAG_ID
                item.setTextAlignment(2, Qt.AlignCenter)  # Center align the RUN_TIME

                
                if parent is None:
                    self.tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                    font = QtGui.QFont()
                    font.setBold(True)  # Set the font to bold for simulation items
                    parent.setFont(0, font) 
                
                self.populate_tree(value, item)
            else:
                continue

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.tree.resizeColumnToContents(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TreeDialog()
    window.show()
    sys.exit(app.exec())