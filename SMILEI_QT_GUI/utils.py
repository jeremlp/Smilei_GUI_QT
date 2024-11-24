# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 11:39:03 2024

@author: jerem
"""
from pyqttoast import Toast, ToastPreset
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
class Popup():

    def showToast(self,msg1,msg2=None, preset=ToastPreset.SUCCESS):
        toast = Toast()
        toast.setDuration(10000)  # Hide after 10 seconds
        toast.setTitle(msg1)
        toast.setText(msg2)
        toast.applyPreset(preset)  # Apply style preset
        toast.setBorderRadius(2)  # Default: 0
    
        toast.show()
        
    def showError(self, message):
        error_msg = QtWidgets.QMessageBox()
        error_msg.setIcon(QtWidgets.QMessageBox.Critical)
        error_msg.setWindowTitle("Error")
        error_msg.setText(message)
        error_msg.exec_()
        
def encrypt(text,s):
    result = ""
    # transverse the plain text
    for i in range(len(text)):
       char = text[i]
       # Encrypt uppercase characters in plain text
    
       if (char.isupper()):
          result += chr((ord(char) + s-65) % 26 + 65)
       # Encrypt lowercase characters in plain text
       elif ord(char) > 90:
           result += chr((ord(char) + s - 97) % 26 + 97)
       else:
          result += chr((ord(char) + s - 48) % 26 + 48)
    return result

def getSimRunTime(sim_path):
    try:
        with open(sim_path+"\\log") as f:
            text = f.readlines()
            for i, line in enumerate(text):
                if "push time [ns]" in line:
                    pt = int(np.mean([int(text[i+n].split()[-1]) for n in range(1,20)]))
                    # print("-----------------")
                if "Time_in_time_loop" in line:
                    run_time = float(line.split()[1])
        return run_time, pt
    except FileNotFoundError:
        return 0,0
