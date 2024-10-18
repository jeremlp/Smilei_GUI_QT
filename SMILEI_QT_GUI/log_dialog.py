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

class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def saveValue(self,text):
        self.text_history.append(text)

    def getvalue(self,text):
        return self.text_history
    def flush(self):
        pass


class LogsDialog(QtWidgets.QMainWindow):
    def __init__(self, default_std_out, main_app):
        super(LogsDialog, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Smilei Logs")

        myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self.setWindowIcon(QtGui.QIcon('D:/JLP/CMI/_MASTER 2_/_STAGE_LULI_/SMILEI_QT_GUI/log_icon.png'))
        self.home()

        self.main_app = main_app

        self.old_stdout = default_std_out
        self.stream = Stream(newText=self.onUpdateText)

        self.buffer = []

        sys.stdout = self.stream
        print("init dialog",sys.stdout)

    def onUpdateText(self, text):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()
        self.main_app.processEvents()
        self.bufferForConsol(text)
        sys.stdout = self.stream
        self.main_app.processEvents()
        return

    def bufferForConsol(self,text):
        sys.stdout = self.old_stdout
        if text =="\n":
            long_print ="".join(self.buffer)
            self.buffer = []
            print(long_print)
        else:
            self.buffer.append(text)
        sys.stdout = self.stream

    def initHistory(self,history):
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(history)

    def saveHistory(self):
        return self.text_edit.toPlainText()


    def __del__(self):
        sys.stdout = self.old_stdout

    def home(self):

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        lay = QtWidgets.QVBoxLayout(w)
        # btn = QtWidgets.QPushButton("Generate")
        # btn.clicked.connect(self.generateText)

        self.text_edit  = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.moveCursor(QtGui.QTextCursor.Start)
        self.text_edit.ensureCursorVisible()
        self.text_edit.setLineWrapColumnOrWidth(500)
        self.text_edit.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)

        # lay.addWidget(btn)
        lay.addWidget(self.text_edit)

        self.show()

    def generateText(self):
        print("test","of","smth")
        # for i in range(15):
        #     print("test","of","smth")
        #     time.sleep(0.25)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    GUI = LogsDialog(sys.stdout,app)
    sys.exit(app.exec_())