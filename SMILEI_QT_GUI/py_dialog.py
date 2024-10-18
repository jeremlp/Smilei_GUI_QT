# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 19:07:19 2024

@author: jerem
"""
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QFont
import sys
import numpy as np
import time
import io

import ctypes
import re


# from pyqtconsole.console import PythonConsole
# from pyqtconsole.highlighter import format

class PyDialog(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super(PyDialog, self).__init__()
        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle("Smilei IPython")

        myappid = u'mycompany.myproduct.subproduct.version' # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        self.setWindowIcon(QtGui.QIcon('D:/JLP/CMI/_MASTER 2_/_STAGE_LULI_/SMILEI_QT_GUI/code_icon.jpg'))
        self.home()

        self.main = parent

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

    def home(self):

        font = QtGui.QFont()
        font.setPointSize(12)

        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        lay = QtWidgets.QVBoxLayout(w)
        submit_BUTTON = QtWidgets.QPushButton("Submit")
        submit_BUTTON.clicked.connect(self.submitCode)

        self.text_edit  = QtWidgets.QTextEdit()
        self.text_edit.setFont(font)
        # self.text_edit.setStyleSheet("background-color: #FFFFF;")
        self.base_color = QtGui.QColor(0, 0, 0)
        self.text_edit.setTextColor(self.base_color)
        #ffffff

        #19232d

        # self.text_edit.setReadOnly(True)
        self.text_edit.moveCursor(QtGui.QTextCursor.Start)
        self.text_edit.ensureCursorVisible()
        self.text_edit.setLineWrapColumnOrWidth(500)
        self.text_edit.setLineWrapMode(QtWidgets.QTextEdit.FixedPixelWidth)


        lay.addWidget(self.text_edit)
        lay.addWidget(submit_BUTTON)
        self.text_edit.setTextColor(self.base_color)

        # console = PythonConsole(formats={
        # 'keyword': format('darkBlue', 'bold')
        # })
        # # console.push_local_ns('greet', greet)
        # console.show()
        # console.eval_queued()
        self.text_edit.textChanged.connect(self.onTextEdited)
        self.text_edit.cursorPositionChanged.connect(self.onCursor)
        self.text_edit.copyAvailable.connect(self.onSelection)
        self.isBeingColored = False
        self.isBeingSelected = False
        self.count_built_in = 0
        self.count_numbers = 0
        self.count_self = 0
        self.count_str = 0

        self.show()
    def onSelection(self):
        if self.isBeingSelected: return
        # print("onSelection")
        text = self.text_edit.toPlainText()

        built_in_word = ["print", "int", "float"]
        pattern_built_in = r'\b(' + '|'.join(built_in_word) + r')\b'
        pattern_str = r'"(.*?)"'

        nb_print = text.count("print")
        nb_built_in = len(re.findall(pattern_built_in, text))
        nb_str = len(re.findall(pattern_str, text))
        nb_numbers = len(re.findall(r'\d', text))
        nb_self = text.count("self")


        old_cursor_pos = self.text_edit.textCursor().position()
        old_cursor_start = self.text_edit.textCursor().selectionStart()
        old_cursor_end= self.text_edit.textCursor().selectionEnd()
        # print(old_cursor_pos, old_cursor_start,old_cursor_end)
        select_length = old_cursor_end-old_cursor_start

        new_text = re.sub(pattern_built_in, r"<span style='color: #ae81ff'>\1</span>", text)
        new_text = new_text.replace("self", "<span style='color: #f92672'>self</span>")
        new_text = re.sub(r'(?<!#)\b\d+\b(?![a-fA-F0-9])', r"<span style='color: #a6e22e'>\g<0></span>", new_text)
        new_text = re.sub(pattern_str, r'<span style="color: green">"\1"</span>', new_text)

        self.isBeingColored = True
        self.isBeingSelected = True
        self.text_edit.setText("")
        # print(new_text.split("\n"))
        for i,txt in enumerate(new_text.split("\n")):
            if txt=="": continue
            self.text_edit.insertHtml(txt)
            # print(i,repr(txt))
            if i < len(new_text.split("\n")):
                self.text_edit.append("")

        cursor = self.text_edit.textCursor()
        # print("cursor moved")

        if old_cursor_pos != old_cursor_start or old_cursor_pos != old_cursor_end:
            if old_cursor_pos==old_cursor_start:
                cursor.setPosition(old_cursor_pos+1)
                cursor.setPosition(old_cursor_end,QtGui.QTextCursor.KeepAnchor)
            else:
                cursor.setPosition(old_cursor_pos-1)
                cursor.setPosition(old_cursor_start,QtGui.QTextCursor.KeepAnchor)
        else:
            cursor.setPosition(old_cursor_pos)
        # print("new:",cursor.position())
        self.text_edit.setTextCursor(cursor)

        self.isBeingSelected = False
        self.isBeingColored = False
        self.count_built_in = nb_built_in
        self.count_numbers = nb_numbers
        self.count_self = nb_self
        self.count_str = nb_str
        self.text_edit.setTextColor(self.base_color)
        # print("====== selected ======")

    def onCursor(self):
        self.text_edit.setTextColor(self.base_color)

    def onTextEdited(self):
        # print("onTextEdited")
        # return
        # print("-------------")
        text = self.text_edit.toPlainText()

        # print(text)

        built_in_word = ["print", "int", "float"]
        pattern_built_in = r'\b(' + '|'.join(built_in_word) + r')\b'
        pattern_str = r'"(.*?)"'

        nb_print = text.count("print")
        nb_built_in = len(re.findall(pattern_built_in, text))
        nb_str = len(re.findall(pattern_str, text))
        nb_numbers = len(re.findall(r'\d', text))
        nb_self = text.count("self")

        count_lower_cond = nb_built_in<self.count_built_in or nb_numbers < self.count_numbers or nb_self < self.count_self or nb_str < self.count_str
        count_diff_cond = nb_built_in!=self.count_built_in or nb_numbers != self.count_numbers or nb_self != self.count_self or nb_str != self.count_str

        if count_lower_cond:
            self.count_built_in = nb_built_in
            self.count_numbers = nb_numbers
            self.count_self = nb_self
            self.count_str = nb_str
            self.text_edit.setTextColor(self.base_color)
            return

        # print(self.count_built_in,self.count_numbers)
        # print(text)
        new_text = re.sub(pattern_built_in, r"<span style='color: #ae81ff'>\1</span>", text)
        new_text = new_text.replace('self', "<span style='color: #f92672'>self</span>")
        new_text = re.sub(r'(?<!#)\b\d+\b(?![a-fA-F0-9])', r"<span style='color: #a6e22e'>\g<0></span>", new_text)
        new_text = re.sub(pattern_str, r'<span style="color: green">"\1"</span>', new_text)

        if new_text != text and  count_diff_cond and not self.isBeingColored:
            # print('EDITED HIGHLIGHT')
            old_cursor_pos = self.text_edit.textCursor().position()
            # print(old_cursor_pos)
            # print("color")
            save_cursor = self.text_edit.textCursor()
            self.isBeingColored = True
            self.text_edit.setText("")
            for i,txt in enumerate(new_text.split('\n')):
                if txt == "": continue
                self.text_edit.insertHtml(txt)
                if i < len(new_text.split('\n')):
                    self.text_edit.append('')

            cursor = self.text_edit.textCursor()
            cursor.setPosition(old_cursor_pos)
            # print("new:",cursor.position())
            self.text_edit.setTextCursor(cursor)


            self.isBeingColored = False
            self.count_built_in = nb_print
            self.count_numbers = nb_numbers
            self.count_self = nb_self
            self.count_str = nb_str
            self.text_edit.setTextColor(self.base_color)



    def submitCode(self):

        text = self.text_edit.toPlainText()
        text_list = text.split("\n")
        for expr in text_list:
            exec(expr)


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    GUI = PyDialog(None)
    sys.exit(app.exec_())