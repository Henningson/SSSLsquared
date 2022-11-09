
import os, sys, glob
from functools import partial
from PyQt5 import QtCore, QtSql
from PyQt5.QtGui import QPixmap, QTransform, QImage
from os.path import expanduser, dirname
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsView, QMenu, QLabel, QFileDialog, QHBoxLayout, QGridLayout
import skvideo.io
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QGraphicsPolygonItem, QGraphicsEllipseItem
from PyQt5.QtCore import QSize, pyqtSignal, QPointF
from PyQt5.QtGui import QIcon, QPen, QBrush, QPolygonF, QColor
import torch
from model import SPLSS, LSQLocalization
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import Visualizer

class MainWindow(QMainWindow):  
    def __init__(self, video_path):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(800, 600))            

        self.zoom = 1.0

        self.setWindowTitle("Structured Light Labelling")
        self.mainwidget = ButtonGrid(18)
        ## Set the central widget of the Window.
        self.setCentralWidget(self.mainwidget)
    






class ButtonGrid(QWidget):
    buttonSignal = pyqtSignal(int, int)

    def __init__(self, grid_size, parent=None):
        super(ButtonGrid, self).__init__()

        self.setLayout(QGridLayout())
        self.buttons = [ [None]*grid_size for i in range(grid_size)]

        for y in range(grid_size):
            for x in range(grid_size):
                button = GridButton(x, y)
                button.clicked.connect(self.clicked_button)
                self.layout().addWidget(button, x, y, 1, 1)
                self.buttons[x][y] = button

    def clicked_button(self, x, y):
        print(x)
        print(y)


class GridButton(QPushButton):
    buttonSignal = pyqtSignal(int, int)

    def __init__(self, x, y, parent=None):
        super(GridButton, self).__init__("")
        self.x = x
        self.y = y
        self.clicked.emit(x, y)




if __name__ == '__main__':

    app = QApplication(sys.argv)
    shufti = MainWindow("data/Human_P181133_top_Broc5_4001-4200.avi")
    shufti.show()
    sys.exit(app.exec_())