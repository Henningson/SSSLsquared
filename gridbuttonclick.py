
import os, sys, glob
from functools import partial
from PyQt5 import QtCore, QtSql
from PyQt5.QtGui import QPixmap, QTransform, QImage
from os.path import expanduser, dirname
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QGraphicsView, QMenu, QLabel, QFileDialog, QFormLayout, QHBoxLayout, QGraphicsRectItem, QGridLayout
import skvideo.io
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QGraphicsPolygonItem, QGraphicsEllipseItem
from PyQt5.QtCore import QSize, pyqtSignal, QPointF, QRectF, QLineF
from PyQt5.QtGui import QIcon, QPen, QBrush, QPolygonF, QColor
import torch
from model import SPLSS, LSQLocalization
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import Visualizer
from Camera import Camera
from Laser import Laser
from viewer import ZoomableView, cvImgToQT

class ZoomableView(QGraphicsView):

    def __init__(self, parent=None):
        super(ZoomableView, self).__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.zoom = 1.0

    def wheelEvent(self, event):
        mouse = event.angleDelta().y()/120
        if mouse > 0:
            self.zoomIn()
        elif mouse < 0:
            self.zoomOut()

    def updateView(self):
        self.setTransform(QTransform().scale(self.zoom, self.zoom))

    def zoomIn(self):
        self.zoom *= 1.1
        self.updateView()

    def zoomOut(self):
        self.zoom /= 1.1
        self.updateView()

class MainWindow(QMainWindow):  
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle("Structured Light Labelling")
        #self.mainwidget.buttonSignal.connect(self.getGridButtonClicked)
        ## Set the central widget of the Window.
        self.setMinimumSize(500, 500)

        calib_path = "data/Calib_Hemi_P181133.mat"
        self.camera = Camera(calib_path)
        self.laser = Laser("data/laser_calib.json", "JSON")

        self.img = QPixmap(cvImgToQT(np.ones((1200, 800, 3), dtype=np.uint8)*255))
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.img)
        self.view = ZoomableView(self)
        self.view.setScene(self.scene)

        self.lines = []
        self.generateEpipolarLines(40.0, 41.0)

        self.drawEpipolarLines()
        self.setCentralWidget(self.view)

    def generateEpipolarLines(self, minDistance, maxDistance):
        self.lines = []
        minPoints = self.generatePointsAt(minDistance)
        maxPoints = self.generatePointsAt(maxDistance)

        for pointA, pointB in zip(minPoints.tolist(), maxPoints.tolist()):
            self.lines.append(QLineF(pointA[0], pointA[1], pointB[0], pointB[1]))

    def generatePointsAt(self, distance):
        return self.camera.project(self.laser.origin().reshape(-1, 3) + self.laser.rays() * distance)

    def drawEpipolarLines(self):
        for line in self.lines:
            self.scene.addLine(line, QPen(QColor(0, 0, 0, 255)))

    @QtCore.pyqtSlot(int, int)
    def getGridButtonClicked(self, x, y):
        print("Clicked button {} {}".format(x, y))

class ButtonGrid(QWidget):
    buttonSignal = pyqtSignal(int, int)

    def __init__(self, grid_size=18, parent=None):
        super(ButtonGrid, self).__init__()

        self.setLayout(QGridLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        for y in range(grid_size):
            for x in range(grid_size):
                button = GridButton(x, y)
                button.clickedSignal.connect(self.clicked_button)
                self.layout().addWidget(button, y, x, 1, 1)

    @QtCore.pyqtSlot(int, int)
    def clicked_button(self, x, y):
        self.buttonSignal.emit(x, y)

class GridButton(QPushButton):
    clickedSignal = pyqtSignal(int, int)

    def __init__(self, x, y, parent=None):
        super(GridButton, self).__init__("")
        self.x = x
        self.y = y
        self.clicked.connect(self.on_clicked)
        self.setContentsMargins(0, 0, 0, 0)
        self.setFixedSize(QSize(25, 25))
        self.setStyleSheet("border: 1px solid #333333;")

    def on_clicked(self, bool):
        self.setStyleSheet("background-color : #33DD33")
        self.clickedSignal.emit(self.x, self.y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    shufti = MainWindow()
    shufti.show()
    sys.exit(app.exec_())