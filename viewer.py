#!/usr/bin/env python3

'''
"Segment Images"
"Generate Points"
"Remove Points"
"Add Points"
"Remove Bounding Boxes"
"Compute Correspondences"
"Show Bounding Boxes"
"Show Pointlabels"
'''

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
from models.UNet import SPLSS, LSQLocalization
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import Visualizer
import utils

from Camera import Camera
from Laser import Laser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cvImgToQT(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    return QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)


class ZoomableView(QGraphicsView):
    pointSignal = pyqtSignal(QPointF)
    removePointSignal = pyqtSignal(QPointF)

    def __init__(self, parent=None):
        super(ZoomableView, self).__init__(parent)
        self.pointRemovalMode = False

    def togglePointRemovalMode(self):
        self.pointRemovalMode = not self.pointRemovalMode

    def wheelEvent(self, event):
        mouse = event.angleDelta().y()/120
        if mouse > 0:
            shufti.zoomIn()
        elif mouse < 0:
            shufti.zoomOut()

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.addAction('Zoom in               +, E', shufti.zoomIn)
        menu.addAction('Zoom out              -, D', shufti.zoomOut)
        menu.addAction('Toggle fullscreen     F11',  shufti.toggleFullscreen)
        menu.addAction('Next image            D',    shufti.nextImage)
        menu.addAction('Previous image        D',    shufti.prevImage)
        menu.addAction('Forward n frames      E',    shufti.nextImage)
        menu.addAction('Backward n frames     A',    shufti.prevImage)
        menu.addAction('Fit view              F',    shufti.fitView)
        menu.addAction('Reset zoom            1',    shufti.zoomReset)
        menu.addAction('Validate Segmentation Enter',shufti.validateSegmentation)
        menu.addAction('Quit                  ESC',  shufti.close)
        menu.exec_(event.globalPos())

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        point = self.mapToScene(self.mapFromGlobal(event.globalPos()))
        
        print(type(self.scene().itemAt(point, QTransform())))

        if not self.pointRemovalMode:
            self.pointSignal.emit(point)
        elif type(self.scene().itemAt(point, QTransform())) == QGraphicsEllipseItem:
            self.scene().removeItem(self.scene().itemAt(point, QTransform()))
            self.removePointSignal.emit(point)

class MainWindow(QMainWindow):    
    def createAction(self, text, shortcut, statustip, function):
        action = QAction(text, self)        
        action.setShortcut(shortcut)
        action.setStatusTip(statustip)
        action.triggered.connect(function)
        return action

    def __init__(self, video_path, calibration_path, laser_path):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(800, 600))            

        # Create new action
        newAction = self.createAction('&New', 'Ctrl+N', 'New Project', self.newCall)
        openAction = self.createAction('&Open', 'Ctrl+O', 'Open Project', self.openCall)
        saveAction = self.createAction('&Save', 'Ctrl+S', 'Save Project', self.saveCall)
        exitAction = self.createAction('&Exit', 'Ctrl+Q', 'Exit program', self.exitCall)

        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(newAction)
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)

        self.zoom = 1.0

        self.setWindowTitle("Structured Light Labelling")

        self.video = skvideo.io.vread(video_path)
        self.current_img_index = 0

        self.img = QPixmap(cvImgToQT(self.video[self.current_img_index]))
        self.scene = QGraphicsScene()
        self.scene.addPixmap(self.img)
        self.view = ZoomableView(self)
        self.view.setScene(self.scene)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.pointsize = 10

        self.mainwidget = QWidget()
        self.menu_widget = LeftMenuWidget()
        self.base_layout = QHBoxLayout(self.menu_widget)
        self.base_layout.addWidget(self.menu_widget)
        self.base_layout.addWidget(self.view)
        self.mainwidget.setLayout(self.base_layout)
        ## Set the central widget of the Window.
        self.setCentralWidget(self.mainwidget)

        self.segmentation_mode = False
        self.pointAddMode = False
        self.removeBoundingBoxMode = False
        self.boundingBoxMode = False

        self.menu_widget.button_dict["Segment Images"].clicked.connect(self.toggleSegmentation)
        self.menu_widget.button_dict["Generate Points"].clicked.connect(self.generatePoints)
        self.menu_widget.button_dict["Remove Points"].clicked.connect(self.view.togglePointRemovalMode)
        self.menu_widget.button_dict["Add Points"].clicked.connect(self.togglePointAddMode)
        self.menu_widget.buttonGrid.buttonSignal.connect(self.getGridButtonClicked)
        self.menu_widget.button_dict["Compute Correspondences"].clicked.connect(self.computeCorrespondences)
        self.menu_widget.button_dict["Remove Bounding Boxes"].clicked.connect(self.toggleRemoveBoundingBoxMode)
        self.menu_widget.button_dict["Show Bounding Boxes"].clicked.connect(self.toggleShowBoundingBoxes)
        self.menu_widget.button_dict["Show Pointlabels"].clicked.connect(self.toggleShowLabels)
        self.menu_widget.button_dict["Show Epipolar Lines"].clicked.connect(self.toggleShowEpipolarLines)

        self.points2d = []
        self.labels = []
        self.polygon_points = []
        self.segmentationPoints = []
        self.segmentation = None

        self.view.pointSignal.connect(self.segmentationPointAdded)
        self.view.removePointSignal.connect(self.removePoint)
        self.view.pointSignal.connect(self.addPoint)
        self.view.pointSignal.connect(self.generateBoundingBox)
        self.view.pointSignal.connect(self.removeBoundingBox)

        self.menu_widget.edit_dict["Min Distance"].editingFinished.connect(self.generateEpipolarLines)
        self.menu_widget.edit_dict["Max Distance"].editingFinished.connect(self.generateEpipolarLines)

        self.showBoundingBoxes = True
        self.showLabels = True
        self.showEpipolarLines = True

        self.boundingBoxTuple = [None, None]
        self.boundingBoxes = []
        self.boundingBoxIndex = 0
        self.currentButton = [0, 0]

        self.epipolarLines = None

        self.polygonhandle = None

        self.camera = Camera(calibration_path)
        self.laser = Laser(laser_path, "JSON")

        self.pointArray = np.zeros([self.video.shape[0], 18, 18, 2], dtype=np.float32)
        self.pointArray[:] = np.nan

        self.generateEpipolarLines()
        self.redraw()


    def generateEpipolarLines(self):
        self.epipolarLines = []

        minDistance = self.menu_widget.getValueFromEdit("Min Distance")
        maxDistance = self.menu_widget.getValueFromEdit("Max Distance")
        minPoints = self.generatePointsAt(minDistance)
        maxPoints = self.generatePointsAt(maxDistance)

        for pointA, pointB in zip(minPoints.tolist(), maxPoints.tolist()):
            self.epipolarLines.append(QLineF(pointA[0], pointA[1], pointB[0], pointB[1]))

        self.redraw()

    def generatePointsAt(self, distance):
        return self.camera.project(self.laser.origin().reshape(-1, 3) + self.laser.rays() * distance)

    def toggleRemoveBoundingBoxMode(self):
        self.removeBoundingBoxMode = not self.removeBoundingBoxMode

        if self.removeBoundingBoxMode:
            self.menu_widget.disableEverythingExcept("Remove Bounding Boxes")
        else:
            self.menu_widget.enableEverything()

    def toggleShowBoundingBoxes(self):
        self.showBoundingBoxes = not self.showBoundingBoxes
        self.redraw()

    def computeCorrespondencesFromEpipolars(self, threshold=3.0):
        self.labels = []

        for perFramePoints in self.points2d:
            self.labels.append([])
            for line_index, line in enumerate(self.epipolarLines):
                for i in range(perFramePoints.shape[0]):
                    if utils.pointLineSegmentDistance(line[0], line[0], perFramePoints[i]) < threshold:
                        x, y = self.laser.getXYfromN(line_index)
                        self.pointArray[i, x, y, 1] = perFramePoints[i, 1]
                        self.pointArray[i, x, y, 0] = perFramePoints[i, 1]
                        self.labels[-1].append([x, y])

    def computeCorrespondences(self):

        # Go through every frame
        for perFramePoints in self.points2d:
            self.labels.append([])

            # For every bounding box
            for boundingbox in self.boundingBoxes:
                # And for every point inside this frame
                for i in range(perFramePoints.shape[0]):
                    # Check if the bounding box contains the point (point is ordered in opencv Y, X fashion)
                    if boundingbox.contains(perFramePoints[i, 1], perFramePoints[i, 0]):
                        self.pointArray[i, boundingbox.x, boundingbox.y, 1] = perFramePoints[i, 1]
                        self.pointArray[i, boundingbox.x, boundingbox.y, 0] = perFramePoints[i, 0]
                        self.labels[-1].append([boundingbox.x, boundingbox.y])
    
    def toggleShowEpipolarLines(self):
        self.showEpipolarLines = not self.showEpipolarLines
        self.redraw()

    def toggleShowLabels(self):
        self.showLabels = not self.showLabels

    def toggleBoundingBoxMode(self):
        self.boundingBoxMode = not self.boundingBoxMode

        if self.boundingBoxMode:
            self.menu_widget.disableEverythingExcept("")
        else:
            self.menu_widget.enableEverything()

    QtCore.pyqtSlot(QPointF)
    def generateBoundingBox(self, point):
        if not self.boundingBoxMode:
            return

        if self.boundingBoxIndex == 0:
            self.boundingBoxTuple[0] = point
            self.boundingBoxIndex = 1
            return

        if self.boundingBoxIndex == 1:
            self.boundingBoxTuple[1] = point
            self.boundingBoxIndex = 0

        x = self.currentButton[0]
        y = self.currentButton[1]

        self.boundingBoxes.append(IdentifiableRectItem(self.boundingBoxTuple[0].toPoint(), self.boundingBoxTuple[1].toPoint(), x, y))
        self.toggleBoundingBoxMode()
        self.redraw()

    @QtCore.pyqtSlot(QPointF)
    def removeBoundingBox(self, point):
        if not self.removeBoundingBoxMode:
            return

        for boundingBox in self.boundingBoxes:
            item = self.scene.itemAt(point, QTransform())
            if type(item) == QGraphicsRectItem and boundingBox.isEqualsToQGraphicsRect(item):
                self.boundingBoxes.remove(boundingBox)

        self.redraw()

    @QtCore.pyqtSlot(int, int)
    def getGridButtonClicked(self, x, y):
        print("Setting laser point {} {}".format(x, y))
        self.currentButton = (x, y)
        self.boundingBoxIndex = 0
        self.toggleBoundingBoxMode()
    
    @QtCore.pyqtSlot(QPointF)
    def segmentationPointAdded(self, point):
        if not self.segmentation_mode:
            return

        if point.x() < 0 or point.y() < 0:
            return

        if point.y() > self.video[0].shape[0] - 1 or point.x() > self.video[0].shape[1] - 1:
            return

        self.segmentationPoints.append(point)

        if len(self.segmentationPoints) > 0:
            self.drawSegmentation()

    @QtCore.pyqtSlot(QPointF)
    def removePoint(self, clicked_point):
        points = self.points2d[self.current_img_index]
        clicked_point = np.array([clicked_point.y(), clicked_point.x()]).reshape(-1, 2)
        minimum = np.sqrt(np.sum((points - clicked_point)**2, axis=1)).argmin()
        self.points2d[self.current_img_index] = np.delete(points, minimum, axis=0)

    def togglePointAddMode(self):
        self.pointAddMode = not self.pointAddMode

        if self.pointAddMode:
            self.menu_widget.disableEverythingExcept("Add Points")
        else:
            self.menu_widget.enableEverything()

    @QtCore.pyqtSlot(QPointF)
    def addPoint(self, clicked_point):
        if not self.pointAddMode:
            return

        point = np.array([clicked_point.y(), clicked_point.x()])
        self.points2d[self.current_img_index] = np.concatenate([self.points2d[self.current_img_index], point.reshape(-1, 2)])
        self.redraw()

    def toggleSegmentation(self):
        self.segmentation_mode = not self.segmentation_mode

        if self.segmentation_mode:
            self.menu_widget.disableEverythingExcept("Segment Images")
        else:
            self.menu_widget.enableEverything()


        if len(self.segmentationPoints) > 2:
            self.generateCVSegmentation()

    def drawSegmentation(self):
        if self.polygonhandle:
            self.scene.removeItem(self.polygonhandle)

        self.polygonhandle = self.scene.addPolygon(QPolygonF(self.segmentationPoints), QPen(QColor(128, 128, 255, 128)), QBrush(QColor(128, 128, 255, 128)))

    def drawBoundingBoxes(self):
        for bounding_box in self.boundingBoxes:
            self.scene.addRect(bounding_box, QPen(QColor(128, 255, 128, 128)), QBrush(QColor(128, 255, 128, 128)))

    def drawLabels(self):
        try:
            for label, pos in zip(self.labels[self.current_img_index], self.points2d[self.current_img_index].tolist()):
                if np.isnan(np.array(pos)).any():
                    continue

                text = self.scene.addText("{},{}".format(label[0], label[1]))
                text.setPos(pos[1], pos[0])
                text.setDefaultTextColor(QColor(255, 128, 128, 255))
        except:
            return

    def drawEpipolarLines(self):
        for line in self.epipolarLines:
            self.scene.addLine(line, QPen(QColor(255, 255, 255, 255)))

    def generateCVSegmentation(self):
        base = np.zeros((self.video[0].shape[0], self.video[0].shape[1]), dtype=np.uint8)
        np_points = np.array([[point.x(), point.y()] for point in self.segmentationPoints], dtype=np.int32)
        test = cv2.drawContours(base, [np_points], 0, thickness=-1, color=1)
        self.segmentation = test

    def generatePoints(self):
        if self.segmentation is None:
            print("Please generate a segmentation")
            return

        model = SPLSS(in_channels=1, out_channels=2, state_dict=torch.load("reinhardnetv2.pth.tar")).to(DEVICE)
        loc = LSQLocalization(local_maxima_window=25)
        transform = A.Compose([A.Resize(height=1200, width=800), A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0,), ToTensorV2(), ])
        segment_transform = A.Compose([A.Resize(height=1200, width=800), ToTensorV2(),])

        with torch.no_grad():
            for count in tqdm(range(self.video.shape[0])):
                image = cv2.cvtColor(self.video[count], cv2.COLOR_RGB2GRAY)
                image = transform(image=image)["image"].to(DEVICE)
                segment = segment_transform(image=self.segmentation)["image"].to(DEVICE)

                with torch.cuda.amp.autocast():
                    prediction = model(image.unsqueeze(0)).softmax(dim=1)
                    _, mean, _ = loc.test(prediction, segmentation=segment)

                    means = mean[0].detach().cpu().numpy()

                    self.points2d.append(means[~np.isnan(means).any(axis=1)])

        self.redraw()

    def draw_point_estimates(self):
        if not len(self.points2d) > 0:
            return

        print("Num of Points2D at Frame {0}: {1}".format(self.current_img_index, self.points2d[self.current_img_index].shape[0]))

        for point in self.points2d[self.current_img_index].tolist():
            self.scene.addEllipse(point[1] - self.pointsize//2, point[0] - self.pointsize//2, self.pointsize, self.pointsize, QPen(QColor(128, 128, 255, 128)), QBrush(QColor(128, 128, 255, 128)))

    def closeEvent(self, event):
        self.winState()

    def openCall(self):
        self.camera_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Camera Calibration file', '', "Camera Calibration Files (*.json *.mat)")
        self.laser_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Laser Calibration file', '', "Laser Calibration Files (*.json *.mat)")
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)

    def newCall(self):
        self.camera_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Camera Calibration file', '', "Camera Calibration Files (*.json *.mat)")
        self.laser_calib_path, _ = QFileDialog.getOpenFileName(self, 'Open Laser Calibration file', '', "Laser Calibration Files (*.json *.mat)")
        self.video_path, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', "Video Files (*.avi *.mp4 *.mkv *.AVI *.MP4)")
        self.fileOpenedSignal.emit(self.camera_calib_path, self.laser_calib_path, self.video_path)

    def exitCall(self):
        print('Exit app')

    def saveCall(self):
        print('Save')
    
    def toggleFullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F11:
            self.toggleFullscreen()
        elif event.key() == QtCore.Qt.Key_Equal or event.key() == QtCore.Qt.Key_W:
            self.zoomIn()
        elif event.key() == QtCore.Qt.Key_Minus or event.key() == QtCore.Qt.Key_S:
            self.zoomOut()
        elif event.key() == QtCore.Qt.Key_1:
            self.zoomReset()
        elif event.key() == QtCore.Qt.Key_F:
            self.fitView()
        elif event.key() == QtCore.Qt.Key_D:
            self.nextImage()
        elif event.key() == QtCore.Qt.Key_A:
            self.prevImage()
        elif event.key() == QtCore.Qt.Key_Q:
            self.close()

    def zoomIn(self):
        self.zoom *= 1.1
        self.updateView()

    def zoomOut(self):
        self.zoom /= 1.1
        self.updateView()

    def zoomReset(self):
        self.zoom = 1
        self.updateView()

    def fitView(self):
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.zoom = self.view.transform().m11()

    def updateView(self):
        self.view.setTransform(QTransform().scale(self.zoom, self.zoom))

    def winState(self):
        self.winsizex = self.geometry().width()
        self.winsizey = self.geometry().height()
        self.vscroll = self.view.verticalScrollBar().value()
        self.hscroll = self.view.horizontalScrollBar().value()
        self.winposx = self.pos().x()
        self.winposy = self.pos().y()

    def prevImage(self):
        prev_index = self.current_img_index - 1 if self.current_img_index > 0 else self.current_img_index
        self.current_img_index = prev_index

        self.setImage(self.video[self.current_img_index])

    def nextImage(self):
        next_index = self.current_img_index + 1 if self.current_img_index < self.video.shape[0] - 1 else self.current_img_index
        self.current_img_index = next_index

        self.setImage(self.video[self.current_img_index])

    def setImage(self, image):
        self.scene.clear()
        self.img = QPixmap(cvImgToQT(image))
        self.scene.addPixmap(self.img)
        self.draw_point_estimates()

        if self.showBoundingBoxes:
            self.drawBoundingBoxes()

        if self.showLabels:
            self.drawLabels()

        if self.showEpipolarLines:
            self.drawEpipolarLines()

    def redraw(self):
        self.scene.clear()
        self.img = QPixmap(cvImgToQT(self.video[self.current_img_index]))
        self.scene.addPixmap(self.img)
        self.draw_point_estimates()
        
        if self.showBoundingBoxes:
            self.drawBoundingBoxes()

        if self.showLabels:
            self.drawLabels()

        if self.showEpipolarLines:
            self.drawEpipolarLines()

    def resetScroll(self):
        self.view.verticalScrollBar().setValue(0)
        self.view.horizontalScrollBar().setValue(0)

    def getScreenRes(self):
        self.screen_res = app.desktop().availableGeometry()
        self.screenw = self.screen_res.width()
        self.screenh = self.screen_res.height()

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QPushButton
from gridbuttonclick import ButtonGrid

class LeftMenuWidget(QWidget):
    def __init__(self, parent=None):
        super(LeftMenuWidget, self).__init__()
        #self.setStyle(QFrame.Panel | QFrame.Raised)
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(Qt.AlignTop)
        self.button_dict = {}
        self.edit_dict = {}
        self.buttonGrid = ButtonGrid()

        self.addButton("Segment Images")
        self.addButton("Generate Points")
        self.addButton("Remove Points")
        self.addButton("Add Points")
        self.addLineEdit("Min Distance", 80.0)
        self.addLineEdit("Max Distance", 100.0)
        self.addButton("Show Epipolar Lines")
        self.addButton("Show Bounding Boxes")
        self.addButton("Show Pointlabels")
        self.layout().addWidget(self.buttonGrid)
        self.addButton("Remove Bounding Boxes")
        self.addButton("Compute Correspondences")

    def getValueFromEdit(self, key):
        return float(self.edit_dict[key].text())

    def disableEverythingExcept(self, button_key):
        for key in self.button_dict.keys():
            if button_key != key:
                self.button_dict[key].setEnabled(False)

    def enableEverything(self):
        for button in self.button_dict.values():
            button.setEnabled(True)

    def addLineEdit(self, label, defaultvalue):
        widget = QWidget()
        widget.setLayout(QFormLayout())
        lineedit = QLineEdit(str(defaultvalue), widget)
        widget.layout().addRow(QLabel(label, widget), lineedit)
        self.edit_dict[label] = lineedit
        self.layout().addWidget(widget)

    def addButton(self, label):
        button = QPushButton(label)
        self.layout().addWidget(button)
        self.button_dict[label] = button


class IdentifiableRectItem(QRectF):
    def __init__(self, pointa, pointb, x, y):
        super(IdentifiableRectItem, self).__init__(pointa, pointb)
        self.x = x
        self.y = y

    def isEquals(self, x, y):
        return self.x == x and self.y == y

    def isEqualsToQGraphicsRect(self, qGraphicsRect):
        return self == qGraphicsRect.rect()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    shufti = MainWindow("data/Human_P181133_top_Broc5_4001-4200.avi", "data/Calib_Hemi_P181133.mat", "data/laser_calibration.json")
    shufti.show()
    sys.exit(app.exec_())