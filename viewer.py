
import os
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)

from PyQt5.QtGui import QPixmap, QTransform, QImage, qRgb
from PyQt5.QtWidgets import QMainWindow, QGraphicsScene, QGraphicsView, QHBoxLayout, QWidget
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPen, QBrush, QColor

import numpy as np


def cvImgToQT(image):
    height, width, channel = image.shape
    bytesPerLine = 3 * width
    return QImage(np.require(image, np.uint8, 'C'), width, height, bytesPerLine, QImage.Format_RGB888)


all_views = []
class ZoomableView(QGraphicsView):
    def __init__(self, parent=None):
        super(ZoomableView, self).__init__(parent)
        self.zoom = 1.0
        all_views.append(self)

    def wheelEvent(self, event):
        mouse = event.angleDelta().y()/120

        for view in all_views:
            if mouse > 0:
                view.zoomIn()
            else:
                view.zoomOut()
            
    def zoomIn(self):
        self.zoom *= 1.1
        self.updateView()

    def zoomOut(self):
        self.zoom /= 1.1
        self.updateView()

    def zoomReset(self):
        self.zoom = 1
        self.updateView()

    def updateView(self):
        self.setTransform(QTransform().scale(self.zoom, self.zoom))

    def mousePressEvent(self, event):
        for view in all_views:
            super(type(view), view).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        for view in all_views:
            super(type(view), view).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        for view in all_views:
            super(type(view), view).mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def initScene(self):
        scene = QGraphicsScene()
        view = ZoomableView(self)
        view.setScene(scene)
        view.setDragMode(QGraphicsView.ScrollHandDrag)
        view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        return scene, view

    def __init__(self, video, segmentations, gt_segmentations, errors, pred_points, gt_points):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(800, 600))
        self.setWindowTitle("Inference Viewer")

        self.video = video
        self.segmentations = segmentations
        self.gt_segmentations = gt_segmentations
        self.errors = errors

        self.gt_points = gt_points
        self.pred_points = pred_points
        
        self.current_img_index = 0

        self.video_scene, self.video_view = self.initScene()
        self.seg_scene, self.seg_view = self.initScene()
        self.gtseg_scene, self.gtseg_view = self.initScene()
        self.error_scene, self.error_view = self.initScene()
        
        mainwidget = QWidget()
        mainwidget.setLayout(QHBoxLayout())
        mainwidget.layout().addWidget(self.video_view)
        mainwidget.layout().addWidget(self.seg_view)
        mainwidget.layout().addWidget(self.gtseg_view)
        mainwidget.layout().addWidget(self.error_view)
        self.setCentralWidget(mainwidget)

        self.pen = QPen(QColor(0, 255, 0, 255))
        self.brush = QBrush(QColor(0, 255, 0, 128))

        self.gt_pen = QPen(QColor(0, 255, 255, 255))
        self.gt_brush = QBrush(QColor(0, 255, 255, 128))

        self.show_pred_points = True
        self.show_gt_points = True

        self.pointsize = 5

        self._redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.nextImage()
        elif event.key() == Qt.Key_A:
            self.prevImage()
        elif event.key() == Qt.Key_W:
            self.show_pred_points = not self.show_pred_points
            print("Drawing Prediction Points: {}".format(self.show_pred_points))
            self._redraw()
        elif event.key() == Qt.Key_S:
            self.show_gt_points = not self.show_gt_points
            print("Drawing GT Points: {}".format(self.show_gt_points))
            self._redraw()
        elif event.key() == Qt.Key_Space:
             self.seg_view.zoomReset()
             self.video_view.zoomReset()
        elif event.key() == Qt.Key_P:
            self.saveImages()
        
        print(self.current_img_index)

    def prevImage(self):
        prev_index = self.current_img_index - 1 if self.current_img_index > 0 else self.current_img_index
        self.current_img_index = prev_index
        self._redraw()

    def nextImage(self):
        next_index = self.current_img_index + 1 if self.current_img_index < self.video.shape[0] - 1 else self.current_img_index
        self.current_img_index = next_index
        self._redraw()

    def saveImages(self):
        self.saveImage(self.video_view, "GT_Image.png")
        self.saveImage(self.seg_view, "Seg_image.png")
        self.saveImage(self.gtseg_view, "GTSeg_Image.png")
        self.saveImage(self.error_view, "Error_Image.png")

    def saveImage(self, view, filename):
        img = view.grab()
        img.save(filename)

    def _redraw(self):
        self.redraw(self.seg_scene, self.segmentations)
        self.redraw(self.gtseg_scene, self.gt_segmentations)
        self.redraw(self.error_scene, self.errors)
        self.redraw(self.video_scene, self.video)

        if self.show_pred_points:
            self.drawPoints(self.video_scene, self.pred_points, self.pen, self.brush)
        
        if self.show_gt_points:
            self.drawPoints(self.video_scene, self.gt_points, self.gt_pen, self.gt_brush)

    def redraw(self, scene, images):
        scene.clear()
        img = QPixmap(cvImgToQT(images[self.current_img_index]))
        scene.addPixmap(img)

    def drawPoints(self, scene, points, pen, brush):
        for point in points[self.current_img_index].tolist():
            xpos = point[1] - self.pointsize//2
            ypos = point[0] - self.pointsize//2
            scene.addEllipse(xpos, ypos, self.pointsize, self.pointsize, pen, brush)