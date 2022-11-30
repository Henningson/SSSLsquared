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

    def __init__(self, video, segmentations, gt_segmentations, errors, points):
        QMainWindow.__init__(self)
        self.setMinimumSize(QSize(800, 600))
        self.setWindowTitle("Inference Viewer")

        self.video = video
        self.segmentations = segmentations
        self.gt_segmentations = gt_segmentations
        self.errors = errors
        self.points = points
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

        self.pen = QPen(QColor(128, 255, 128, 255))
        self.brush = QBrush(QColor(128, 255, 128, 128))
        self.pointsize = 5

        self._redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_D:
            self.nextImage()
        elif event.key() == Qt.Key_A:
            self.prevImage()
        elif event.key() == Qt.Key_Space:
             self.seg_view.zoomReset()
             self.video_view.zoomReset()

    def prevImage(self):
        prev_index = self.current_img_index - 1 if self.current_img_index > 0 else self.current_img_index
        self.current_img_index = prev_index
        self._redraw()

    def nextImage(self):
        next_index = self.current_img_index + 1 if self.current_img_index < self.video.shape[0] - 1 else self.current_img_index
        self.current_img_index = next_index
        self._redraw()

    def _redraw(self):
        self.redraw(self.seg_scene, self.segmentations)
        self.redraw(self.gtseg_scene, self.gt_segmentations)
        self.redraw(self.error_scene, self.errors)
        self.redraw(self.video_scene, self.video, draw_points=True)

    def redraw(self, scene, images, draw_points = False):
        scene.clear()
        img = QPixmap(cvImgToQT(images[self.current_img_index]))
        scene.addPixmap(img)

        if draw_points:
            self.draw_points(scene)

    def draw_points(self, scene):
        for point in self.points[self.current_img_index].tolist():
            xpos = point[1] - self.pointsize//2
            ypos = point[0] - self.pointsize//2
            scene.addEllipse(xpos, ypos, self.pointsize, self.pointsize, self.pen, self.brush)