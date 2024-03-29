
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
        self.start_save_index = 0
        self.stop_save_index = video.shape[0]

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
        elif event.key() == Qt.Key_J:
            self.start_save_index = self.current_img_index
            print("Set starting index for video saving to: {0}".format(self.start_save_index))
        elif event.key() == Qt.Key_K:
            self.stop_save_index = self.current_img_index
            print("Set stopping index for video saving to: {0}".format(self.stop_save_index))
        elif event.key() == Qt.Key_L:
            self.save_videos()
        
        print(self.current_img_index)

    def prevImage(self):
        prev_index = self.current_img_index - 1 if self.current_img_index > 0 else self.current_img_index
        self.current_img_index = prev_index
        self._redraw()

    def nextImage(self):
        next_index = self.current_img_index + 1 if self.current_img_index < self.video.shape[0] - 1 else self.current_img_index
        self.current_img_index = next_index
        self._redraw()

    def saveImages(self, gt_image_name = "input.png", seg_name = "Seg.png", gtseg_name = "GTSeg.png", error_name = "Error.png"):
        self.saveImage(self.video_view, gt_image_name)
        self.saveImage(self.seg_view, seg_name)
        self.saveImage(self.gtseg_view, gtseg_name)
        self.saveImage(self.error_view, error_name)

    def save_videos(self, base_folder = ""):
        if self.start_save_index >= self.stop_save_index:
            print("Start index ({0}) greater than stop index ({1})".format(self.start_save_index, self.stop_save_index))
            return

        path_gt = "0_GT" if base_folder == "" else os.path.join(base_folder, path_gt)
        path_point = "0_Points" if base_folder == "" else os.path.join(base_folder, path_gt)
        path_gtpoint = "0_GTPoints" if base_folder == "" else os.path.join(base_folder, path_gt)
        path_seg = "0_Seg" if base_folder == "" else os.path.join(base_folder, path_gt)
        path_gtseg = "0_GTSeg" if base_folder == "" else os.path.join(base_folder, path_gt)
        path_error = "0_Error" if base_folder == "" else os.path.join(base_folder, path_gt)

        if base_folder != "":
            os.makedirs(base_folder, exist_ok=True)

            path_gt = os.path.join(base_folder, path_gt)
            path_point = os.path.join(base_folder, path_point)
            path_gtpoint = os.path.join(base_folder, path_gtpoint)
            path_seg = os.path.join(base_folder, path_seg)
            path_gtseg = os.path.join(base_folder, path_gtseg)
            path_error = os.path.join(base_folder, path_error)

        os.makedirs(path_gt, exist_ok=True)
        os.makedirs(path_gtpoint, exist_ok=True)
        os.makedirs(path_point, exist_ok=True)
        os.makedirs(path_seg, exist_ok=True)
        os.makedirs(path_gtseg, exist_ok=True)
        os.makedirs(path_error, exist_ok=True)

        tmp_curr_img_index = self.current_img_index
        tmp_show_gt_points = self.show_gt_points
        tmp_show_pred_points = self.show_pred_points

        self.current_img_index = self.start_save_index
        self.show_gt_points = False
        self.show_pred_points = False
        self._redraw()
        for i in range(self.start_save_index, self.stop_save_index):
            self.saveImages("{0}/{1:05d}.png".format(path_gt,  i), 
                            "{0}/{1:05d}.png".format(path_seg,  i), 
                            "{0}/{1:05d}.png".format(path_gtseg,  i), 
                            "{0}/{1:05d}.png".format(path_error,  i))
            
            self.show_pred_points = True
            self._redraw()
            self.saveImage(self.video_view, "{0}/{1:05d}.png".format(path_point,  i))
            self.show_pred_points = False

            self.show_gt_points = True
            self._redraw()
            self.saveImage(self.video_view, "{0}/{1:05d}.png".format(path_gtpoint,  i))
            self.show_gt_points = False

            self.nextImage()

        self.current_img_index = tmp_curr_img_index
        self.show_gt_points = tmp_show_gt_points
        self.show_pred_points = tmp_show_pred_points
        self._redraw()
        print("Done!")


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