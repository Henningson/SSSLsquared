from PyQt5 import QtCore
from PyQt5.QtCore import QSize, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QGridLayout, QWidget, QPushButton
import sys

class MainWindow(QMainWindow):  
    def __init__(self):
        QMainWindow.__init__(self)

        self.setWindowTitle("Structured Light Labelling")
        self.mainwidget = ButtonGrid(18)
        self.mainwidget.buttonSignal.connect(self.getGridButtonClicked)
        ## Set the central widget of the Window.
        self.setCentralWidget(self.mainwidget)

    @QtCore.pyqtSlot(int, int)
    def getGridButtonClicked(self, x, y):
        print("Clicked button {} {}".format(x, y))

class ButtonGrid(QWidget):
    buttonSignal = pyqtSignal(int, int)

    def __init__(self, grid_size, parent=None):
        super(ButtonGrid, self).__init__()

        self.setLayout(QGridLayout())
        self.layout().setSpacing(0)
        self.layout().setContentsMargins(0, 0, 0, 0)

        for y in range(grid_size):
            for x in range(grid_size):
                button = GridButton(x, y)
                button.clickedSignal.connect(self.clicked_button)
                self.layout().addWidget(button, x, y, 1, 1)

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
        self.clickedSignal.emit(self.x, self.y)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    shufti = MainWindow()
    shufti.show()
    sys.exit(app.exec_())