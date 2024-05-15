import sys
import cv2
import numpy as np

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from utils.utils import read_json_skeleton
from animator.animator_v2 import Animator, VideoAnimator

from label3d_v2 import Label3D

from scipy.io import loadmat


class MainWindow(QMainWindow):
    def __init__(self, laoder):
        super().__init__()
        self.initUI()

    def initUI(self, ):
        self.setWindowTitle("3D Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.label3d = Label3D()
        layout.addWidget(self.label3d)
        
        self.setLayout(layout)

        self.menuBar = self.menuBar()
        helpMenu = self.menuBar.addMenu("Help")
        helpMenu.addAction("Operations")
        


if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()