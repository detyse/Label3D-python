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
    def __init__(self, camParams=None, videos=None, skeleton_path=None, loader=None, *args, **kwargs):
        super().__init__()
        self.camParams = camParams
        self.videos = videos
        self.skeleton_path = skeleton_path

        self.loader = loader
        self.args = args
        self.kwargs = kwargs
        self.initUI()


    def initUI(self, ):
        self.setWindowTitle("3D Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.label3d = Label3D(camParams=self.camParams, videos=self.videos, skeleton_path=self.skeleton_path, loader=self.loader, *self.args, **self.kwargs)
        layout.addWidget(self.label3d)
        
        self.setLayout(layout)
        self.setCentralWidget(self.label3d)

        self.menuBar = self.menuBar()
        helpMenu = self.menuBar.addMenu("Help")
        self.showManual = QAction("User Manual", self)
        helpMenu.addAction(self.showManual)

        self.showManual.triggered.connect(self.user_manual)
        
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready") 


    def user_manual(self, ):
        # show a message box to show the show the user manual
        self.manualBox = QMessageBox()
        self.manualBox.setWindowTitle("User Manual")
        self.manualBox.setText("This is a user manual")
        self.manualBox.exec()

        # layout = QVBoxLayout()
        # text = QLabel("This is a user manual")
        # layout.addWidget(text)

        # self.manualDialog.setLayout(layout)
        

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    # set params here
    PARAMS_PATH = r"d:\YL_Wang\_personal\temp\24-03-13\Cylinder_smallFOV_20230719_Label3D_dannce.mat"
    
    cam_params = loadmat(PARAMS_PATH)['params']

    frame2label = 100   # frame number when to label

    load_camParams = []
    for i in range(len(cam_params)):
        load_camParams.append(cam_params[i][0])

    # path 
    VIDEO_PATHS = [
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera1\1.avi",
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera2\1.avi",
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera3\1.avi",
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera4\1.avi",
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera5\1.avi",
                r"d:\YL_Wang\_personal\temp\24-03-13\videos\Camera6\1.avi"]

    SKELETON_PATH = r"D:\YL_Wang\Label3d\utils\skeletons\test.json"

    app = QApplication([])
    window = MainWindow(camParams=load_camParams, videos=VIDEO_PATHS, skeleton_path=SKELETON_PATH, loader=None)
    window.show()
    app.exec()

    
