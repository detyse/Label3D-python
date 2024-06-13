import sys
import cv2
import numpy as np
from scipy.io import loadmat

from PySide6.QtWidgets import *
from PySide6.QtCore import *

from utils.utils import read_json_skeleton
from animator.animator import VideoAnimator

from label3d import Label3D


PARAMS_PATH = r""  # here put the path of params, in the format of label3d.  eg. r"D:\Projects\Label3D-python\Data\xxx.mat"
cam_params = loadmat(PARAMS_PATH)['camParams']

# get camera parameters here
load_camParams = []
for i in range(len(cam_params)):
    load_camParams.append(cam_params[i][0])

VIDEO_PATH = [ #r"D:\Projects\Label3D-python\Data\xxx.mp4",     # here put the video path list, sort in views
             ...
             ]

# put skeleton here, should store in json format, transformer in utils
SKELETON_PATH = r"D:\Projects\Label3D-python\utils\skeletons\test.json"

# main function here
if __name__ == "__main__":
    app = QApplication([])
    window = Label3D(camParams=load_camParams, videos=VIDEO_PATH, skeleton_path=SKELETON_PATH, loader=None)
    window.show()
    sys.exit(app.exec())
