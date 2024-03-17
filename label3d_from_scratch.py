import os
import sys
import time 

import numpy as np
import pandas as pd

import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from utils.utils import CameraParams, Loader, read_json_skeleton
from animator.animator import Animator, VideoAnimator, Keypoint3DAnimator


# inherit from Animator for all animator sync 
class Label3D(Animator):
    def __init__(self, camParams=None, videos=None, skeleton_path=None, loader=None, *args, **kwargs):
        super().__init__()
        '''
        can load from loader or from inputs
        the loader is a dataclass, storing all the input we need,
        to use loader, instantiate it first, #could read params from a mat file (not implemented yet)#

        '''
        assert (camParams is not None and videos is not None and skeleton_path is not None) or loader is not None, "Either loader or camParams, videos, and skeletons must be provided"
        if loader:
            self.camParams = self.loader.cam_params
            self.videos = self.loader.video_paths
            self.skeleton = self.loader.skeleton
        else:
            self.camParams = camParams
            self.videos = videos
            self.skeleton = read_json_skeleton(skeleton_path)
        assert len(self.camParams) == len(self.videos), "The number of cameras and videos must be the same"
        
        # 衍生属性，extended properties from inputs
        # self.cams_num = len(self.camParams)
        # self.skeletons_in_idx = self.get_skeletons_in_idx(self.skeletons)

        # 
        # self._skeleton = self.skeleton

        self._init_property()

        self.initGUI()

    def _init_property(self, ):
        '''
        get properties from inputs, also assert some unexpected inputs
        '''
        # check videos
        for video in self.videos:
            assert os.path.exists(video), f"Video {video} does not exist"

        self.view_num = len(self.camParams)
        assert len(self.camParams) == len(self.videos), "The number of cameras and videos must match"


        # get the skeleton properties
        self._joint_names = self.skeleton["joint_names"]
        self._joints_idx = self.skeleton["joints_idx"]        # the connection of joints, joints are indicated by index + 1
        self._color = self.skeleton["color"]              # the color of the joints

        # properties used as defults
        self.current_joint = None
        self.exist_markers = []
        self.joints2markers = {}


    def initGUI(self, ):        # do not ihnerit the initUI from Animator
        '''
        use a side bar in the main window to indicate the labeling joints
        '''
        self.setWindowTitle("3D Labeling Tool")

        # self.setGeometry(100, 100, 1300, 800)        # rec     # get the size accroding to the layout

        # TODO: check GUI line by line

        # # set the main widget for views of videos
        video_widget = QWidget()
        # video_widget.setGeometry(0, 0, 1000, 800)
        # # set a side bar for joints labeling
        side_bar = QWidget()

        # layout
        left_layout = QVBoxLayout()
        
        main_layout = QHBoxLayout()

        # # # side bar buttons, change the current joint, also change color to indicate weather it is labeled
        self.joint_buttons = {}

        for joint in self._joint_names:
            button = QRadioButton(joint)
            self.joint_buttons[joint] = button
            left_layout.addWidget(button)

        # # # also add a nothing button?
        nothing_button = QRadioButton("Nothing")
        self.joint_buttons["Nothing"] = nothing_button
        left_layout.addWidget(nothing_button)

        # # # set the layout
        side_bar.setLayout(left_layout)

        main_layout.addWidget(side_bar)

        # 可以直接尝试使用 Grid Layout
        # # get the pos and size of the video animators widget 
        # video_widget_geometry = [video_widget.x(), video_widget.y(), video_widget.width(), video_widget.height()]
        views_layout = QGridLayout()
        # # first get animators
        video_animators = self.get_animators()
        pos = self.get_views_position()

        for i, animator in enumerate(video_animators):
            print(type(animator))
            print(pos[i, 0], pos[i, 1])
            views_layout.addWidget(animator, pos[i, 0], pos[i, 1],)
            # animator.view.show()

        video_widget.setLayout(views_layout)
        video_widget.show()
        # # video widget done here

        main_layout.addWidget(video_widget)
        self.setLayout(main_layout)  

    def get_animators(self, ):
        '''
        return the animators for each video
        '''
        video_animators = []
        for video in self.videos:
            video_animators.append(VideoAnimator(video))

        return video_animators
    
    # 返回view的位置，第几行 第几列, 保存在一个数组中 对应 views， 
    def get_views_position(self, nViews=None):
        if nViews is None:
            nViews = self.view_num
        
        nRows = np.floor(np.sqrt(nViews))

        pos = np.zeros((nViews, 2))
        if nViews > 3:
            for i in range(nViews):
                row = np.floor(i / nRows)
                col = i % nRows
                pos[i, :] = [row, col]

        return pos
    
    ## video animator GUI ends here
    
    ## start the point labeling part
    def get_skeletons_in_idx(self, skeletons=None):
        '''
        sort the joint by joints idx,
        the joints is stored in a dict of dict. (no list to avoid errors)
        '''
        joint_names = skeletons['joint_names']
        # skeletons_dict = 

        return skeletons['joints_idx']

    # def joint_switch(self, ):

        # self.current_joint = 
    
