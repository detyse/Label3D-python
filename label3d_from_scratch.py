import os
import sys
import time 

from PySide6.QtGui import QMouseEvent
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

        # self.install_event_filters()


        self._init_property()

        self.initGUI()

       
        # eater = KeyPressEater()
        # self.installEventFilter(eater)
        # self.install_event_filters()
        

        # self.installEventFilter(KeyPressEater([self, *self.video_animators]))

    # def eventFilter(self, obj, event):
    #     print(f"get event in event filter {obj}")
    #     if event.type() == QEvent.KeyPress:
    #         print(f"key {event.key()} is pressed in event filter")
    #         for widget in self.widgets:
    #             if widget != obj: 
    #                 QCoreApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, event.key(), event.modifiers(), event.text(), event.isAutoRepeat(), event.count()))
    #                 # widget.keyPressEvent(event)
    #         return True
    #     return False


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
        self.current_joint_idx = None
        self.exist_markers = []
        self.joints2markers = {}

        # joint update

    def initGUI(self, ):        # do not ihnerit the initUI from Animator
        '''
        use a side bar in the main window to indicate the labeling joints
        '''
        self.setWindowTitle("3D Labeling Tool")

        # self.setGeometry(100, 100, 1300, 800)        # rec     # get the size accroding to the layout

        # TODO: check GUI line by line

        # # set the main widget for views of videos
        video_widget = QWidget(self)
        # video_widget.installEventFilter(self.eater)
        # video_widget.setGeometry(0, 0, 1000, 800)
        # # set a side bar for joints labeling
        side_bar = QWidget(self)

        # layout
        left_layout = QVBoxLayout()
        
        main_layout = QHBoxLayout()

        # # # side bar buttons, change the current joint, also change color to indicate weather it is labeled
        self.joint_buttons = {}

        for joint in self._joint_names:
            button = QRadioButton(joint)
            self.joint_buttons[joint] = button
            button.clicked.connect(self.button_select_joint)
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
        self.video_animators = self.get_animators()
        pos = self.get_views_position()

        for i, animator in enumerate(self.video_animators):
            # print(type(animator))
            # print(pos[i, 0], pos[i, 1])
            views_layout.addWidget(animator, pos[i, 0], pos[i, 1],)
            # show the parent of the animator
            print(animator.parentWidget())

        video_widget.setLayout(views_layout)
        # video_widget.show()
        # # video widget done here

        main_layout.addWidget(video_widget)
        self.setLayout(main_layout)  

    # def install_event_filters(self, filter_obj):
    #     for animator in [self, *self.video_animators]:
    #         animator.installEventFilter(filter_obj)


    # links all the animators
    def link_animators(self, ):
        # check the frame or align the frame
        for i in range(len(self.video_animators)):
            assert self.video_animators[i].nFrames == self.video_animators[0].nFrames, "The frame number of videos must be the same"
            assert self.video_animators[i].frame == self.video_animators[0].frame, "The frame index of videos must be the same"
        
        # set label3d frame property
        self.frame = self.video_animators[0].frame        
        self.nFrames = self.video_animators[0].nFrames
        self.frameInd = np.arange(self.nFrames)
        # TODO: should use restrict function to update the frame property

        self.linkAll([self, *self.video_animators])


    def get_animators(self, ):
        '''
        return the animators for each video
        '''
        video_animators = []
        for video in self.videos:
            animator = VideoAnimator(video, self.skeleton)
            animator.setParent(self)
            video_animators.append(animator)

        return video_animators
    
    # 返回view的位置，第几行 第几列, 保存在一个数组中 对应 views， 
    def get_views_position(self, nViews=None):
        if nViews is None:
            nViews = self.view_num
        
        nRows = np.floor(np.sqrt(nViews))

        pos = np.zeros((nViews, 2))
        if nViews > 3:
            for i in range(nViews):
                row = i % nRows
                col = np.floor(i / nRows)
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

    # some operations will change the joint status,
    def button_select_joint(self, ):
        button_joint = self.sender().text()
        print(f"Button {button_joint} is clicked")
        joint_idx = self._joint_names.index(button_joint)
        if button_joint == "Nothing":
            self.update_joint_status(None)
        self.update_joint_status(joint_idx)

    # including the current joint and the exist markers
    # also sync to all the animators, (using the joint idx)
    def update_joint_status(self, change_joint):       # change_joint is am index, 
        # if change_joint is None:
        #     self.current_joint = self.sender().text()
        #     self.current_joint_idx = self._joint_names.index(self.current_joint)
        # else:
        #     self.current_joint_idx = change_joint
        #     self.current_joint = self._joint_names[change_joint]

        if change_joint is not None:
            self.current_joint_idx = change_joint
            self.current_joint = self._joint_names[change_joint]
        else:
            self.current_joint = None
            self.current_joint_idx = None

        # set as checked
        if self.current_joint is None:
            self.current_joint_idx = None
            self.joint_buttons["Nothing"].setChecked(True)
        else:
            self.joint_buttons[self.current_joint].setChecked(True)

        # set the background color of exist markers.
        # 每个view的marker情况不一样，用tabel更方便
        # for marker in self.exist_markers:
            # self.joint_buttons[marker].setStyleSheet("background-color: lightcyan")

        for animator in self.video_animators:           # in animator there are only joint idx
            animator.current_joint_idx = self.current_joint_idx 

        print(f"joint update implemented, joint changed to {self.current_joint}, joint idx is {self.current_joint_idx}")



    def mousePressEvent(self, event: QMouseEvent) -> None:
        return super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        return super().mouseMoveEvent(event)

    def keyPressEvent(self, event):     # next frame
        super().keyPressEvent(event)        # all the operation sycn to all the animators should be here,
                                            # so no super call in animator key press event
    
        # if event.key() == Qt.Key_J:
            # self.next_frame()
        # test on the linked key press event, 
        # using the same key event as the animator
        if event.key() == Qt.Key_S:
            print("S is pressed in main window")


        # update the frame for all the animators
        
        
    