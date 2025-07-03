# TODO: 250120 do some GUI adjustment

import os
import sys
import time
import datetime
import scipy
import numpy as np
import pandas as pd

import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from utils.utils import read_json_skeleton, LoadYaml
from animator.animator import Animator, VideoAnimator

# use track to track the crush bug
import traceback

# is a GUI for manual labeling of 3D keypoints in multiple cameras.
# TODO: check the frame_num2label param and the frame_indexes param
class Label3D(Animator):
    # signal
    update_status = Signal(str)

    def __init__(self, camParams=None, video_folder=None, skeleton=None, frame_num2label=None, save_path=None, frame_indexes=None, total_frame_num=None, qc_mode=False, view_mode=False, **kwargs) -> None:      # the qc_mode could be defined in the yaml file
        super().__init__()
        # assert
        self.camParams = camParams          # the camera parameters
        self.video_folder = video_folder        # return a list of video_path for each view
        # now the property is meaningless
        
        # will be a just a folder
        self.skeleton = read_json_skeleton(skeleton)
        self.label_num = frame_num2label        # this property is not used at all
        self.total_frame_num = total_frame_num
        self.save_path = save_path

        self.view_mode = view_mode
        self.kwargs = kwargs
        self.views_video = self.get_views_video()        # the video path for each view, the video_folder    

        # FIXME not work here... 
        assert len(self.camParams) == len(self.views_video), f"The number of cameras and videos should be the same, {len(self.camParams)} != {len(self.views_video)}"
        self.view_num = len(self.camParams)
        self.frame_indexes = frame_indexes      # this is used for show the frame index in the video
        self.qc_mode = qc_mode
        # self.duplication_index = frame_index        # to get the duplication frames position

        self._unpack_camParams()
        self.frame_align_with_animators()
        self._init_properties()
        self._initGUI()
        self._load_labels()

        self.preview_mode_change()

    # 
    def get_views_video(self, ):
        # just read the frames in the output folder
        if self.view_mode is True:
            view_folders = [f for f in os.listdir(self.video_folder) if
                            os.path.isdir(os.path.join(self.video_folder, f))]
            view_folders.sort()
            views = [os.path.join(self.video_folder, f) for f in view_folders]
            return views

        # read the frames in the save folder
        video_folder = self.save_path
        frames_path = os.path.join(video_folder, "frames")
        
        view_folders = [f for f in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, f))]
        view_folders.sort()

        # join the path
        views = [os.path.join(frames_path, f) for f in view_folders]

        # print("get_views: " ,views) 
        return views


    def _unpack_camParams(self, ):
        r = []
        t = []
        K = []
        RDist = []
        TDist = []

        # TODO: confirm the camParams format and order
        # FIXME add the print to check the shape of the cam params
        for cam in self.camParams:          # keep order? 
            r.append(cam["r"][0][0].T)      #
            # print(cam["r"][0][0].T.shape)
            t.append(cam["t"][0][0])
            # print(cam["t"][0][0].shape)
            K.append(cam["K"][0][0].T)        # temp change
            # print(cam["K"][0][0].T.shape)
            RDist.append(cam["RDistort"][0][0])
            # print(cam["RDistort"][0][0].shape)
            TDist.append(cam["TDistort"][0][0])
            # print(cam["TDistort"][0][0].shape)

        self.r = np.array(r)
        self.t = np.array(t)
        self.K = np.array(K)
        self.RDist = np.array(RDist)
        self.TDist = np.array(TDist)


    def _init_properties(self, ):        
        # assert the cam and video number, keep the order
        assert len(self.camParams) == len(self.views_video), "The number of cameras and videos should be the same"
        print(f"Number of cameras: {len(self.camParams)}")
        print(f"Number of videos: {len(self.views_video)}")

        self.view_num = len(self.camParams)

        # the skeleton format aligned with DANNCE and Label3D
        # NOTE: _params are the immutable parameters
        self._joint_names = self.skeleton["joint_names"]
        self._joint_idx = self.skeleton["joints_idx"]               # no used
        self._color = self.skeleton["color"]

        self.current_joint = None
        self.current_joint_idx = None
        
        self.joints3d = np.full((self.nFrames, len(self._joint_names), 3), np.nan)          # NOTE: data to be saved
        
        # for Jiehan: add the original label points saving (not reprojected points) # quit large
        self.labeled_points = np.full((self.view_num, self.nFrames, len(self._joint_names), 2), np.nan)     # NOTE: data to be saved
        # how to get the original point position from the animator?

        if self.view_mode is False:
            if os.path.exists(os.path.join(self.save_path, "joints3d.npy")):
                print("Loading existing labels")
                self.joints3d = np.load(os.path.join(self.save_path, "joints3d.npy"))

            if os.path.exists(os.path.join(self.save_path, "labeled_points.npy")):
                print("Loading existing original labels")
                self.labeled_points = np.load(os.path.join(self.save_path, "labeled_points.npy"))

        else:
            self.joints3d = self.kwargs['joints3d']
            self.p_max = self.kwargs['p_max']
            # the labeled points could be not defined

        # for qc mode, and check the parameters are given or not
        # the qc mode should follow the config file
        if self.qc_mode:            
            self._tolerant_error = self.skeleton["tolerant_error"]
            self.qc_frames = []             # should be a list to store the low quality frames
            self.qc_passed = []               
            # get the frame index from the video folder
            self.duplication_index = np.load(os.path.join(self.save_path, "indexes.npy"))
            # TODO: check the duplication index only duplicate once

        # add the contrastive change for all frames
        self.contrast_factor = 1.0

        # set the preview mode
        if self.view_mode:
            self.preview_mode = False

        else:
            self.preview_mode = True


    def _initGUI(self, ):
        self.setCursor(Qt.ArrowCursor)
        self.setWindowTitle("3D Labeling Tool")

        video_widget = QWidget(self)
        side_bar = QWidget(self)

        left_layout = QVBoxLayout()
        main_layout = QHBoxLayout()
        frame_info_layout = QHBoxLayout()
        # frame_info_layout.setAlignment(Qt.AlignLeft)
        outer_layout = QVBoxLayout()

        # frame label bar
        self.frame_banner = QLabel(f"Current index in video: ", self)

        self.joint_button = {}

        for joint in self._joint_names:                     # use the radio button to show the label stituation
            button = QRadioButton(joint, side_bar)
            self.joint_button[joint] = button
            button.clicked.connect(self.button_select_joint)        
            # TODO: function button_select_joint connect to the joint change
            left_layout.addWidget(button)

        side_bar.setLayout(left_layout)
        main_layout.addWidget(side_bar)

        views_layout = QGridLayout()
        # set the space between widgets
        views_layout.setHorizontalSpacing(0)
        views_layout.setVerticalSpacing(0)

        positions = self._set_views_position()          # get the positions of the views
        
        for i, animator in enumerate(self.video_animators):
            views_layout.addWidget(animator, *positions[i])

        video_widget.setLayout(views_layout)

        next_frame_button = QToolButton(self)
        next_frame_button.setArrowType(Qt.RightArrow)
        next_frame_button.clicked.connect(lambda: self.frame_jump(True))
        # last_frame_button = QPushButton("Last Frame", self)
        last_frame_button = QToolButton(self)
        last_frame_button.setArrowType(Qt.LeftArrow)
        last_frame_button.clicked.connect(lambda: self.frame_jump(False))
        self.frame_info = QLabel(f"Frame: {self.frame + 1} / {self.nFrames}", self)
        frame_info_layout.addWidget(last_frame_button)
        frame_info_layout.addWidget(self.frame_info)
        frame_info_layout.addWidget(next_frame_button)
        
        self.jump_to = QLineEdit(self)
        # limit the max length of the input
        self.jump_to.setMaxLength(5)        # limit the character length
        self.jump_to.setValidator(QIntValidator(1, self.nFrames, self))        # limit the input range
        self.jump_to.setMaximumWidth(100)
        self.jump_to.setPlaceholderText("Jump to frame")
        self.jump_to.returnPressed.connect(lambda: self.jump_to_frame(int(self.jump_to.text()) - 1))
        frame_info_layout.addWidget(self.jump_to)

        self.jump_rate = QLabel(f"Jump Rate: {self.frameRate}", self)
        frame_info_layout.addWidget(self.jump_rate)

        frame_info_layout.addStretch()

        frame_info_layout.addWidget(self.frame_banner)
        
        frame_info_layout.addStretch()

        # TODO: add the preview button for preview mode change
        self.preview_button = QPushButton("Preview Mode", self)
        self.preview_button.clicked.connect(self.preview_mode_change)
        if self.view_mode:
            self.preview_button.setEnabled(False)
        frame_info_layout.addWidget(self.preview_button)

        # TODO: add the qc button on the frame info layout
        self.qc_button = QPushButton("Quality Check", self)
        if not self.qc_mode:
            self.qc_button.setEnabled(False)
        self.qc_button.clicked.connect(self.run_quality_check)
        frame_info_layout.addWidget(self.qc_button)

        main_layout.addWidget(video_widget)
        outer_layout.addLayout(main_layout)
        outer_layout.addLayout(frame_info_layout)

        self.setLayout(outer_layout)

    def _set_animators(self, ):         # the video passed to the animator should be a list of all the video files
        video_animators = []            # a function to get the corresponding video list, in the utils: yaml loader
        for video in self.views_video:        
            animator = VideoAnimator(video, self.skeleton, self.frame_indexes)
            animator.setParent(self)
            video_animators.append(animator)
        return video_animators
        # the linkage of animators and the video is here, have nothing with the position
    
    def _set_views_position(self, nViews=None):
        if nViews is None:
            nViews = self.view_num

        nRows = int(np.ceil(np.sqrt(nViews)))
        nCols = int(np.ceil(nViews / nRows))

        if nRows > nCols:
            nRows = nCols

        pos = np.zeros((nViews, 2))
        for i in range(nViews):
            row = i % nRows
            col = i // nRows
            pos[i, :] = [row, col]
        return pos

    # 
    def _load_labels(self, ):
        # load the self.joints3d to animators
        # Check if joints3d is empty
        if self.joints3d.size == 0:
            print("Error: self.joints3d is empty")
            return

        # Ensure joints3d is the correct shape and type
        if self.joints3d.ndim != 3:
            print(f"Error: self.joints3d should be 3-dimensional, but it has {self.joints3d.ndim} dimensions")
            return

        if self.joints3d.dtype != np.float32 and self.joints3d.dtype != np.float64:
            print(f"Warning: self.joints3d dtype is {self.joints3d.dtype}, converting to float32")
            self.joints3d = self.joints3d.astype(np.float32)
        self.reproject_for_load()
        self.update_radio_background()
        self.update_frame()
        for animator in self.video_animators:
            animator.reset_the_scale()

    def frame_align_with_animators(self, ):
        self.video_animators = self._set_animators()

        # check the frame or align the frame
        for i in range(len(self.video_animators)):
            assert self.video_animators[i].nFrames == self.video_animators[0].nFrames, "The frame number of videos must be the same"
            assert self.video_animators[i].frame == self.video_animators[0].frame, "The frame index of videos must be the same"
        
        # set label3d frame property
        self.frame = self.video_animators[0].frame
        self.nFrames = self.video_animators[0].nFrames

        # align the frame index
        # NOTE: the index of qc should be a shuffled list, so the len of index should be the nFrames
        if self.qc_mode:
            assert self.nFrames == 2 * len(self.frame_indexes), "The frame index is not aligned with the video frames"
        else:
            assert self.nFrames == len(self.frame_indexes), "The frame index is not aligned with the video frames"

        assert self.nFrames == len(self.frame_indexes), "The frame index is not aligned with the video frames"

        self.frameInd = np.arange(self.nFrames)


    ## methods used to handle the GUI
    def button_select_joint(self, ):
        button_joint = self.sender().text()
        self.update_joint(button_joint)

    
    def update_radio_background(self, ):             # will be called when joint is updated and t is pressed
        # update the radio button state and the frame info
        # if the joints3d is not nan, set the radio button background color to cyan
        for i, joint in enumerate(self.joints3d[self.frame]):
            if not np.isnan(joint).any():       # in is situation all and any are the same
                self.joint_button[self._joint_names[i]].setStyleSheet("QRadioButton { background-color: #7fb7be; }")
            else:
                self.joint_button[self._joint_names[i]].setStyleSheet("")


    def update_radio_checked(self, ):                # will be called when use tab to change the joint
        # update the radio button checked state
        if self.current_joint is not None:
            self.joint_button[self.current_joint].setChecked(True)
        else: 
            for button in self.joint_button.values():
                button.setChecked(False)


    def frame_jump(self, forward=True):
        print("frame jump is called")

        current_frame = self.frame
        if forward:
            if self.frame < self.nFrames - 1:
                if self.frame + self.frameRate < self.nFrames:
                    self.frame += self.frameRate
                else: 
                    self.frame = self.nFrames - 1
            else: self.frame = self.nFrames - 1
        else:
            if self.frame >= self.frameRate:
                self.frame -= self.frameRate
            else: self.frame = 0
        
        if current_frame != self.frame:
            # if there is no nan in the current frame joints3d, return False, do not need to warn
            if not np.isnan(self.joints3d[current_frame]).any():
                # all the joints are labeled, update frame directly
                self.update_frame()
            elif not self.preview_mode:
                if self.warning_for_framechange():
                    self.update_frame()
                else: 
                    self.frame = current_frame
            else:
                self.update_frame()
            # show the p_max value in the radio button
            return
        else:
            # do nothing
            return


    def update_p_max(self, ):
        # print(f"debug 20250310 {self.p_max.shape}")

        # update the p_max value in the radio button
        # iterate the joints name
        for i, joint in enumerate(self._joint_names):
            self.joint_button[joint].setText(f"{joint}: {self.p_max[self.frame, i]:.2f}")
        return
    

    def jump_to_frame(self, frame_index):
        current_frame = self.frame
        if frame_index < 0:
            self.frame = 0
        elif frame_index >= self.nFrames:
            self.frame = self.nFrames - 1
        else:
            self.frame = frame_index

        if current_frame != self.frame:
            # if there is no nan in the current frame joints3d, return False, do not need to warn
            if not np.isnan(self.joints3d[current_frame]).any():
                # all the joints are labeled, update frame directly
                self.update_frame()
                return
            elif not self.preview_mode:
                if self.warning_for_framechange():
                    self.update_frame()
                else: 
                    self.frame = current_frame
                    return
            else:
                self.update_frame()
                return
        else:
            # do nothing
            return


    # do not needed now
    def update_frameRate(self, forward=True):       # keep the value be int
        if forward:
            if self.frameRate < self.nFrames:
                if self.frameRate * 2 < self.nFrames:
                    self.frameRate *= 2
                else:
                    self.frameRate = self.nFrames - 1
            else: self.frameRate = self.nFrames - 1
        else:
            if self.frameRate > 1:
                self.frameRate //= 2
            else: self.frameRate = 1
        self.jump_rate.setText(f"Jump Rate: {self.frameRate}")


    # NOTE: when the preview mode is on, unable the key press event and unable the side bar buttons
    # change the joint into none, and unable the joint change
    def preview_mode_change(self, ):
        print(f"Preview mode change called, current mode: {self.preview_mode}")
        self.preview_mode = not self.preview_mode
        
        if self.preview_mode:
            # change the animator to the preview mode
            for animator in self.video_animators:
                animator.preview_mode = True
            # unable the joint buttons
            for button in self.joint_button.values():
                button.setEnabled(False)
            self.preview_button.setText("Change to Label Mode")

        else:
            for animator in self.video_animators:
                animator.preview_mode = False
            #  enable the joint buttons
            for button in self.joint_button.values():
                button.setEnabled(True)
            self.preview_button.setText("Change to Preview Mode")
        return 
    

    def keyPressEvent(self, event):
        # next frame, temporarily use the key F, will use the arrow Right key later
        if event.key() == Qt.Key_D:
            print("D is pressed")
            # if self.frame < self.nFrames - 1:
            #     self.frame += self.frameRate
            # else: self.frame = self.nFrames - 1
            # self.update_frame()
            self.frame_jump(True)
        
        # previous frame, temporarily use the key D, will use the arrow Left key later
        elif event.key() == Qt.Key_A:
            print("A is pressed")
            # if self.frame >= self.frameRate:
            #     self.frame -= self.frameRate
            # else: self.frame = 0
            # self.update_frame()
            self.frame_jump(False)

        elif event.key() == Qt.Key_Up:            # double the frame rate
            print("UP is pressed")
            # if self.frameRate < self.nFrames:
            #     self.frameRate *= 2
            # else: self.frameRate = self.nFrames - 1
            # self.jump_rate.setText(f"Jump Rate: {self.frameRate}")
            self.update_frameRate(True)

        elif event.key() == Qt.Key_Down:        # half the frame rate
            print("DOWN is pressed")
            # if self.frameRate > 1:
            #     self.frameRate /= 2
            # else: self.frameRate = 1
            # self.jump_rate.setText(f"Jump Rate: {self.frameRate}")
            self.update_frameRate(False)

        elif event.key() == Qt.Key_Q and not self.preview_mode  and not self.view_mode:
            print("Q is pressed")
            if self.current_joint_idx is None:
                self.update_joint(0)
            elif self.current_joint_idx > 0:
                self.update_joint(self.current_joint_idx - 1)
            else:
                self.update_joint(len(self._joint_names) - 1)

        # switch joint
        elif event.key() == Qt.Key_E and not self.preview_mode and not self.view_mode:
            print("E is pressed")
            if self.current_joint_idx is None:
                self.update_joint(0)
            elif self.current_joint_idx < len(self._joint_names) - 1:
                self.update_joint(self.current_joint_idx + 1)
            else:
                self.update_joint(0)
        

        # BUG: the bug is, delete some point and redraw them with a "T" will cause the point disappear
        # triangulate the 3D joint
        # TODO: check the function
        elif event.key() == Qt.Key_T and not self.preview_mode and not self.view_mode:
            print("T is pressed")
            # FIXME: when there are only one view is labeled, the reprojection will case the marker disappear
            if self.triangulate_all_joints():
                print("Triangulate all joints successfully")
                # reproject the 3D joint to the views
                self.reproject_for_load()
                self.update_radio_background()
                self.update_joint(self.current_joint_idx)


        elif event.key() == Qt.Key_S and not self.preview_mode and not self.view_mode:
            print("S is pressed")
            self.save_labels()

        elif event.key() == Qt.Key_M and not self.preview_mode and not self.view_mode:
            print("M is pressed")
            self.save_mat()

        elif event.key() == Qt.Key_R and not self.qc_mode and not self.view_mode:
            print("R is pressed")
            # clear the current joint
            self.clear_current_joint()
            self.update_radio_background()

        # define the quality check shortcut
        # elif event.key() == Qt.Key_C and not self.preview_mode:
        #     print("C is pressed")
        #     self.run_quality_check()

        elif event.key() == Qt.Key_F:
            print("F is pressed")
            # refresh the view -> rescale them
            for i, animator in enumerate(self.video_animators):
                animator.reset_the_scale()
            self.update_frame()

        if event.key() == Qt.Key_BracketLeft:
            print("key press event (animator): bracket left")
            self.contrast_factor -= 0.1
            if self.contrast_factor < 0.1:
                self.contrast_factor = 0.1
            self.contrastive_change()        # hope not need to use the update_frame method

        elif event.key() == Qt.Key_BracketRight:
            print("key press event (animator): bracket right")
            self.contrast_factor += 0.1
            if self.contrast_factor > 5.0:
                self.contrast_factor = 5.0
            self.contrastive_change()

        else:
            super().keyPressEvent(event)


    def contrastive_change(self, ):
        # change the contrast for all frames
        for i, animator in enumerate(self.video_animators):
            animator.contrast_change(self.contrast_factor)
        return 


    ## update the current joint, called by button and key press
    ## could update the button?
    def update_joint(self, input=None):
        # print("Update joint: Label3d - update_joint is called")
        if input is None:       # init the joint state
            self.current_joint = None
            self.current_joint_idx = None

            # TODO: test work
            for button in self.joint_button.itervalues():
                button.setChecked(False)

        else: 
            if isinstance(input, int):
                assert input < len(self._joint_names) and input >= 0, f"Index {input} is out of range"
                self.current_joint_idx = input
                self.current_joint = self._joint_names[input]

            elif isinstance(input, str):
                assert input in self._joint_names, f"Joint {input} is not in the joint list"
                self.current_joint = input
                self.current_joint_idx = self._joint_names.index(input)

            # TODO: test work
            self.joint_button[self.current_joint].setChecked(True)

        for animator in self.video_animators:
            animator.set_joint(self.current_joint_idx)
            # could highlight the current joint in this function and 

        self.update_radio_checked()


    # update the frame to control the animator frame change
    def update_frame(self, ):
        # update the frame index
        self.frame_banner.setText(f"Current index in video: {self.frame_indexes[self.frame]}")

        self.frame_info.setText(f"Frame: {self.frame + 1} / {self.nFrames}")

        for i, animator in enumerate(self.video_animators):
            #animator.reset_the_scale()
            animator.update_frame(self.frame)
            # collect the original labeled points
            self.labeled_points[i, self.frame] = np.array(animator.get_all_original_marker_2d())

        self.update_radio_background()
        self.update_radio_checked()


        # is self.p_max is defined and self.view_mode is True
        if hasattr(self, 'p_max') and self.view_mode:
            self.update_p_max()

        # TODO: will the enable state change?
        if not self.view_mode:
            self.save_labels()


    # turn the current joint data into nan
    # TODO: this method would induce a bug, check where the bug could be
    # FIXME: this function may cause crush, using try catch to capture the crush bug
    def clear_current_joint(self, ):
        try:
            if self.current_joint is None:
                print("Please select a joint first")
                return False
            
            self.joints3d[self.frame, self.current_joint_idx, ...] = np.nan
            self.labeled_points[:, self.frame, self.current_joint_idx] = np.nan

            # update the frames_markers of each frame
            for i, animator in enumerate(self.video_animators):
                animator.clear_marker_2d()          # do not need here to sync the frame and the joint

            self.update_frame()
        except Exception as e:
            print("Clear current joint got an error:: ", e)
            traceback.print_exc()
        
        return True


    def triangulate_all_joints(self, ):         # called when T is pressed
        # update all the joints at once
        for i in range(len(self._joint_names)):         # or joint in self._joint_idx
            
            # do not update joint just update the joints3d
            frame_view_markers = np.full((self.view_num, 2), np.nan)
            for j, animator in enumerate(self.video_animators):         # get the joint position in each view
                # print(f"joint: {self._joint_names[i]} in view: {j}, return the marker {animator.get_marker_2d(self.frame, i)}")
                frame_view_markers[j] = animator.get_marker_2d(self.frame, i)       # FIXME: bug here
                
            # get the how many views have the joint
            view_avaliable = ~np.isnan(frame_view_markers).all(axis=1)
            if np.sum(view_avaliable) < 2:
                self.joints3d[self.frame, i, :] = np.nan
                continue

            print("T the joint: ", self._joint_names[i])

            points2d = frame_view_markers[view_avaliable, :]
            r = self.r[view_avaliable]
            t = self.t[view_avaliable]
            K = self.K[view_avaliable]
            RDist = self.RDist[view_avaliable]
            TDist = self.TDist[view_avaliable]

            point_3d = triangulateMultiview(points2d, r, t, K, RDist, TDist)
            self.joints3d[self.frame, i, :] = point_3d

        return True
    
    
    # FIXME: could not handle the situation that the joint have nan value
    def reproject_for_load(self, ):
        # reproject the 3d points to views at once
        views_frames_markers = np.full((self.view_num, self.nFrames, len(self._joint_names), 2), np.nan) 
        for k, frame_points in enumerate(self.joints3d):        # hopefully be the frames index
            # if np.isnan(frame_points).any():            # NOTE: all and any are the same also
            #     for u, animator in enumerate(self.video_animators):
            #         views_frames_markers[u, k, :] = np.nan
            #     continue
            for i in range(len(self._joint_names)):
                if np.isnan(frame_points[i]).any():
                    views_frames_markers[:, k, i] = np.nan
                    continue
                point3d = frame_points[i]
                for j, animator in enumerate(self.video_animators):
                    reprojected_points = reprojectToViews(point3d, self.r, self.t, self.K, self.RDist, self.TDist, self.view_num)
                    views_frames_markers[j, k, i] = reprojected_points[j]

        # integrate animator update in the 
        for i, animator in enumerate(self.video_animators):
            animator.load_labels(views_frames_markers[i], self.labeled_points[i])
        return views_frames_markers

    
    # reproject the 3D joint to the views
    # TODO: check the function
    # the reprojection is using opencv projectPoints function
    # just reproject the current joint 
    def reproject_joints(self, ):       # function not used
        if self.current_joint is None:
            print("Please select a joint first")
            return False
        # 
        reprojected_points = reprojectToViews(self.joints3d[self.frame, self.current_joint_idx],
                                             self.r, self.t, self.K, self.RDist, self.TDist, self.view_num) 
        for i, animator in enumerate(self.video_animators):
            animator.set_marker_2d(reprojected_points[i], reprojection=True)
        return True
    
 
    def save_labels(self, ):        # 
        # save the labeled 3D joints and the original 2D points, will overwrite when save again
        # name the saved files
        joints3d = np.array(self.joints3d)
        labeled_points = np.array(self.labeled_points)

        # save the data
        np.save(os.path.join(self.save_path, "joints3d.npy"), joints3d)
        np.save(os.path.join(self.save_path, "labeled_points.npy"), labeled_points)

        print("Data saved!")
        return True

    def save_mat(self, ):           # this function should not be here, move to the frame sampler part, which will leave a file to build the mat file
        #TODO: modify the data structure to fit the DANNCE format
        print('saving the joints3d and labeled_points into a DANNCE format mat file')
        #saving the joints3d and labeled_points into a DANNCE format mat file
        joints3d = np.array(self.joints3d)
        labeled_points = np.array(self.labeled_points)

        fps = 25
        fpm = 1000/fps  # frame period in ms
        total_frames_num = self.total_frame_num
        sync_data_frame = np.arange(total_frames_num).astype("float64")
        sync_data_sampleID = sync_data_frame * fpm + 1
        sync_data_2d = np.zeros((total_frames_num, 2 * labeled_points.shape[2]))
        sync_data_3d = np.zeros((total_frames_num, 3 * joints3d.shape[1]))

        # reshaping the joints3d and labeled_points to DANNCE format :(frame, joint*3) and (view, frame, joint*2)
        joints3d = joints3d.reshape((joints3d.shape[0], joints3d.shape[1] * joints3d.shape[2]))
        labeled_points = labeled_points.reshape(
            (labeled_points.shape[0], labeled_points.shape[1], labeled_points.shape[3] * labeled_points.shape[2]))

        camParams = []
        labelData = []
        camnames = []
        sync = []
        frame_indexs = np.load(self.frame_indexes)

        for i in range(self.view_num):
            camnames.append(np.array("Camera" + str(i+1)))
            labelData.append({"data_2d":labeled_points[i,:,:], "data_3d": joints3d, "data_frame":frame_indexs, "data_sampleID": sync_data_sampleID[:, np.newaxis][frame_indexs]})
            sync.append({"data_frame":sync_data_frame[:, np.newaxis], "data_sampleID":sync_data_sampleID[:, np.newaxis], "data_2d":sync_data_2d, "data_3d":sync_data_3d})
            camParams.append([self.camParams[i][0][0]])


        now = str(datetime.datetime.now())
        now = now[:now.find(".")]
        now = now.replace(" ", "_")
        now = now.replace(":", "")
        mat_file_path = os.path.join(self.save_path, now + '_py_annotation' ".mat")

        scipy.io.savemat(mat_file_path, {"camnames":camnames,"labelData": labelData, "sync": sync, 'params':camParams})
        print("Mat file saved to: ", mat_file_path)


    def closeEvent(self, event):
        for i, animator in enumerate(self.video_animators):
            self.labeled_points[i, self.frame] = np.array(animator.get_all_original_marker_2d())
            animator.close()
        if not self.view_mode:
            self.save_labels()
        event.accept()
        return
    

    # a warning message box jump out when the label is not saved
    # could continue or turn back
    def warning_for_framechange(self, ):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("There are unsaved labels, press T to reproject the 3D joint \nClick OK to continue")
        msg.setWindowTitle("Warning")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)
        button = msg.exec()
        if button == QMessageBox.Ok:
            return True
        else:
            return False


    # do qc in this function, 
    # 1. check the all the joint is labeled or jump a warning message, if all labeled, jump the info message
    # 2. load the addtional information for the qc: the tolerant error for each joint, load from skeleton file (the error is the pixel error). the duplication index for the shuffled frames, could laod from an additional file
    # 3. a new label3d window(could jump to these frames) for the not good frames(or just give the index to the user), the not good frames are the frames that the error is larger than the tolerant error
    # 4. recheck the not good frames, and save the new labels if qc passed

    # TODO: add the popup message box for the qc frames, and save an excel file to show the qc frames
    # write the qc information into the excel file
    def run_quality_check(self, ):
        # get the data from te 
        duplicated_frames = []      # is a list of pairs of the frames index, the duplicated frames will be together
        # 
        frame_index_collection = sorted(list(set(self.duplication_index)))
        for index in frame_index_collection:
            # get the position pair of the duplicated frames (index1, index2)
            position_pair = [j for j, x in enumerate(self.duplication_index) if x == index]
            duplicated_frames.append(position_pair)
        
        # excel file
        qc_df = pd.DataFrame()
        # build the joint col, self._joint_name
        joint_col = []
        for joint in self._joint_names:
            joint_col.append(joint)

        # write the joint col into the df
        qc_df['joints/frames'] = joint_col

        # remove the repeated elements in the list
        list_of_tuples = [tuple(i) for i in duplicated_frames]
        duplicated_frames = list(set(list_of_tuples))

        # check the error for each frame and each joint
        for frame_pair in duplicated_frames:            # list of list
            frame_one = self.joints3d[frame_pair[0]]        # shape: (joint_num, 3)
            frame_two = self.joints3d[frame_pair[1]]

            # if there are nan in the labeled points, skip the frame
            if np.isnan(frame_one).any() or np.isnan(frame_two).any():
                continue

            # calculate the error for each joint
            error = np.sqrt(np.sum((frame_one - frame_two) ** 2, axis=1))
            
            frame_qc_col = []
            # check the error for each joint
            qc_pass = True
            for i, joint_error in enumerate(error):
                if joint_error > self._tolerant_error[i]:
                    frame_qc_col.append("Fail")
                    self.qc_frames.append(frame_pair)
                    qc_pass = False
                else:
                    frame_qc_col.append("Pass")

            qc_df[f"{frame_pair[0]}_{frame_pair[1]}"] = frame_qc_col

            if qc_pass:
                self.qc_passed.append(frame_pair)

        if not qc_pass:
            # write the df into the csv file, get current time as the file name
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
            qc_file_name = f"qc_{current_time}.csv"
            qc_df.to_csv(os.path.join(self.save_path, qc_file_name), index=False)

        # if there is no qc frames, jump a message box
        if len(self.qc_frames) == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("There is no frame need to be checked")
            msg.setWindowTitle("Quality Check")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Ok)
            button = msg.exec()
            return

        # if there are qc frames, jump a new label3d window for the qc frames, -> save the joint different into a excel file in the output folder
        # TODO: the message box should now be closed
        else: 
            # jump a message box to show the qc frames
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("There are frames that need to be checked showing at the bottom, \nclick OK to continue")
            msg.setWindowTitle("Quality Check")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Ok)
            button = msg.exec()

            # update the frames info into the main window status bar
            self.update_status.emit(f"Quality Check: frame {self.qc_frames} need to be checked")



        # save the qc pass indexes
        save_path = os.join(self.save_path, "qc_passed.npy")
        np.save(save_path, self.qc_passed)

        return True


    # not use now, only change one joint at a time
    def triangulate_joints(self, ):         # called when T is pressed
        # print("triangulate is called")
        if self.current_joint is None:
            print("Please select a joint first")
            return False
        
        frame_view_markers = np.full((self.view_num, 2), np.nan)
        for i, animator in enumerate(self.video_animators):     # get the joint position in each view
            frame_view_markers[i] = animator.get_marker_2d(self.frame, self.current_joint_idx)

        # get the how many views have the joint
        view_avaliable = ~np.isnan(frame_view_markers).all(axis=1)      
        # print("the index of the avaliable view:   ", view_avaliable)       # should be a vector with 0 and 1, as index for the views

        if np.sum(view_avaliable) < 2:
            print("At least two views are needed to triangulate a joint")
            return False
        
        points2d = frame_view_markers[view_avaliable, :]
        
        r = self.r[view_avaliable]
        t = self.t[view_avaliable]
        K = self.K[view_avaliable]
        RDist = self.RDist[view_avaliable]
        TDist = self.TDist[view_avaliable]

        # change to for loop to handle all the joints

        point_3d = triangulateMultiview(points2d, r, t, K, RDist, TDist)
        self.joints3d[self.frame, self.current_joint_idx, :] = point_3d
        
        # print("the 3D joint position: ", point_3d)
        return True     

############################################################################################################
# utils

# triangulate
def triangulateMultiview(points2d, r, t, K, RDist, TDist):
    cam_points = []
    for i in range(len(points2d)):
        the_K = K[i]
        # the_K[0, 1] = 0         # set the skew to 0
        point = np.array(points2d[i], dtype=np.float32).reshape(-1, 1, 2)

        # TODO: check skip the undistort, it is fine
        dist_vec = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        undistort_point = cv2.undistortPoints(point, the_K, dist_vec, P=the_K)
        
        # TODO: check the undistort, it is fine too, but the error is quite large, recalibrate the camera intrinsic
        pixel_coords = np.array([undistort_point[0][0][0], undistort_point[0][0][1], 1])
        inv_K = np.linalg.inv(the_K)
        cam_point = np.dot(inv_K, pixel_coords)   

        cam_points.append(cam_point[:2])
        # check the undist format 

    # triangulate the points
    cam_points = np.array(cam_points)

    A = []
    for i in range(len(r)):
        P = np.hstack((r[i], t[i].T))       # shape: (3, 4)  # there is a transpose, stack the P matrix here
        A.append(np.vstack((cam_points[i][0] * P[2, :] - P[0, :],
                            cam_points[i][1] * P[2, :] - P[1, :])))

    A = np.array(A)
    A = np.concatenate(A, axis=0)

    # SVD method to solve the hyper define problem
    _, _, V = np.linalg.svd(A)
    point_3d = V[-1, :]             # find the largest eign value vector
    point_3d = point_3d / point_3d[-1]
    return point_3d[:3]


# reprojection
def reprojectToViews(points3d, r, t, K, RDist, TDist, view_num):
    rvec = []
    for i in range(view_num):
        rvec.append(cv2.Rodrigues(r[i])[0])        # transfer the rotation vector into rotation matrix

    reprojected_points = []
    for i in range(view_num):
        the_K = K[i]
        # the_K[0, 1] = 0         # set the skew to 0
        dist_coef = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        # print(f"debug 20250311 view_num: {i}")
        # print(f"debug 20250311 rvec[i]: {rvec[i]}")
        # print(f"debug 20250311 t[i]: {t[i]}")
        # print(f"debug 20250311 the_K: {the_K}")
        # print(f"debug 20250311 dist_coef: {dist_coef}")
        # print(f"debug 20250311 points3d: {points3d}")
        # print(f"====================================================")
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], the_K, dist_coef)      # dist_coef
        reprojected_points.append(reprojected_point)

    reprojected_points = np.array(reprojected_points).squeeze()
    # print("the reprojected points: ", reprojected_points.shape)
    return reprojected_points


############################################################################################################