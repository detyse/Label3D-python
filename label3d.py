# the main ui here

import os
import sys
import time 

import numpy as np
import pandas as pd

import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from utils.utils import read_json_skeleton, LoadYaml
from animator.animator import Animator, VideoAnimator


# is a GUI for manual labeling of 3D keypoints in multiple cameras.
class Label3D(Animator):
    def __init__(self, camParams=None, videos=None, skeleton=None, frame_num2label=None, save_path=None) -> None:
        super().__init__()  
        # assert
        self.camParams = camParams          # the camera parameters
        self.videos = videos
        print(f'label3d videos format: {self.videos}')
        self.skeleton = read_json_skeleton(skeleton)
        self.label_num = frame_num2label

        self.save_path = save_path

        assert len(self.camParams) == len(self.videos)
        self.view_num = len(self.camParams)

        self._unpack_camParams()
        self.frame_align_with_animators()
        self._init_properties()
        
        self._initGUI()

        self._load_labels()


    def _unpack_camParams(self, ):
        r = []
        t = []
        K = []
        RDist = []
        TDist = []

        # TODO: confirm the camParams format and order
        for cam in self.camParams:          # keep order? 
            r.append(cam["r"][0][0].T)
            t.append(cam["t"][0][0])
            K.append(cam["K"][0][0].T)
            RDist.append(cam["RDistort"][0][0])
            TDist.append(cam["TDistort"][0][0])

        self.r = np.array(r)
        self.t = np.array(t)
        self.K = np.array(K)
        self.RDist = np.array(RDist)
        self.TDist = np.array(TDist)


    def _init_properties(self, ):
        # no need for video assert because video is a list
        # for video in self.videos:       # 
        #     assert os.path.exists(video), f"Video file {video} not found"
        
        # assert the cam and video number, keep the order
        # TODO: change here to keep the order, maybe using dict 
        assert len(self.camParams) == len(self.videos), "The number of cameras and videos should be the same"

        self.view_num = len(self.camParams)

        # the skeleton format aligned with DANNCE and Label3D
        self._joint_names = self.skeleton["joint_names"]
        self._joint_idx = self.skeleton["joints_idx"]
        self._color = self.skeleton["color"]

        self.current_joint = None
        self.current_joint_idx = None

        self.joints3d = np.full((self.nFrames, len(self._joint_names), 3), np.nan)          # NOTE: data to be saved
        
        # for Jiehan: add the original label points saving (not reprojected points) # quit large
        self.labeled_points = np.full((self.view_num, self.nFrames, len(self._joint_names), 2), np.nan)     # NOTE: data to be saved
        # how to get the original point position from the animator?
        
        if os.path.exists(os.path.join(self.save_path, "joints3d.npy")):
            print("Loading existing labels")
            self.joints3d = np.load(os.path.join(self.save_path, "joints3d.npy"))
            
        if os.path.exists(os.path.join(self.save_path, "labeled_points.npy")):
            print("Loading existing original labels")
            self.labeled_points = np.load(os.path.join(self.save_path, "labeled_points.npy"))


    def _initGUI(self, ):
        self.setCursor(Qt.ArrowCursor)
        self.setWindowTitle("3D Labeling Tool")

        video_widget = QWidget(self)
        side_bar = QWidget(self)

        left_layout = QVBoxLayout()
        main_layout = QHBoxLayout()

        self.joint_button = {}

        for joint in self._joint_names:
            button = QRadioButton(joint, side_bar)
            self.joint_button[joint] = button
            button.clicked.connect(self.button_select_joint)        
            # TODO: function button_select_joint connect to the joint change
            left_layout.addWidget(button)

        side_bar.setLayout(left_layout)
        main_layout.addWidget(side_bar)

        # TODO: shrink the margin of the video_widget
        views_layout = QGridLayout()
        # set the space between widgets
        views_layout.setHorizontalSpacing(0)
        views_layout.setVerticalSpacing(0)

        positions = self._set_views_position()          # get the positions of the views
        
        for i, animator in enumerate(self.video_animators):
            views_layout.addWidget(animator, *positions[i])

        video_widget.setLayout(views_layout)
        main_layout.addWidget(video_widget)

        self.setLayout(main_layout)


    def _set_animators(self, ):         # the video passed to the animator should be a list of all the video files
        video_animators = []            # a function to get the corresponding video list, in the utils: yaml loader
        for video in self.videos:        
            animator = VideoAnimator(video, self.skeleton, self.label_num)
            animator.setParent(self)
            video_animators.append(animator)
        return video_animators
        # the linkage of animators and the video is here, have nothing with the position


    def _set_views_position(self, nViews=None):
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
    

    # 
    def _load_labels(self, ):
        # load the self.joints3d to animators
        views_frames_markers = self.reproject_for_load()
        for i, animator in enumerate(self.video_animators):
            animator.load_labels(views_frames_markers[i], self.labeled_points[i])

        print(f"self.frame: {self.frame}")
        # self.update_frame()
        return True


    def frame_align_with_animators(self, ):
        self.video_animators = self._set_animators()

        # check the frame or align the frame
        for i in range(len(self.video_animators)):
            assert self.video_animators[i].nFrames == self.video_animators[0].nFrames, "The frame number of videos must be the same"
            assert self.video_animators[i].frame == self.video_animators[0].frame, "The frame index of videos must be the same"
        
        # set label3d frame property
        self.frame = self.video_animators[0].frame        
        self.nFrames = self.video_animators[0].nFrames
        self.frameInd = np.arange(self.nFrames)
        return 


    ## methods used to handle the GUI
    def button_select_joint(self, ):
        button_joint = self.sender().text()
        self.update_joint(button_joint)


    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        
        # next frame, temporarily use the key F, will use the arrow Right key later
        if event.key() == Qt.Key_F:
            print("f is pressed")
            if self.frame < self.nFrames - 1:
                self.frame += self.frameRate
            else: self.frame = self.nFrames - 1
            self.update_frame()
        
        # previous frame, temporarily use the key D, will use the arrow Left key later
        elif event.key() == Qt.Key_D:
            print("d is pressed")
            if self.frame >= self.frameRate:
                self.frame -= self.frameRate
            else: self.frame = 0
            self.update_frame()
        
        # switch joint
        elif event.key() == Qt.Key_Tab:
            print("tab is pressed")
            if self.current_joint_idx is None:
                self.update_joint(0)
            elif self.current_joint_idx < len(self._joint_names) - 1:
                self.update_joint(self.current_joint_idx + 1)
            else:
                self.update_joint(0)
        
        # triangulate the 3D joint
        elif event.key() == Qt.Key_T:
            print("t is pressed")
            if self.triangulate_joints():
                # reproject the 3D joint to the views
                self.reproject_joints()

        elif event.key() == Qt.Key_S:
            print("s is pressed")
            self.save_labels()

        elif event.key() == Qt.Key_R and QApplication.keyboardModifiers() == Qt.ControlModifier:
            print("Ctrl+R is pressed")
            # clear the current joint
            self.clear_current_joint()


    ## update the current joint, called by button and key press
    ## could update the button?
    def update_joint(self, input=None):
        print("Update joint: Label3d - update_joint is called")
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


    # update the frame to control the animator frame change
    def update_frame(self, ):
        for i, animator in enumerate(self.video_animators):
            animator.update_frame(self.frame)
            # collect the original labeled points
            self.labeled_points[i, self.frame] = np.array(animator.get_all_original_marker_2d())

        self.save_labels()


    # turn the current joint data into nan
    def clear_current_joint(self, ):
        if self.current_joint is None:
            print("Please select a joint first")
            return False
        
        self.joints3d[self.frame, self.current_joint_idx] = np.nan
        self.labeled_points[:, self.frame, self.current_joint_idx] = np.nan

        # update the frames_markers of each frame
        for i, animator in enumerate(self.video_animators):
            animator.clear_marker_2d()          # do not need here to sync the frame and the joint

        self.update_frame()
        return True


    # TODO: check the function
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
        
        # print(points2d)

        r = self.r[view_avaliable]
        t = self.t[view_avaliable]
        K = self.K[view_avaliable]
        RDist = self.RDist[view_avaliable]
        TDist = self.TDist[view_avaliable]

        point_3d = triangulateMultiview(points2d, r, t, K, RDist, TDist)
        self.joints3d[self.frame, self.current_joint_idx, :] = point_3d
        
        # print("the 3D joint position: ", point_3d)
        return True


    def reproject_for_load(self, ):
        # reproject the 3d points to views at once
        views_frames_markers = np.full((self.view_num, self.nFrames, len(self._joint_names), 2), np.nan) 
        for k, frame_points in enumerate(self.joints3d):        # hopefully be the frames index
            if np.isnan(frame_points).all():
                continue
            for i in range(len(self._joint_names)):
                if np.isnan(frame_points[i]).all():
                    continue
                point3d = frame_points[i]
                for j, animator in enumerate(self.video_animators):
                    reprojected_points = reprojectToViews(point3d, self.r, self.t, self.K, self.RDist, self.TDist, self.view_num)
                    views_frames_markers[j, k, i] = reprojected_points[j]
        return views_frames_markers

    
    # reproject the 3D joint to the views
    # TODO: check the function
    # the reprojection is using opencv projectPoints function
    def reproject_joints(self, ):
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

        print("data saved")
        return True


    def closeEvent(self, event):
        for i, animator in enumerate(self.video_animators):
            self.labeled_points[i, self.frame] = np.array(animator.get_all_original_marker_2d())
            animator.close()
        self.save_labels()
        event.accept()
        return


# triangulate
# def 
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
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], the_K, dist_coef)      # dist_coef
        reprojected_points.append(reprojected_point)

    reprojected_points = np.array(reprojected_points).squeeze()
    # print("the reprojected points: ", reprojected_points.shape)
    return reprojected_points


############################################################################################################