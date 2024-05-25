import os
import sys
import time 

import numpy as np
import pandas as pd

import cv2
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

from utils.utils import Loader, read_json_skeleton
from animator.animator_v2 import Animator, VideoAnimator


class Label3D(Animator):
    '''
    Inputs:
    camParams: list of dict, each dict contains the camera parameters, including r, t, K, RDistort, TDistort. The format is consistent with label3d.
    videos: list of str, the paths of videos
    skeleton_path: str, the path of the skeleton json file, containing the joint names, joint connections, and joint colors.
    pick_frames: list(or array) of int, the frames to be picked for labeling, if not provided, all frames will be labeled.

    loader: Loader, the loader object, containing all the inputs, including camParams, videos, skeleton_path, and pick_frames.
    '''
    def __init__(self, camParams=None, videos=None, skeleton_path=None, loader=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
     
        self._unpack_camParams()      # get settings from params
        self.align_animators()        # could be reuse  # 
        self._init_property()

        self._initGUI()

    ## methods used to initialize the class
    def _unpack_camParams(self, ):
        r = []
        t = []
        K = []
        RDist = []
        TDist = []
        for cam in self.camParams:
            r.append(cam["r"][0][0])                # shape: (3, 3), rotation matrix
            t.append(cam["t"][0][0])                # shape: (1, 3), translation vector
            K.append(cam["K"][0][0].T)                # shape: (3, 3), intrinsic matrix, need to transpose to fit the shape
            RDist.append(cam["RDistort"][0][0])     # shape: (1, 3), corresponding to k1, k2, k3
            TDist.append(cam["TDistort"][0][0])     # shape: (1, 2), corresponding to p1, p2
        
        self.r = np.array(r)
        self.t = np.array(t)
        self.K = np.array(K)
        self.RDist = np.array(RDist)
        self.TDist = np.array(TDist)

    def _init_property(self, ):
        for video in self.videos:
            assert os.path.exists(video), f"Video {video} does not exist"

        assert len(self.camParams) == len(self.videos), "The number of cameras and videos must be the same"
        self.view_num = len(self.camParams)

        self._joint_names = self.skeleton["joint_names"]
        self._joint_idx = self.skeleton["joints_idx"]
        self._color = self.skeleton["color"]

        self.current_joint = None
        self.current_joint_idx = None

        self.joints3d = np.full((self.nFrames, len(self._joint_names), 3), np.nan)


    def _initGUI(self, ):        # get animators 
        self.setCursor(Qt.ArrowCursor)
        self.setWindowTitle("3D Labeling Tool")

        video_widget = QWidget(self)
        side_bar = QWidget(self)

        left_layout = QVBoxLayout()
        main_layout = QHBoxLayout()

        self.joint_button = {}

        for joint in self._joint_names:
            button = QRadioButton(joint, side_bar)      # set parent to side_bar?
            self.joint_button[joint] = button
            button.clicked.connect(self.button_select_joint)
            left_layout.addWidget(button)

        # do not set nothing button
        
        side_bar.setLayout(left_layout)
        main_layout.addWidget(side_bar)
        
        views_layout = QGridLayout()
        pos = self._set_views_position()

        for i, animator in enumerate(self.video_animators):
            views_layout.addWidget(animator, *pos[i])

        video_widget.setLayout(views_layout)
        main_layout.addWidget(video_widget)

        self.setLayout(main_layout)

    def _set_animators(self, ):
        video_animators = []
        for video in self.videos:
            animator = VideoAnimator(video, self.skeleton)
            animator.setParent(self)        # 
            video_animators.append(animator)
        return video_animators

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
        
    def align_animators(self, ):
        self.video_animators = self._set_animators()
        
        # check the frame or align the frame
        for i in range(len(self.video_animators)):
            assert self.video_animators[i].nFrames == self.video_animators[0].nFrames, "The frame number of videos must be the same"
            assert self.video_animators[i].frame == self.video_animators[0].frame, "The frame index of videos must be the same"
        
        # set label3d frame property
        self.frame = self.video_animators[0].frame        
        self.nFrames = self.video_animators[0].nFrames
        self.frameInd = np.arange(self.nFrames)
        # TODO: should use restrict function to update the frame property
        # reset

    ## methods used to handle the GUI
    def button_select_joint(self, ):
        button_joint = self.sender().text()
        self.update_joint(button_joint)


    def keyPressEvent(self, event):
        super().keyPressEvent(event)

        # next frame, temporarily use the key F, will use the arrow Right key later
        if event.key() == Qt.Key_F:
            if self.frame < self.nFrames - 1:
                self.frame += self.frameRate
            else: self.frame = self.nFrames - 1
            self.update_frame()
        
        # previous frame, temporarily use the key D, will use the arrow Left key later
        elif event.key() == Qt.Key_D:
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
            if self.triangulate_joints():
                # reproject the 3D joint to the views
                self.reproject_joints()

        
    ## methods to update the states of the GUI
    def update_joint(self, input=None):
        print("update joint")
        if input is None:
            self.current_joint_idx = None
            self.current_joint = None
            # turn off all the buttons
            for button in self.joint_button.values():       # self.joint_button is a dict
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
        
            self.joint_button[self.current_joint].setChecked(True)
        
        for animator in self.video_animators:
            animator.set_joint(self.current_joint_idx)

    def update_frame(self, ):
        for animator in self.video_animators:
            animator.update_frame(self.frame)

    def triangulate_joints(self, ):         # triangulate the current frame current joint 2D points
        if self.current_joint is None:
            print("Please select a joint first")
            return False
        
        frame_view_markers = np.full((self.view_num, 2), np.nan)
        for i, animator in enumerate(self.video_animators):
            frame_view_markers[i] = animator.get_marker_2d(self.frame, self.current_joint_idx)   # shape: (view_num, 2), the params could ignore 

        view_available = ~np.isnan(frame_view_markers).all(axis=1)

        # at least two views to triangulate
        if np.sum(view_available) < 2:
            print("At least two views are needed to triangulate")
            return False
        
        # triangulate the 3D joint
        points2d = frame_view_markers[view_available, :]
        r = self.r[view_available]
        t = self.t[view_available]
        K = self.K[view_available]
        RDist = self.RDist[view_available]
        TDist = self.TDist[view_available]

        point_3d = triangulateMultiview(points2d, r, t, K, RDist, TDist)
        self.joints3d[self.frame, self.current_joint_idx] = point_3d

        return True

    def reproject_joints(self, ):
        if self.current_joint is None:
            print("Please select a joint first")    # will not happen
            return
        
        reprojected_point = reprojectToViews(self.joints3d[self.frame, self.current_joint_idx], self.r, self.t, self.K, self.RDist, self.TDist, self.view_num)
        for i, animator in enumerate(self.video_animators):         # animators are corresponding to the views. TODO: make sure the order is correct
            animator.set_marker_2d(reprojected_point[i], self.frame, self.current_joint_idx)           # the params could ignore

    def animator_key_press(self, ):
        pass

## utils
# ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# ref: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
# ref: https://gutsgwh1997.github.io/2020/03/31/%E5%A4%9A%E8%A7%86%E5%9B%BE%E4%B8%89%E8%A7%92%E5%8C%96/
def triangulateMultiview(points2d, r, t, K, RDist, TDist):
    # 
    # print(f"points2d: {points2d.shape}, r: {r.shape}, t: {t.shape}, K: {K.shape}, RDist: {RDist.shape}, TDist: {TDist.shape}")
    undist = []
    for i in range(len(points2d)):
        # modify some params
        the_K = K[i]
        the_K[0, 1] = 0
        point = points2d[i]
        dist_vec = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        undistort_point = cv2.undistortPoints(point, the_K, dist_vec)
        undist.append(undistort_point)
    undist = np.array(undist)       # shape: (len, 1, 1, 2)
    undist = undist.squeeze()       # shape: (len, 2)
    # print(f"undist: {undist.shape}")

    # triangulation
    A = []    # the matrix for triangulation
    for i in range(len(r)):
        # get the projection matrix
        P = np.hstack((r[i], t[i].T))       # shape: (3, 4)
        A.append(np.vstack((undist[i][0] * P[2, :] - P[0, :],
                            undist[i][1] * P[2, :] - P[1, :])))      

    A = np.array(A)     # shape A: (len, 2, 4)
    A = np.concatenate(A, axis=0)       # shape A: (len*2, 4)

    # SVD
    _, _, V = np.linalg.svd(A)      # shape V: (4, 4), V should be V^T in SVD decompose
    point_3d = V[-1, :]             # why not the 
    point_3d = point_3d / point_3d[-1]      # the last dim of point 3d corresponding 1, in 3D position[x, y, z, 1]
    return point_3d[:3]


# just the reverse calculation of triangulation, 
def reprojectToViews(points3d, r, t, K, RDist, TDist, view_num):
    # just use opencv cv2.projectPoints
    rvec = []
    for i in range(view_num):
        rvec.append(cv2.Rodrigues(r[i])[0])

    reprojected_points = []
    for i in range(view_num):
        # modify some params
        the_K = K[i]
        the_K[0, 1] = 0
        dist_coef = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], the_K, dist_coef)
        reprojected_points.append(reprojected_point)

    # the point shape: (view_num, 1, 2)
    reprojected_points = np.array(reprojected_points).squeeze()    # shape: (view_num, 2)
    return reprojected_points
