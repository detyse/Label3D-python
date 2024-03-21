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
        
        self._init_property()

        self.initGUI()

        self.align_animators()

        self.unpack_camParams()


    def unpack_camParams(self, ):
        r = []
        t = []
        K = []
        RDist = []
        TDist = []
        for cam in self.camParams:
            r.append(cam["r"][0][0])                # shape: (3, 3), rotation matrix
            t.append(cam["t"][0][0])                # shape: (1, 3), translation vector
            K.append(cam["K"][0][0].T)              # shape: (3, 3), intrinsic matrix, need to transpose to fit the shape
            RDist.append(cam["RDistort"][0][0])     # shape: (1, 3), corresponding to k1, k2, k3
            TDist.append(cam["TDistort"][0][0])     # shape: (1, 2), corresponding to p1, p2
        
        self.r = np.array(r)
        self.t = np.array(t)
        self.K = np.array(K)
        self.RDist = np.array(RDist)
        self.TDist = np.array(TDist)
 

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

        # 3d joints
        self.joints3d = np.full((self.nFrames, len(self._joint_names), 3), np.nan)        # 3d joints, x, y, z


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
    def align_animators(self, ):
        # check the frame or align the frame
        for i in range(len(self.video_animators)):
            assert self.video_animators[i].nFrames == self.video_animators[0].nFrames, "The frame number of videos must be the same"
            assert self.video_animators[i].frame == self.video_animators[0].frame, "The frame index of videos must be the same"
        
        # set label3d frame property
        self.frame = self.video_animators[0].frame        
        self.nFrames = self.video_animators[0].nFrames
        self.frameInd = np.arange(self.nFrames)
        # TODO: should use restrict function to update the frame property



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
        if event.key() == Qt.Key_F:
            print("S is pressed in main window")
            # next frame
            self.frame += 1 if self.frame < self.nFrames - 1 else 0
            self.frame_update()

        elif event.key() == Qt.Key_B:
            self.frame -= 1 if self.frame > 0 else 0
            self.frame_update()

        # triangulate the current frame, current joint and reproject to all views
        elif event.key() == Qt.Key_T:
            self.triangulate_to_3d()
            reprojection_points = self.reproject_to_all_views()
            for i, animator in enumerate(self.video_animators):
                animator.receive_reprojection(reprojection_points[i])
            # update point position
        # update the frame for all the animators
                
        elif event.key() == Qt.Key_P and event.modifiers() == Qt.ControlModifier:
            self.save_3d_joints("3d_joints.csv")
        
    # for all animators, not only change the frame, but the exist markers
    def frame_update(self, ):
        for animator in self.video_animators:
            animator.frame = self.frame
            animator.frame_update()
        

    def triangulate_to_3d(self, ):        # trianglate this frame
        # get the 2d joints of frames from all views
        # current exist points
        frame_view_markers = np.full((self.view_num, len(self._joint_names), 2), np.nan)
        for i, animator in enumerate(self.video_animators):
            frame_view_markers[i, self.current_joint_idx] = animator.frames_markers[self.frame, self.current_joint_idx]

        # at least two views
        view_avaliable = np.zeros(self.view_num)
        for i in range(self.view_num):
            if not np.isnan(frame_view_markers[i, self.current_joint_idx]).all():
                view_avaliable[i] = 1

        if view_avaliable.sum() < 2:
            print("At least two views are needed to trianglate 3d joints")
            return
        else:
            # get variables for triangulation
            points2d = []
            camera_poses = []
            intrinsics = []
            for avaliable, points, cam_pose, intr in zip(view_avaliable, frame_view_markers, self.camParams, self.camParams):
                if avaliable == 1:
                    points2d.append(points)
                    camera_poses.append(cam_pose)
                    intrinsics.append(intr)
            point_3d = triangulateMultiview(points2d, self.r, self.t, self.K, self.RDist, self.TDist)
            self.joints3d[self.frame, self.current_joint_idx] = point_3d

    
    def reproject_to_all_views(self, ):
        # get the 3d joints of this frame and joint
        points3d = self.joints3d[self.frame, self.current_joint_idx]
        if np.isnan(points3d).all():
            print("No 3d points are available for reprojecting")
            return
        else:
            pass

        # return 

    def save_3d_joints(self, save_name):
        return self.joints3d 
    

# function for multiview triangulation
# could handle various number of views

# ref: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
# ref: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
# ref: https://gutsgwh1997.github.io/2020/03/31/%E5%A4%9A%E8%A7%86%E5%9B%BE%E4%B8%89%E8%A7%92%E5%8C%96/
def triangulateMultiview(points2d, r, t, K, RDist, TDist):
    # 
    undist = []
    for i in range(len(points2d)):
        point = points2d[i]
        dist_vec = [RDist[i][0], RDist[i][1], TDist[i][0], TDist[i][1], RDist[i][2]]
        undistort_point = cv2.undistortPoints(point, K, dist_vec)
        undist.append(undistort_point)

    # triangulation
    A = []    # the matrix for triangulation
    for i in range(len(r)):
        # get the projection matrix
        P = np.hstack((r[i], t[i].T))       # shape: (3, 4)
        A.append(np.vstack((undist[i][0, 0] * P[2, :] - P[0, :],
                            undist[i][0, 1] * P[2, :] - P[1, :])))      

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
        dist_coef = [RDist[i][0], RDist[i][1], TDist[i][0], TDist[i][1], RDist[i][2]]
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], K[i], dist_coef)
        reprojected_points.append(reprojected_point)

    # the point shape: (view_num, 1, 2)
    return reprojected_points