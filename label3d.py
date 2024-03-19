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

from utils.utils import CameraParams
from animator.animator import Animator, VideoAnimator, Keypoint3DAnimator

# is a GUI for manual labeling of 3D keypoints in multiple cameras.
class Label3D(Animator):
    def __init__(self, camParams=None, videos=None, skeleton_path=None, loader=None, *args, **kwargs) -> None:
        super().__init__()  
        

        # properties private
        self._origNFrames = 0
        self._initialMarkers = None
        self._isKP3Dplotted = False
        self._gridColor = (0.7, 0.7, 0.7)        # why not int?
        # self.
        self._labelPosition = []     # ?
        self._tablePosition = []     # ?
        self._instructions = []      # 
        self._statusMsg = []     # status message

        self._camPoints = None # 2D camera points for each frame, SHAPE: (#markers, #cams, 2, #frames)
        self._handlabeled2D = None # 2D hand labeled points only (subset of camPoints)

        self._hiddenAxesPos = []
        self._isLabeled = 2
        self._isInitialized = 1
        self._counter = None# total number of labeled frames
        self._sessionDatestr = None

        # properties public
        self.autosave = True
        self.clipboard = None
        self.origCamParams = None
        self.cameraParams = None       # cam params: intrinsic 内参
        self.markers = None
        self.camPoints = None       # 2D camera points for each frame. SHAPE: (#markers, #cams, 2, #frames)
        self.handLabeled2D = None   # 2D hand labeled points only (subset of camPoints) used to auto label

        self.points3D = None       # 3D points for frames. SHAPE: (#markers, 3, #frames)  # for all cameras got a 3D point
        self.status = None         # status message
        self.selectedNode = None   # ID of selected joint in joint table (use a radio button and key press to select)
        self.skeleton = None       # skeletons selected

        self.ImageSize = None
        self.nMarkers = None 
        self.nCams = None
        self.jointsPanel = None     # "panel" for joints window
        self.jointsControl = None   # "uicontrol" object for joints window

        self.savePath = ""
        self.kp3d = None          # 3D keypoint animator show the 3D keypoints
        self.statusAnimator = None  # animator for status heatmap window  # different from joints panel
        self.h = None              # cell of animators 
        self.verbose = True         # unused 

        self.undistortedImages = False  # undistorted images, if true, treat images as undistorted (aka, do not apply intrinsics to frame array)
        self.sync = None        # camera sync object 同步
        self.framesToLabel = None # frames number's to label: [1 x nFrames] (optional)
        self.videoPositions = None # x, y, width, height (origin = upper left corner) for each video window, shape: [nCams x 4]

        self.defScale = None  # global scale for image 
        self.pctScale = None  # scale images by this fraction  

        self.DragPointColor = [1, 1, 1]     # passed to DraggableKeypoint2DAnimator constructor
        self.visibleDragPoints = True       # passed to DraggableKeypoint2DAnimator constructor

        # self._init_from_scratch(camParams, videos, skeleton, *args, **kwargs)

    # constructors
    def _init_from_scratch(self, camParams, videos, skeleton, *args, **kwargs):
        # Animator parameters
        self.origCamParams = camParams
        self.nFrames = videos[0]
        self.origNFrames = self.nFrames
        self.frameInds = range(self.nFrames)
        self.nMarkers = len(skeleton)
        self.sessionDatestr = time.strftime("%Y%m%d-%H%M%S")
        self.savePath = []
        
        # Set up the cameras
        self.nCams = []
        self.h = []
        self.ImageSize = []
        [self.cameraParams, self.orientations, self.locations] = self.loadcamParams(self.origCamParams)
        self.cameraPoses = self.getCameraPoses()

        # make the GUI
        if self.videoPositions is None:
            self.videoPositions = self.getPositions(self.nCams)
        for nCam in self.nCams:
            pos = self.videoPositions[nCam]
            # self.h[nCam] =            # 原文：obj.h{nCam} = VideoAnimator(videos{nCam}, 'Position');
            

        self.camParams = camParams
        self.videos = videos
        self.skeleton = skeleton
        self.construct_animators()

        self.arrange_videos()

        self.initGUI()



    # 暂时不会用的方法
    def _init_load_state(self, file, videos, *args, **kwargs):
        pass    # file: path to saved label3d state file (with or without videos)
        # videos: list of h x w x c x nFrames videos
    def _init_load_file(self, file, *args, **kwargs):
        pass    # file: path to saved label3d state file (with videos)
    def _init_gui(self, *args, **kwargs):
        pass    # ???
    @classmethod
    def load_and_merge(cls, file, *args, **kwargs):
        # file: cell array of paths to saved label3d state files (with videos)
        return cls(file=file, *args, **kwargs)
    

    @staticmethod
    def construct_animators(self, ):
        
        pass 


    # set the init GUI, 
    # 1. set the layout of the GUI
    # add a radio button to select the joint, in a additional window or in the main window
    def initGUI(self, ):
        animators = self.getAnimators()
        pass 


    # methods 
    def loadcamParams(self, camParams):
        pass

    def getCameraPoses(self, ):
        # return table of camera poses
        pass

    # 
    def buildFromScratch(self, camParams, videos, skeleton):

        return

    # set the position of each video animator    
    def positionFromNRows(self, nViews, nRows):
        # get position with views number and nRows, not useful
        # nRows 表示 每行视图个数
        len = np.ceil(nViews / nRows)
        pos = np.zeros((nViews, 4))
        for i in range(nViews):
            row = np.floor(i / nRows)
            col = i % nRows
            pos[i, :] = [col / len, 1 - (row + 1) / nRows, 1 / len, 1 / nRows]
        # pos 是 画面分割的比例 y, x, h, w
    
    def getPositions(self, nViews=None):
        # get the axes positions of each camera view
        # inputs: nViews - number of views
        if nViews is None:
            nViews = self.nCams
        nRows = np.floor(np.sqrt(nViews))
        if nViews > 3:
            pos = self.positionFromNRows(nViews, nRows)
        else:
            pos = self.positionFromNRows(nViews, 1)
        return pos

    def getAnimators(self, ):
        # get the animators
        # including the video animators and one 3D keypoint animator
        
        pass

    def loadcamParams(self, camParams):
        # Helper to load in camera params into cameraParameters into cameraParameters objects
        #  and 保存世界方向和世界位置
        # inputs: camParams - cell array of camera parameters structs
        # outputs: [c, orientations, locations] - camera parameters, orientations, and locations

        (c, orientations, locations) = [], [], []
        
        for i in range(len(camParams)):
            K = camParams[i]["K"]
            RDistort = camParams[i]["RDistort"]
            TDistort = camParams[i]["TDistort"]
            r = camParams[i]["r"]
            rotationVector = rotationMatrixToVector(r)
            translationVector = camParams[i]["t"]

            c.append({
                'IntrinsicMatrix': K, 
                'ImageSize': self.imageSize,
                'RotationVector': rotationVector, 
                'TranslationVector': translationVector, 
                'RadialDistortion': RDistort, 
                'TangentialDistortion': TDistort})
            # 没有 cameraParameters 这个类
        
            orientation = np.transpose(r)                           # MATLAB CODE: orientation = r'
            orientations.append(orientation)           
            location = -np.dot(translationVector, orientation)      # MATLAB CODE: location = -translationVector * orientation
            locations.append(location)

        return [c, orientations, locations]
    
    def getCameraPoses(self, ):
        # Helper function to store the camera poses for triangulation
        view_ids = np.arange(1, self.nCams + 1)
        
        # 保存相机姿态
        cameraPoses = pd.DataFrame({
            'ViewId': view_ids,
            'Orientation': self.orientations,
            'Location': self.locations
        })
        # ignored matlab code:
        # for i = 1:obj.nCams
        #    cameraPoses(i).Location = obj.locations{i};
        # end
        return cameraPoses

    def triangulateView(self, ):
        # Triangulate labeled points and zoom all images around the those points    :need zoom?
        
        # Make sure there is at least one point could be triangulated
        frame = self.frame
        meanPts = np.mean(self.camPoints[:, :, :, frame], axis=1)
        # get points number
        if meanPts.shape[0] < 2:
            return
        
        # get all camera intrinsics
        intrinsics = self.cameraParams  # TODO: check the data saving format 
        validCams = np.where(~np.isnan(meanPts[:, 0]))[0]
        pointTrackers = pointTrack()

        # zoom
        # if a global scale has been defined, use it. Otherwise use a percentage of the image size
        # 作用不明...
        
        return 
    
    def getLabeledJoints(self, frame):
        # Label3D: Look within a frame and return all joints with at least two labeled views, 
        # as well as a logical vector denoting which two views.  # 两个视图的逻辑向量, 两视图间是否相连
        s = np.zeros((self.nMarkers, self.nCams))
        for i in range(self.nMarkers):
            s[i, :] = ~np.isnan(self.camPoints[i, :, 0, frame])
        return 
    

    def getPointTrack(self, frame, jointId, camIds):
        # in Label3D, return a pointTrack object storing the 2D points for a given joint and frame

        return

    def forceTriangulateLabeledPoints(self, ):
        # 
        return 

    def pointTrack(self, ):
        # get the point 
        pass 

    # 按键事件
    def keyPressEvent(self, event): 
        # inherited from Animator
        if event.key() == Qt.Key_H:
            print("Help message: ")
        elif event.key() == Qt.Key_Backspace:
            # delete the selected joint
            pass
        elif event.key() == Qt.Key_T:
            # auto label # main function 
            pass 
        # elif event.key() == Qt.Key_


        return 
    
    # 鼠标事件
    def mousePressEvent(self, event):
        # inherited from VideoAnimator
        # return position and draw a point
        
        return


    @staticmethod
    def triangulateMultiview(pointTrack, cameraPoses, intrinsics):
        # Triangulate points from multiple views
        # use a for loop to deal with each view
        
        return 
    
    
# helper functions for triangulation   
def rotationMatrixToVector(R):
    # convert rotation matrix to rotation vector
    # inputs: R - 3x3 rotation matrix
    # outputs: rVec - 3x1 rotation vector
    rVec = cv2.Rodrigues(R)[0]
    return rVec



class pointTrack():
    def __init__(self, ):
        pass 


# the triangulate function
def triangulateMultiview(pointTrack, cameraPoses, intrinsics):
    # the point after intrinsics
    # point track: 2D points (points pos from scene) for a given joint of all views. SHAPE: (#cams, 2)
    # cameraPoses: camera poses for all views (rotation and translation). SHAPE: (#cams, (3, 3), (3, 1))    # TODO: CHECK THE SHAPE
        # cameraPoses should saved in a dictor or a tuple
    # intrinsics: camera intrinsics for all views. SHAPE: (#cams, 3, 3)
    
    assert len(pointTrack) == len(cameraPoses) == len(intrinsics), "The number of cameras should be the same"

    # get the point after intrinsics
    points2d = []
    for i in range(len(pointTrack)):
        points2d.append(cv2.undistortPoints(pointTrack[i], intrinsics[i], None))        # ??

    # calculate the 3D points
    A = []
    for point, cameraPose in zip(points2d, cameraPoses):
        point_array = np.array([point[0], point[1], 1])
        projection_matrix = np.array([cameraPose[0], cameraPose[1], cameraPose[2]])     # projection matrix = [R | t], R is 3x3, t is 3x1, so the shape is (3, 4)

    the_matrix = np.dot(np.transpose)

    # USE eignvalue and eignvector to get x, y, z
        


    xyzPoints = []
    return xyzPoints


# an event filter
class KeyPressEater(QObject):
    def __init__(self, widgets):
        super().__init__()
        self.widgets = widgets

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            for widget in self.widgets:
                # widget.keyPressEvent(event)       # 直接调用keyPressEvent方法
                if widget != obj:
                    QCoreApplication.sendEvent(widget, QKeyEvent(QEvent.KeyPress, event.key(), event.modifiers(), event.text(), event.isAutoRepeat(), event.count()))
                    # sendEvent vs postEvent
            return True
        return False