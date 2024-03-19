import cv2
import numpy as np

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *



class Animator(QWidget):  # inherit from QWidget
    '''
    A base class for all animators, mainly for **frame operation** which will used in all animator instances
    Also, it will be used for the sync of all animators
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.nFrames = 1        # total number of frames
        self.frame = 0          # current frame

        self.frameRate = 1

        self.speedUp = 5
        self.slowDown = 5

        # 没有 大跨度翻页的需求

        self.initUI()


    # the animators need show the frame, use this method to init a scene
    def initUI(self, ):
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)


    def keyPressEvent(self, event):     # the frame operation, will be used in all animators
        # the maximum frame rate is 60
        if event.key() == Qt.Key_Left:
            self.frame += self.frameRate
            if self.frame >= self.nFrames:
                self.frame = 0
        
        elif event.key() == Qt.Key_Right:
            self.frame -= self.frameRate
            if self.frame < 0:
                self.frame = self.nFrames - 1

        elif event.key() == Qt.Key_Up:
            self.frameRate += self.speedUp
            self.frameRate = min(self.frameRate, 60)

        elif event.key() == Qt.Key_Down:
            self.frameRate -= self.slowDown
            self.frameRate = max(1, self.frameRate)

        # then implement the frame change function


class VideoAnimator(Animator):
    def __init__(self, video_path, skeleton):
        super().__init__()
        # load properties from input
        self.video_path = video_path
        self.video_frames = self.read_video()
        
        # update properties inherited from animator
        self.nFrames = len(self.frames)
        self.frame = 0      # the index start from 0
         # other properties as default  # self.frameRate = 1
        
        # properties for 2d keypoint animator
        self._joint_names = skeleton["joint_names"]
        self._joints_idx = skeleton["joints_idx"]        # the connection of joints, joints are indicated by index + 1
        self._color = skeleton["color"]              # the color of the joints
        self.joints_num = len(self._joint_names)
        
        # on this frame
        self.f_current_joint_idx = None
        self.f_exist_markers = []           # a list of joints idx indicating the joints on this frame       
        self.joints2markers = {}            # a dict map the joint name to the marker on this frame

        # 
        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)

        # constant for item data saving
        self.d_joint_name = 0       # d for data
        self.d_joint_index = 1
        self.d_lines = 2

        # trivial properties
        self.marker_size = 10       # the size of the point

    def read_video(self, video_path=None):
        if video_path is None:
            video_path = self.video_path
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        cap.release()
        return frames

    def initUI(self, ):
        super().initUI()

    def _initView(self, ):
        # set the QGraphicsView to show the frame
        layout = QVBoxLayout()

        self.view.setMouseTracking(True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)         # find a better solution for drag mode

        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.view.setRenderHint(QPainter.Antialiasing)              # 去锯齿
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)     # 
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)           # transformation do what?
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)                   # 

        layout.addWidget(self.view)
        self.setLayout(layout)


    def scene_update(self, frame_ind=None):       # this function should be use after the frame change, also used to init the scene        
        # clear the scene
        self.scene.clear()
        
        # frame_ind: the index of video frames
        if frame_ind is not None:
            self.frame = frame_ind

        current_frame = self.video_frames[self.frame]
        height, width, channels = current_frame.shape       # the frame shape would change
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        self.scene.addPixmap(pixmap)

        # plot the labeled joint markers and lines on this frame
        for i in range(self.joints_num):
            if not np.isnan(self.frames_markers[self.frame, i]).all():
                pos = self.frames_markers[self.frame, i]
                self.plot_joint_marker_and_lines(pos, i)

        self.scene.update()


    # def plot_joint_marker_and_lines(self, pos, joint_idx=None, ):
        
    