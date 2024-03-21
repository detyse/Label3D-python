from PySide6.QtGui import QWheelEvent
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

    # like there is no need to handle the key press event in this class
    # def keyPressEvent(self, event):     # the frame operation, will be used in all animators
    #     # the maximum frame rate is 60
    #     if event.key() == Qt.Key_Left:
    #         self.frame += self.frameRate
    #         if self.frame >= self.nFrames:
    #             self.frame = 0
        
    #     elif event.key() == Qt.Key_Right:
    #         self.frame -= self.frameRate
    #         if self.frame < 0:
    #             self.frame = self.nFrames - 1

    #     elif event.key() == Qt.Key_Up:
    #         self.frameRate += self.speedUp
    #         self.frameRate = min(self.frameRate, 60)

    #     elif event.key() == Qt.Key_Down:
    #         self.frameRate -= self.slowDown
    #         self.frameRate = max(1, self.frameRate)

        # then implement the frame change function


class VideoAnimator(Animator):
    def __init__(self, video_path, skeleton):
        super().__init__()
        # load properties from input
        self.video_path = video_path
        self.video_frames = self.read_video()
        
        # update properties inherited from animator
        self.nFrames = len(self.video_frames)
        self.frame = 0      # the index start from 0
         # other properties as default  # self.frameRate = 1
        
        # properties for 2d keypoint animator
        self._joint_names = skeleton["joint_names"]
        self._joints_idx = skeleton["joints_idx"]        # the connection of joints, joints are indicated by index + 1
        self._color = skeleton["color"]              # the color of the joints
        self.joints_num = len(self._joint_names)
        
        # on this frame, f for only in this frame
        self.f_current_joint_idx = None
        self.f_exist_markers = []           # a list of joints idx indicating the joints on this frame       
        self.f_joints2markers = {}            # a dict map the joint name to the marker on this frame

        # 
        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)

        # d for data, constant for item data saving
        # self.d_joint_name = 0
        self.d_joint_index = 0
        self.d_lines = 1

        # trivial properties
        self.marker_size = 10       # the size of the point

        self.initUI()
        self._initView()
        self.update_frame()

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

    def _initView(self, ):      # TODO: inhibit the scroll move
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

    ## 
    def set_joint(self, joint_idx):
        self.f_current_joint_idx = joint_idx

    def set_marker_2d(self, pos, frame=None, joint_idx=None):       # the params are unnecessary
        # if frame is not None:
        #     self.update_frame(frame)

        if joint_idx is not None:
            self.set_joint(joint_idx)

        self.plot_marker_and_lines(pos)

    def get_marker_2d(self, frame=None, joint_idx=None):
        # if frame is not None:
        #     self.update_frame(frame)

        if joint_idx is not None:
            self.set_joint(joint_idx)

        return self.frames_markers[self.frame, self.f_current_joint_idx]

    def update_frame(self, frame_ind=None):       # this function should be use after the frame change, also used to init the scene        
        # clear the scene, but the view would change
        self.scene.clear()
        
        # frame_ind: the index of video frames, 
        # update frame and using self.frame as current frame
        if frame_ind is not None:
            if self.frame == frame_ind:
                return
            
            self.frame = frame_ind

        # make sure all the f start properties are reset
        self.f_current_joint_idx = None
        self.f_exist_markers = []
        self.f_joints2markers = {}
        
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
                self.plot_marker_and_lines(pos, i)      # update the exist markers

        self.scene.update()

    # this function should be called whenever the marker will change
    # if the marker is not exist, create a new marker; else update the marker
    def plot_marker_and_lines(self, pos, joint_idx=None, ):
        if joint_idx is None:
            if self.f_current_joint_idx is not None:
                current_index = self.f_current_joint_idx
            else:
                # just do nothing
                return
        else:
            current_index = joint_idx
            # self.f_current_joint_idx = joint_idx      # should not change when frame change

        if current_index in self.f_exist_markers:
            self.reset_marker(self.f_joints2markers[current_index], pos)
            return

        # set the point
        marker = QGraphicsEllipseItem(int(-self.marker_size), int(-self.marker_size), self.marker_size, self.marker_size)
        marker.setPos(pos[0], pos[1])

        brush = QBrush(color2QColor(self._color[current_index+1]))
        brush.setStyle(Qt.SolidPattern)

        marker.setBrush(brush)
        # marker.setFlag(QGraphicsItem.ItemSendsScenePositionChanges, True)
        marker.setData(self.d_joint_index, current_index)
        marker.setData(self.d_lines, [])

        self.scene.addItem(marker)
        
        for index, (i, j) in enumerate(self._joints_idx):
            the_color = self._color[index]
            if current_index == i-1:
                if j-1 in self.f_exist_markers:
                    the_point = self.f_joints2markers[j-1]
                    # draw the line
                    the_line = Connection(marker, the_point, the_color)
                    
                    i_lines = marker.data(self.d_lines)
                    i_lines.append(the_line)
                    marker.setData(self.d_lines, i_lines)
                    
                    j_lines = the_point.data(self.d_lines)
                    j_lines.append(the_line)
                    the_point.setData(self.d_lines, j_lines)
                    
                    self.scene.addItem(the_line)

            elif current_index == j-1:
                if i-1 in self.f_exist_markers:
                    the_point = self.f_joints2markers[i-1]
                    the_line = Connection(the_point, marker, the_color)
                    
                    i_lines = the_point.data(self.d_lines)
                    i_lines.append(the_line)
                    the_point.setData(self.d_lines, i_lines)
                    
                    j_lines = marker.data(self.d_lines)
                    j_lines.append(the_line)
                    marker.setData(self.d_lines, j_lines)

                    self.scene.addItem(the_line)

        # update the exist markers
        self.f_exist_markers.append(current_index)

        self.frames_markers[self.frame, current_index] = pos
        self.f_joints2markers[current_index] = marker

    def reset_marker(self, item, new_pos):
        item.setPos(*new_pos)

        print(item.data(self.d_lines))
        print(item.data(self.d_joint_index))


        for line in item.data(self.d_lines):
            # reset lines
            line.updateLine(item)
        
        # self.f_joints2markers[item.data(self.d_joint_index)] = item
        self.frames_markers[self.frame, self.f_current_joint_idx] = new_pos

    def delete_marker(self, joint_idx=None):
        if joint_idx is None:
            if self.f_current_joint_idx is not None:
                current_index = self.f_current_joint_idx
            else:
                # just do nothing
                return
            
        else:
            current_index = joint_idx

        if current_index in self.f_exist_markers:
            # remove the lines
            for line in self.f_joints2markers[current_index].data(self.d_lines):
                self.scene.removeItem(line)

            self.scene.removeItem(self.f_joints2markers[current_index])
            self.f_exist_markers.remove(current_index)
            self.f_joints2markers.pop(current_index)
            self.frames_markers[self.frame, current_index] = np.nan


    ## 
    def keyPressEvent(self, event):
        # ignore all the key press event, leave it to the parent widget
        # except sevel key press event
        event.ignore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            pos = event.pos()
            pos = self.view.mapToScene(pos)
            self.plot_marker_and_lines([pos.x(), pos.y()])
        elif event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            pos = event.pos()
            pos = self.view.mapToScene(pos)
            item = self.scene.itemAt(pos, self.view.transform())
            if isinstance(item, QGraphicsEllipseItem):
                self.delete_marker(item.data(self.d_joint_index))

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.view.scale(1.1, 1.1)
        else:
            self.view.scale(0.9, 0.9)

# utils
class Connection(QGraphicsLineItem):
    def __init__(self, start_point, end_point, color, shift=5):
        super().__init__()
        self.shift = shift      # to meet the marker center

        self.start_point = start_point
        self.end_point = end_point
        # print("line points", start_point.scenePos(), end_point.scenePos())

        self._line = QLineF(start_point.scenePos(), end_point.scenePos())
        self.setLine(self._line)

        # some defualt properties
        self.setSelected(False)
        
        the_color = QColor(color2QColor(color))
        self.setPen(QPen(the_color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # ...

    def updateLine(self, source):       # 
        if source == self.start_point:
            self._line.setP1(source.scenePos())
        elif source == self.end_point:
            self._line.setP2(source.scenePos())
        else:
            raise ValueError("source should be the start or end point")
        self.setLine(self._line)


def color2QColor(color):
    return QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))


    