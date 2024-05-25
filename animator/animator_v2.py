# animator for video display

from PySide6.QtGui import QEnterEvent, QMouseEvent, QWheelEvent
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
        self.view = SceneViewer(self.scene)

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
        self.update_frame()
        self._initView()


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

        # self.view.setMouseTracking(True)
        # # self.view.setDragMode(QGraphicsView.ScrollHandDrag)         # find a better solution for drag mode, 

        # # and change the mouse cursor to arrow
        # self.view.setCursor(Qt.ArrowCursor)             # the cursor should be arrow but it is not working      

        # self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # self.view.setRenderHint(QPainter.Antialiasing)              # 去锯齿
        # self.view.setRenderHint(QPainter.SmoothPixmapTransform)     # 
        # self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)           # transformation do what?
        # self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)                   # 以鼠标为中心 resize

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
        # frame_ind: the index of video frames, 
        # update frame and using self.frame as current frame
        if frame_ind is not None:
            if self.frame == frame_ind:
                return
            
            self.frame = frame_ind
        
        # clear the scene, but the view would not change
        self.scene.clear()

        # make sure all the f start properties are reset
        self.f_current_joint_idx = None
        self.f_exist_markers = []
        self.f_joints2markers = {}
        
        current_frame = self.video_frames[self.frame]
        height, width, channels = current_frame.shape       # the frame shape would change
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # resize the image to fit the view
        pixmap = pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

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
        item.setPos(*new_pos)       # 

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
            pos = self.view.mapToScene(pos)     # get the position
            self.plot_marker_and_lines([pos.x(), pos.y()])
            print(f"plot point position: {pos.x()}, {pos.y()}")
        elif event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            pos = event.pos()
            pos = self.view.mapToScene(pos)
            item = self.scene.itemAt(pos, self.view.transform())
            if isinstance(item, QGraphicsEllipseItem):
                self.delete_marker(item.data(self.d_joint_index))

    # def wheelEvent(self, event):
    #     if event.angleDelta().y() > 0:
    #         self.view.scale(1.1, 1.1)
    #     else:
    #         self.view.scale(0.9, 0.9)


# # utils
# # some overrided helper classes
# # work for canceling the scroll bar and make the mouse wheel event work and change the cursor
class SceneViewer(QGraphicsView):
    '''
    This class override some methods of QGraphicsView,
    to make the mouse wheel event work and change the cursor,
    also fit the image to the view, engage the lines and points
    '''
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        # self.setDragMode(QGraphicsView.ScrollHandDrag)         # find a better solution for drag mode,
        self.setDragMode(QGraphicsView.NoDrag)    
        
        # and change the mouse cursor to arrow
        self.setCursor(Qt.ArrowCursor)             # the cursor should be arrow but it is not working      

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.setRenderHint(QPainter.Antialiasing)              # 去锯齿
        self.setRenderHint(QPainter.SmoothPixmapTransform)     # 
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)           # transformation do what?
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)                   # 以鼠标为中心 resize

        self._dragging = False
        self._drag_start_pos = None
        self._transform = self.transform()

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            self.scale(1.1, 1.1)
        else:
            self.scale(0.9, 0.9)

    def enterEvent(self, event: QEnterEvent) -> None:
        super().enterEvent(event)
        self.setCursor(Qt.ArrowCursor)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.pos()
            self._transform_start = self.transform()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.pos() - self._drag_start_pos
            translate_x = delta.x()
            translate_y = delta.y()
            self.setTransform(self._transform_start)
            self.translate(translate_x, translate_y)
        super().mouseMoveEvent(event)
        self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


# NOTE: meet some error when using the auto triangle, check if the error comes from here
# 
class Connection(QGraphicsLineItem):        # the line is not necessarily combined with the points, you do not return, so the 
    def __init__(self, start_point, end_point, color, shift=5):
        super().__init__()
        self.shift = shift      # to meet the marker center, pass the shift from somewhere

        self.start_point = start_point
        self.end_point = end_point
        # print("line points", start_point.scenePos(), end_point.scenePos())

        print(f"type of the points of the line {type(start_point)}, {type(end_point)}")

        self._line = QLineF(start_point.mapToScene(-self.shift, -self.shift), end_point.mapToScene(-self.shift, -self.shift))
        self.setLine(self._line)

        # some defualt properties
        self.setSelected(False)
        
        the_color = QColor(color2QColor(color))
        self.setPen(QPen(the_color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # ...

    def updateLine(self, source):       # 
        # source position
        if source == self.start_point:
            self._line.setP1(source.mapToScene(-self.shift, -self.shift))
        elif source == self.end_point:
            self._line.setP2(source.mapToScene(-self.shift, -self.shift))
        else:
            raise ValueError("source should be the start or end point")
        self.setLine(self._line)

# class Connection(QGraphicsLineItem):        # the line is not necessarily combined with the points, you do not return, so the 
#     def __init__(self, start_point, end_point, color, shift=5):
#         super().__init__()
#         self.shift = shift      # to meet the marker center, pass the shift from somewhere

#         self.start_point = start_point
#         self.end_point = end_point
#         # print("line points", start_point.scenePos(), end_point.scenePos())

#         print(f"type of the points of the line {type(start_point)}, {type(end_point)}")

#         self._line = QLineF(start_point.mapToScene(0, 0), end_point.mapToScene(0, 0))
#         self.setLine(self._line)

#         # some defualt properties
#         self.setSelected(False)
        
#         the_color = QColor(color2QColor(color))
#         self.setPen(QPen(the_color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
#         # ...

#     def updateLine(self, source):       # 
#         # source position
#         if source == self.start_point:
#             self._line.setP1(source.mapToScene(0, 0))
#         elif source == self.end_point:
#             self._line.setP2(source.mapToScene(0, 0))
#         else:
#             raise ValueError("source should be the start or end point")
#         self.setLine(self._line)

def color2QColor(color):
    return QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))
