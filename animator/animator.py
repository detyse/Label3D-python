# animator for video display

from PySide6.QtGui import QEnterEvent, QMouseEvent, QWheelEvent
import cv2
import numpy as np
import os

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


# 这个可以有，这个真没有
class Keypoint3DAnimator(Animator):
    def __init__(self, ):
        super().__init__()



## TODO: add the scale factor for currect the transformation 
class VideoAnimator(Animator):                  
    def __init__(self, video_paths, skeleton, label_num):
        super().__init__()
        # load properties from input
        # self.video_paths = video_paths      # a list of video path
        # self.video_frames = self.load_videos(self.video_paths, label_num)
        self.video_frames = self.load_videos(video_paths, label_num)

        # update properties inherited from animator
        self.nFrames = len(self.video_frames)
        self.frame = 0      # the index start from 0
        # other properties as default  # self.frameRate = 1
        
        # print(skeleton)
        # properties for 2d keypoint animator
        self._joint_names = skeleton["joint_names"]
        self._joints_idx = skeleton["joints_idx"]        # the connection of joints, joints are indicated by index + 1
        self._color = skeleton["color"]              # the color of the joints
        self.joints_num = len(self._joint_names)        # total number of joints
        
        # on this frame, f for only in this frame
        self.f_current_joint_idx = None
        self.f_exist_markers = []           # a list of joints idx indicating the joints on this frame       
        self.f_joints2markers = {}            # a dict map the joint name to the marker on this frame
        # will these change after reprojection? if not we could update the frame after the reprojection

        # 
        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)
        # reproject allowed
        self.original_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)
        # check the last mouse press event position and update the value

        # d for data, constant for item data saving
        # self.d_joint_name = 0
        self.d_joint_index = 0
        self.d_lines = 1

        # trivial properties
        self.marker_size = 10       # the size of the point

        self.initUI()
        self.update_frame()
        self._initView()

    
    # consider of just output a list of frames, we just using a function to do the job
    def load_videos(self, video_file_list, label_num):
        '''
        the frame number not aligned situation is not considered
        it is should be fine with a single video(which is the normal situation)
        and assume all the videos want the same frame number

        comment: pretty slow to load the video,
                    test the other method. *grab and retrieve*
        '''
        '''
        add a new condition to load image data as frames for labeling
        '''
        # get the file type, if the file is a directory, then load the images
        if os.path.isdir(video_file_list[0]):
            frames = []
            for image_folder in video_file_list:
                image_list = os.listdir(image_folder)
                image_list.sort()
                for image_file in image_list:
                    frame = cv2.imread(image_file)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)

            print(f"load image frame number {len(frames)}")
            return frames
        
        # else if the file ends with npy, then load the frames
        elif video_file_list[0].endswith(".npy"):
            frames = None
            for npy_file in video_file_list:
                frame = np.load(npy_file)
                print(f"frame shape: {frame.shape}")
                if frames is None:
                    frames = frame
                else:
                    frames = np.concatenate((frames, frame), axis=0)
            
            frames = list(frames)
            print(f"load npy frame number {len(frames)}")
            return frames

        # else if the file ends with mp4 or avi, then load the video
        elif video_file_list[0].endswith(".mp4") or video_file_list[0].endswith(".avi"):
            frame_num_list = []
            frame_index_list = []
            frames = []
            for video_file in video_file_list:
                cap = cv2.VideoCapture(video_file)
                frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_num_list.append(frame_num)
                
                if label_num == 0 or label_num > frame_num:
                    label_num = frame_num

                indexes = np.linspace(0, frame_num-1, label_num, dtype=int)
                frame_index_list.append(indexes)

                # if the frame number is index, then get the frame
                for index in indexes:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)

                    ret = cap.grab()
                    if not ret:
                        break
                    ret, frame = cap.retrieve()
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # 
                    
                    frames.append(rgb_frame)
                cap.release()
                
            print(f"load video frame number {len(frames)}")
        return frames


    def initUI(self, ):
        # set cursor
        self.setMouseTracking(True)
        self.setCursor(Qt.ArrowCursor)
        super().initUI()


    def _initView(self, ):      # TODO: inhibit the scroll move, move the view to the 
        # set the QGraphicsView to show the frame
        layout = QVBoxLayout()

        layout.addWidget(self.view)
        self.setLayout(layout)

    def load_labels(self, frames_markers, original_markers):
        self.frames_markers = frames_markers
        self.original_markers = original_markers
        self.update_frame()

    ## 
    def set_joint(self, joint_idx):
        self.f_current_joint_idx = joint_idx


    def set_marker_2d(self, pos, frame=None, joint_idx=None, reprojection=False):       # the params are unnecessary
        # if frame is not None:
        #     self.update_frame(frame)

        if joint_idx is not None:
            self.set_joint(joint_idx)

        self.plot_marker_and_lines(pos, reprojection=reprojection)


    def get_marker_2d(self, frame=None, joint_idx=None):
        # if frame is not None:
        #     self.update_frame(frame)

        if joint_idx is not None:
            self.set_joint(joint_idx)

        return self.frames_markers[self.frame, self.f_current_joint_idx]
    

    def get_all_original_marker_2d(self, ):         # return the original labeled markers of current frame
        return self.original_markers[self.frame]


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
        # print(f"frame shape: {height}, {width}, {channels}")
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        # check the view size
        # print(f"view size: {self.view.size()}")

        # resize the image to fit the view
        # pixmap = pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        # check the pixmap width and height
        # print(f"pixmap size: {pixmap.width()}, {pixmap.height()}")

        self.scene.addPixmap(pixmap)

        # get the scene rect 
        the_rect = self.scene.sceneRect()
        self.view.fitInView(the_rect, Qt.KeepAspectRatio)       # fit the image to the view

        # update the scale factor
        # self._scale_factor = self.view.transform().m11()
        # just return the scale factor?
        # self._scale_factor = self.view.scale()

        # plot the labeled joint markers and lines on this frame
        for i in range(self.joints_num):
            if not np.isnan(self.frames_markers[self.frame, i]).all():
                pos = self.frames_markers[self.frame, i]
                self.plot_marker_and_lines(pos, i, reprojection=False)      # update the exist markers

        self.scene.update()


    # this function should be called whenever the marker will change
    # if the marker is not exist, create a new marker; else update the marker
    # TODO: check the dot pos() and the mouse pos()
    def plot_marker_and_lines(self, pos, joint_idx=None, reprojection=False):
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
        marker = QGraphicsEllipseItem(int(-self.marker_size//2), int(-self.marker_size//2), self.marker_size, self.marker_size)
        marker.setPos(pos[0], pos[1])
        # print(f"marker position: {pos[0]}, {pos[1]}")

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
        if not reprojection:
            self.original_markers[self.frame, current_index] = pos      # 

        self.f_joints2markers[current_index] = marker


    def reset_marker(self, item, new_pos, reprojection=False):
        item.setPos(*new_pos)       # 

        for line in item.data(self.d_lines):
            # reset lines
            line.updateLine(item)
        
        # self.f_joints2markers[item.data(self.d_joint_index)] = item
        self.frames_markers[self.frame, self.f_current_joint_idx] = new_pos
        if not reprojection:
            self.original_markers[self.frame, self.f_current_joint_idx] = new_pos


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
            # pos = event.pos()           # the event pos relative to the widget, there is a position shift between the widget and the view
            # get the true position in the view
            view_pos = self.view.mapFromGlobal(self.mapToGlobal(event.pos()))
            scene_pos = self.view.mapToScene(view_pos)
            print(f"mouse position in the view: {view_pos.x()}, {view_pos.y()}")
            # pos = self.view.mapToScene(pos)     # get the position
            self.plot_marker_and_lines([scene_pos.x(), scene_pos.y()], reprojection=False)                  # plot the marker       # 
            print(f"mouse press point position: {scene_pos.x()}, {scene_pos.y()}")
            # print("Current transformation matrix:", self.view.transform())
        elif event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
            pos = event.pos()
            pos = self.view.mapToScene(pos)
            item = self.scene.itemAt(pos, self.view.transform())
            if isinstance(item, QGraphicsEllipseItem):
                self.delete_marker(item.data(self.d_joint_index))


# # utils
# # some overrided helper classes
# # work for canceling the scroll bar and make the mouse wheel event work and change the cursor
# NOTE: the unmatch of the mouse tip and the dot center could be caused by the transform
class SceneViewer(QGraphicsView):
    '''
    This class override some methods of QGraphicsView,
    to make the mouse wheel event work and change the cursor,
    also fit the image to the view, engage the lines and points
    '''
    def __init__(self, parent=None):
        super().__init__(parent)            # make sure the QGraphicsView have the init method with parent as the param, checked
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
        self._transform_start = None

    
    def wheelEvent(self, event: QWheelEvent):
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        if event.angleDelta().y() > 0:
            self.scale(1.1, 1.1)        #  
        else:
            self.scale(0.9, 0.9)

    def enterEvent(self, event: QEnterEvent) -> None:       # the event to handle the event that the mouse enter the view
        super().enterEvent(event)
        self.setCursor(Qt.ArrowCursor)


    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and event.modifiers() != Qt.ControlModifier:
            self._dragging = True
            self._drag_start_pos = event.pos()
            print("drag start position: ", self._drag_start_pos.x(), self._drag_start_pos.y())
            self._transform_start = self.transform()    # save the transform                
            print("drag transform: ", self._transform_start)
            # self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)      # super


    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            delta = event.pos() - self._drag_start_pos
            translate_x = delta.x()
            translate_y = delta.y()
            self.translate(translate_x, translate_y)
        super().mouseMoveEvent(event)


    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)


# NOTE: meet some error when using the auto triangle, check if the error comes from here
# 
class Connection(QGraphicsLineItem):        # the line is not necessarily combined with the points, you do not return, so the 
    def __init__(self, start_point, end_point, color):          # , shift=5
        super().__init__()
        # self.shift = shift      # to meet the marker center, pass the shift from somewhere

        self.start_point = start_point
        self.end_point = end_point
        # print("line points", start_point.scenePos(), end_point.scenePos())

        print(f"type of the points of the line {type(start_point)}, {type(end_point)}")

        # self._line = QLineF(start_point.mapToScene(-self.shift, -self.shift), end_point.mapToScene(-self.shift, -self.shift))
        self._line = QLineF(start_point.scenePos(), end_point.scenePos())
        self.setLine(self._line)

        # some defualt properties
        self.setSelected(False)
        
        the_color = QColor(color2QColor(color))
        self.setPen(QPen(the_color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # ...

    def updateLine(self, source):       # 
        # source position
        if source == self.start_point:
            # self._line.setP1(source.mapToScene(-self.shift, -self.shift))
            self._line.setP1(source.scenePos())
        elif source == self.end_point:
            # self._line.setP2(source.mapToScene(-self.shift, -self.shift))
            self._line.setP2(source.scenePos())
        else:
            raise ValueError("source should be the start or end point")
        self.setLine(self._line)


def color2QColor(color):
    return QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))

