# the main class of animator
from PySide6.QtGui import QWheelEvent
import cv2
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class Animator(QWidget):        # not sure if this should be a QWidget, more like a QObject.  (also QAbstractAnimation and QGraphicView)
    # 作为父类 提供所有views 和 label3d 的共有属性和方法 mainly related with frame
    '''                       
    Abstract superclass for data animation
    不一定用于播放视频 所以没有视频相关的设置 / 但是设置 QGraphicsView 和 QGraphicsScene 用于显示
    
    Animator Properties:
        frame - Frame of animation
        frameRate - Frame rate of animation
        frameInds - Indices to use in frame s.t.
                    currentFrame = self.frameInds[self.frame]
        scope - Identifier for current Animator in selective callbacks
        id - Integer 1-9, identifier for scope
        links - Cells array of linked Animators
        speedUp - Framerate increase rate
        slowDown - Framerate decrease rate
        ctrlSpeed - Framerate multiplier for ctrl
        shiftSpeed - Framerate multiplier for shift

    Animator Methods:
        get/set frame 
        get/set frameRate

        keyPressEvent - Handle key press events

        update - Abstract method for updating frames in classes that inherit from Animator 
                | python do not using a abstract method
        runAll - Static method for running the callback funtions of all Animators in links
                 It is useful to assign this function as the WindowKeyPressFcn of a figure with multiple Animators.
        linkAll - Link all Animators in a cell array together.
                
    # Some Methods not use:
        delete - Delete the Animator
        restrict - Restrict the Animator to a range(a subset) of frames

        writeVideo - Render a video of the current Animator and its links

    # Also some axes things related to matlab chart
    '''
    def __init__(self, ):
        super().__init__()
        # properties for animator itself, need to be set in subclass
        self.nFrames = 1        # total number of frames
        self.frame = 0             # the index of current frame start from 0
        self.frameInd = np.arange(self.nFrames)          # the index list of current frames, to control the range of frames

        # properties for control
        self.frameRate = 1         # how many frames jump per key press
        
        self.speedUp = 5
        self.slowDown = 5
        self.ctrlSpeed = 2
        self.shiftSpeed = 2

        # properties for mulit scope
        self.scope = 0      # Identifier for current Animator when using selective callbacks
        self.id = np.arange(9)         # Integer 0-9, identifier for scope
        self.links = []     # a list of linked Animators

        # properties for GUI of pyside
        self.setWindowTitle("Animator")
        
        self.initUI()

    # from GPT4: 
    #   可以将‘QGraphicsScene’看作是一个存放和管理图形项的‘画布’
    #   而‘QGraphicsView’是观看这个画布的窗口
    def initUI(self, ):
        # layout = QVBoxLayout()
        self.scene = QGraphicsScene()       
        self.view = QGraphicsView(self.scene)
        # layout.addWidget(self.view)
        # self.setLayout(layout)

    def update(self, ): 
        # refresh the whole class
        # abstract function
        print("update function need implement in subclass")

    ## event handle
    def keyPressEvent(self, event):
        key_to_scope = {
            Qt.Key_0: 0,
            Qt.Key_1: 1,
            Qt.Key_2: 2,
            Qt.Key_3: 3,
            Qt.Key_4: 4,
            Qt.Key_5: 5,
            Qt.Key_6: 6,
            Qt.Key_7: 7,
            Qt.Key_8: 8,
            Qt.Key_9: 9,
        }
        if event.key() == Qt.Key_Right:
            self.frameInd += self.frameRate
            print("right pressed ???")
        elif event.key() == Qt.Key_Left:
            self.frameInd -= self.frameRate
        elif event.key() == Qt.Key_Up:
            self.frameRate += self.speedUp
        elif event.key() == Qt.Key_Down:
            self.frameRate -= self.slowDown

        elif event.key() in key_to_scope:
            self.scope = key_to_scope[event.key()]

        elif event.key() == Qt.Key_S:
            print("此父非彼父！")



    # # 多窗口事件同步实，只同步按键事件 for frame change
    # # use event filter to handle the key press event, instead of linkall
    # @staticmethod
    # def linkAll(animatorList):       # animators is a list or tuple of animators
    #     '''link all the animators'''
    #     for animator in animatorList:
    #         print(type(animator))
    #         animator.links = animatorList
        
    #     # 假设所有Animator共享一个父窗口，在父窗口中设置键盘事件处理
    #     # 将所有的animator的keyPressEvent都设置为父窗口的keyPressEvent
    #     parent = animatorList[0] 
    #     print(parent.__class__)
    #     if parent:
    #         parent.keyPressEvent = lambda event: Animator.propagateEvent(animatorList, event)      # 这里的event是父窗口的event
    #     # 理论上 runAll 的参数为 animator.links 也可以
    
    # @staticmethod
    # def propagateEvent(animatorList, event):
    #     for animator in animatorList:
    #         if not hasattr(event, 'handled') or not event.handled:
    #             # mark the event as handled
    #             event.handled = True
    #             animator.keyPressEvent(event)

    # # 只同步按键事件
    # @staticmethod
    # def runAll(animatorList, event):      # synchronize the KeyPressEvent of all animators
    #     '''run all the animators'''
    #     for animator in animatorList:
    #         animator.keyPressEvent(event)
    
    
class VideoAnimator(Animator):
    '''
    interactive movie show videos and 2d keypoints
    VideoAnimator Properties:
        V - 4D (i, j, channel, N) movie to animate
        img - Handle to the imshow object

    VideoAnimator Methods:


    note: the index used in properties are all start from 0, used in functions are start from 1
    note: all the marker state should be 
    '''
    def __init__(self, video_path, skeleton):
        super().__init__()

        # properties for video
        self.video_path = video_path        # the path of the video
        self.frames = self.read_video()
        self.image = self.frames[self.frame]      # Represents the current frame
        # update the property about frames
        # self._init_property()

        # properties for 2d keypoint animator
        # self._skeletion = skeleton
        self._joint_names = skeleton["joint_names"]
        self._joints_idx = skeleton["joints_idx"]
        self._color = skeleton["color"]
        self.joints_num = len(self._joint_names)
        # on this frame
        self.current_joint_idx = None       # use index is more convenient, although a little. start from 0
        self.exist_markers = []             # also use index better, 
        self.joints2markers = {}
        # markers on all the frames
        
        self._init_property()      # self.frames_markers

        self.initUI()       # get the view and scene
        self.update()       # update the scene
        self.initView()     # view setting should be after the scene

        # constant for item data saving
        self.data_joint_name = 0
        self.data_joint_index = 1
        self.data_lines = 2

        # trivial properties
        self.marker_size = 10


    def _init_property(self, ):
        # properties from parent class
        self.nFrames = len(self.frames)       # total number of frames
        self.frame = 0            # the index of current frame start from 0s
        self.frameInd = np.arange(self.nFrames)          # the index list of current frames, to control the range of frames

        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)


    def initUI(self, ):
        super().initUI()        # get self.view and self.scene


    def initView(self, ):
        layout = QVBoxLayout()

        self.view.setMouseTracking(True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        # vscroll = self.view.verticalScrollBar()
        # hscroll = self.view.horizontalScrollBar()
        # vscroll.setSingleStep(0)
        # hscroll.setSingleStep(0)      # not work

        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        # self.view.scrollContentsBy(0, 0)      # not work

        layout.addWidget(self.view)
        self.setLayout(layout)      # set the layout for widget show, otherwise the view will not show
        
        # self.view.show()

    def read_video(self, video_path=None):
        # suitable for *.mp4 and *.avi
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


    # update the renderer for frame change
    def update(self, frame_ind=None):
        # super().update()
        if frame_ind is not None:
            self.frame = frame_ind

        current_frame = self.frames[self.frame]
        height, width, channels = current_frame.shape
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        
        # self.scene.clear()
        self.scene.addPixmap(pixmap)

        # plot the markers and lines, replot all the 
        for i in range(self.joints_num):
            if not np.isnan(self.frames_markers[self.frame, i]).all():
                pos = self.frames_markers[self.frame, i]
                self.create_drag_point_and_lines(pos, i)
                # should work here

        self.scene.update()
        # self.view.update()
        # self.view.show()      # TODO: check whether influence the showing
            # could not use view.show(), will create a new window

    def reset(self, ):      # not use for now
        # set the frame to 0, in the nFrames range
        self.restrict(0, self.frames.shape[0])


    def create_drag_point_and_lines(self, pos, joint_idx=None, ):       # draw the points and lines, also reuse for frame change
        if joint_idx is None:
            if self.current_joint_idx is not None:
                current_index = self.current_joint_idx + 1                  # "+1" to meet the matlab index
                current_joint = self._joint_names[self.current_joint_idx]       
        else:
            current_index = joint_idx + 1
            current_joint = self._joint_names[joint_idx]

        # draw the point 
        print("drawing a point")    
        
        the_point = QGraphicsEllipseItem(pos[0], pos[1], self.marker_size, self.marker_size)
        
        brush = QBrush(Qt.red)
        brush.setStyle(Qt.SolidPattern)

        the_point.setBrush(brush)
        the_point.setData(self.data_joint_name, current_joint)
        the_point.setData(self.data_joint_index, current_index)
        the_point.setData(self.data_lines, [])
        self.scene.addItem(the_point)

        self.frames_markers[self.frame, current_index-1] = pos      # checked the code is legal
        
        # draw the line, need current joints plot: self.markers
        for index, (i, j) in enumerate(self._joints_idx):
            the_color = self._color[index]      # maybe need some transform to fit the qt format
            if current_index == i:
                # and next point exist, next point is the other joint
                if self._joint_names[j] in self.exist_markers:
                    marker = self.joints2markers[self._joint_names[j]]
                    # draw the line
                    the_line = Connection(the_point, marker, the_color)
                    the_point.data(self.data_lines).append(the_line)
                    self.scene.addItem(the_line)

            elif current_index == j:
                if self._joint_names[i] in self.exist_markers:
                    marker = self.joints2markers[self._joint_names[i]]
                    the_line = Connection(marker, the_point, the_color)
                    the_point.data(self.data_lines).append(the_line)
                    self.scene.addItem(the_line)

        return the_point

    # change the current joint, from parent class
    def joint_change(self, joint_idx):
        self.current_joint_idx = joint_idx
        self.current_joint = self._joint_names[joint_idx]

        return joint_idx     # return to sync the current joint index


    def reset_marker(self, item, pos):
        item.setPos(pos)

        for line in item.data(self.data_lines):
            line.updateLine(item)
        
        self.scene.update()     # do we need this? 


    def delete_the_point(self, item: QGraphicsItem):
        joint_name = item.data(self.data_joint_name)
        joint_index = item.data(self.data_joint_index)
        self.scene.removeItem(item)
        self.frames_markers[self.frame, joint_index-1] = np.nan
        self.exist_markers.remove(joint_index-1)

        lines = item.data(self.data_lines)
        for line in lines:
            self.scene.removeItem(line)

        self.scene.update()


    # GUI interaction
    def keyPressEvent(self, event):     # override the keyPressEvent, mainly for frame control
        super().keyPressEvent(event)
        
        # inherit the parent class's keyPressEvent
        # add some new key press event
        if event.key() == Qt.Key_S:
            # print current frame and rate
            print("show some information")
            event.ignore()
        elif event.key() == Qt.Key_R:
            self.reset()
        elif event.key() == Qt.Key_H:
            # print help information
            print("show help information")


        # update the renderer
        # self.update()
        
        # super().keyPressEvent(event)

        # pass the event to the parent widget



    # 滚轮事件 实现 图片放大缩小
    def wheelEvent(self, event):     # override the wheelEvent
        if event.angleDelta().y() > 0:
            self.view.scale(1.1, 1.1)
        else:
            self.view.scale(0.9, 0.9)

        # the problem is that the scroll will still be triggered, so we need to stop the event

    def mousePressEvent(self, event):       # override the mousePressEvent
        # plot the current joint        
        if event.buttons() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
            # 如果当前节点已经存在，则重置位置。不存在则新建一个
            if self.current_joint_idx in self.exist_markers:
                # reset the marker position
                self.reset_marker(self.joints2markers[self._joint_names(self.current_joint_idx)], event.scenepos())       # TODO: check the scenePos

            else: 
                if self.current_joint_idx is not None:
                    marker = self.create_drag_point_and_lines((self.view.mapToScene(event.pos()).x(), self.view.mapToScene(event.pos()).y()))
                    self.exist_markers.append(self.current_joint_idx)
                    self.joints2markers[self._joint_names(self.current_joint_idx)] = marker


        elif event.buttons() == Qt.LeftButton:      # 点选 joint marker
            print(self.view.mapToScene(event.pos()))
            item = self.scene.itemAt(self.view.mapToScene(event.pos()), self.view.viewportTransform())       # to scene position
            if isinstance(item, QGraphicsEllipseItem):
                self.joint_change(item.data(self.data_joint_index)-1)       # "-1" to meet the python index





# 这个可以有
class Keypoint3DAnimator(Animator):
    def __init__(self, ):
        super().__init__()
        self.lim = 1
        self.frame = 1
        self.frameRate = 1


# helper class
class Connection(QGraphicsLineItem):
    def __init__(self, start_point, end_point, color):
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point

        # some defualt properties
        self.setSelected(False)
        self.setPen(QPen(color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # ...

        self._line = QLineF(start_point.scenePos(), end_point.scenePos())
        self.setLine(self._line)

    def updateLine(self, source):       # 
        if source == self.start_point:
            self._line.setP1(source.scenepos())
        elif source == self.end_point:
            self._line.setP2(source.scenepos())
        else:
            raise ValueError("source should be the start or end point")
        self.setLine(self._line)

        
