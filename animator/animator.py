# the main class of animator
from PySide6.QtGui import QWheelEvent
import cv2
import numpy as np
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

class Animator(QWidget):        # not sure if this should be a QWidget, more like a QObject.  (also QAbstractAnimation and QGraphicView)
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

    ## set/get methods
    def setFrame(self, frame):
        self.frame = frame

    def getFrame(self, ):
        return self.frame

    # restrict the range of frames, as a method for new frames fitting
    def restrict(self, newFrames):
        self.frameInd = np.arange(newFrames)
        self.nFrames = newFrames
        self.frame = 1

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
        elif event.key() == Qt.Key_Left:
            self.frameInd -= self.frameRate
        elif event.key() == Qt.Key_Up:
            self.frameRate += self.speedUp
        elif event.key() == Qt.Key_Down:
            self.frameRate -= self.slowDown

        elif event.key() in key_to_scope:
            self.scope = key_to_scope[event.key()]
        

    def checkVisible(self, ):
        # check the current frame is visible in the isVisible vector
        # 略
        pass

    @staticmethod
    def linkAll(animatorList):       # animators is a list or tuple of animators
        '''link all the animators'''
        for animator in animatorList:
            animator.links = animatorList
        
        # 假设所有Animator共享一个父窗口，在父窗口中设置键盘事件处理
        # 将所有的animator的keyPressEvent都设置为父窗口的keyPressEvent
        parent = animatorList[0].parent()
        if parent:
            parent.keyPressEvent = lambda event: [animator.runAll(animatorList, event) for animator in animatorList]
        # 理论上 runAll 的参数为 animator.links 也可以

    # 只同步按键事件
    @staticmethod
    def runAll(animatorList, event):      # synchronize the KeyPressEvent of all animators
        '''run all the animators'''
        for animator in animatorList:
            animator.keyPressEvent(event)
    
    
class VideoAnimator(Animator):
    '''
    interactive movie
    VideoAnimator Properties:
        V - 4D (i, j, channel, N) movie to animate
        img - Handle to the imshow object

    VideoAnimator Methods:

    '''
    def __init__(self, video_path, skeleton, view, scene):
        super().__init__()

        # properties for video
        self.video_path = video_path        # the path of the video
        self.frames = self.read_video()
        self.image = self.frames[self.frame]      # Represents the current frame
        # update the property about frames
        # self._init_property()

        # properties for 2d keypoint animator
        self._skeletion = skeleton
        self._joint_names = self._skeletion.joint_names
        self._joints_idx = self._skeletion.joints_idx
        self._color = self._skeletion.color
        self.joints_num = len(self._joint_names)
        # on this frame
        self.current_joint_idx = None       # use index is more convenient, although a little. start from 0
        self.exist_markers = []             # also use index better
        self.joints2markers = {}
        # markers on all the frames
        
        self._init_property()      # self.frames_markers

        self.initUI()
        self.update()


    def _init_property(self, ):
        # properties from parent class
        self.nFrames = len(self.frames)       # total number of frames
        self.frame = 0            # the index of current frame start from 0s
        self.frameInd = np.arange(self.nFrames)          # the index list of current frames, to control the range of frames

        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)


    def initUI(self, ):
        super().initUI()        # get self.view and self.scene

        layout = QVBoxLayout()

        self.view.setMouseTracking(True)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag)

        vscroll = self.view.verticalScrollBar()
        hscroll = self.view.horizontalScrollBar()
        vscroll.setSingleStep(0)
        hscroll.setSingleStep(0)

        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.view.scrollContentsBy(0, 0)

        self.scaleFactor = 1.0

        layout.addWidget(self.view)
        self.setLayout(layout)      # set the layout for widget show, otherwise the view will not show


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
    def update_frame(self, frame_ind):
        # super().update()
        self.frame = frame_ind

        current_frame = self.frames[self.frame]
        height, width, channels = current_frame.shape
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        
        self.scene.clear()
        self.scene.addPixmap(pixmap)

        # plot the markers and lines, replot all the 
        for i in range(self.joints_num):
            if not np.isnan(self.frames_markers[self.frame, i]).all():
                pos = self.frames_markers[self.frame, i]
                self.create_drag_points_and_lines(pos, i)
                # should work here

        self.scene.update()
        # self.view.show()


    def reset(self, ):      # not use for now
        # set the frame to 0, in the nFrames range
        self.restrict(0, self.frames.shape[0])


    def create_drag_points_and_lines(self, pos, joint_idx=None, ):       # draw the points and lines, also reuse for frame change
        if joint_idx is None:
            current_index = self.current_joint_idx + 1                  # "+1" to meet the matlab index
            current_joint = self.joint_names[self.current_joint_idx]       
        else:
            current_index = joint_idx + 1
            current_joint = self.joint_names[joint_idx]

        # draw the point 
        the_point = QGraphicsEllipseItem(pos[0], pos[1], self.marker_size, self.marker_size)
        the_point.setData("joint name", current_joint)
        the_point.setData("joint index", current_index)
        the_point.setData("lines", [])
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
                    the_point.data("lines").append(the_line)
                    self.scene.addItem(the_line)

            elif current_index == j:
                if self._joint_names[i] in self.exist_markers:
                    marker = self.joints2markers[self._joint_names[i]]
                    the_line = Connection(marker, the_point, the_color)
                    the_point.data("lines").append(the_line)
                    self.scene.addItem(the_line)
            else:
                return


    def delete_the_point(self, item: QGraphicsItem):
        joint_name = item.data("joint name")
        joint_index = item.data("joint index")
        self.scene.removeItem(item)
        self.markers.remove()       # remove the marker from the list of points, used for replot
        self.frames_markers[self.frame, joint_index-1] = np.nan
        self.exist_markers.remove(joint_name)

        lines = item.data("lines")
        for line in lines:
            self.scene.removeItem(line)

        self.scene.update()

    # GUI interaction
    def keyPressEvent(self, event):     # override the keyPressEvent, mainly for frame control
        # inherit the parent class's keyPressEvent
        super().keyPressEvent(event)

        # add some new key press event
        if event.key() == Qt.Key_S:
            # print current frame and rate
            print("show some information")
        elif event.key() == Qt.Key_R:
            self.reset()
        elif event.key() == Qt.Key_H:
            # print help information
            print("show help information")

        # update the renderer
        self.update()


    # 滚轮事件 实现 图片放大缩小
    def wheelEvent(self, event):     # override the wheelEvent
        # if len(self.scene().items()) == 0:
        #     return
        
        # curPoint = event.position()
        # scenePoint = self.view.mapToScene(curPoint)

        # viewWidth = self.view.viewport().width()
        # viewHeight = self.view.viewport().height()

        # hScale = 
        # do not inherit the parent class's wheelEvent
        if event.angleDelta().y() > 0:
            self.view.scale(1.1, 1.1)
        else:
            self.view.scale(0.9, 0.9)

        # remove the scroll
        # pass 

    # def myWheelEvent(self, event: QWheelEvent):
    #     if event.angleDelta().y() > 0:
    #         self.view.scale(1.1, 1.1)
    #     else:
    #         self.view.scale(0.9, 0.9)

    # # 鼠标拖拽事件 实现 图片移动
    # def mouseMoveEvent(self, event: QMouseEvent):       # override the mouseMoveEvent
    #     # with ctrl key, move the image
    #     if event.buttons() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
    #         self.view.translate(event.pos() - self.lastPos)
    #         self.lastPos = event.pos()
    #         # self.update()
    
    # # 鼠标点击事件 返回 图片的坐标
    # def mousePressEvent(self, event: QMouseEvent):       # override the mousePressEvent
    #     if event.buttons() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
    #         self.lastPos = event.pos()
    #     elif event.buttons() == Qt.RightButton:
    #         print(self.view.mapToScene(event.pos()))

    def KeepAspectRatio(self, ):
        # keep the aspect ratio of the image, fit the frame size
        pass




class Keypoint3DAnimator(Animator):
    def __init__(self, ):
        super().__init__()
        self.lim = 1
        self.frame = 1
        self.frameRate = 1


class DraggableKeypoint2DAnimator(Animator):
    '''
    the points and lines on video
    '''
    def __init__(self, skeleton, view, scene):
        '''
        view and scene from corresponding QGraphicsView and QGraphicsScene of 
        VideoAnimator, share the same window
        add QGraphicsItem for points and lines to the scene
        '''
        super().__init__()
        self.lim = 1
        self.frame = 1
        self.frameRate = 1
        
        # defualt properties to point
        self.marker_size = 20
        self.line_width = 5
        self.drug_point_color = [255, 0, 0]
        
        # properties for GUI
        self.view = view
        self.scene = scene


        # non changeable properties 
        self._skeleton = skeleton 
        self._joint_names = self._skeleton.joint_names
        self._joints_idx = self._skeleton.joints_idx
        self._color = self._skeleton.color

        # properties for joints
        self.current_joint = None       # it will change by some signal, out of the animator
        self.exist_markers = []     # list of joint names, the exist markers
        # self.joint_names2indexs = {}     # joint name to index # just use the list index + 1
        self.joints2markers = {}        # dict of joint name to markers


    def create_drag_points_and_lines(self, pos):       # draw the points and lines
        # create points, 
        # the color and links depends on the current joint, 
        # if there is a point at the joint, draw a point
        # and lines between the points
        # the position could get from the event of mouse press
        skeleton = self._skeleton 
        
        current_joint = self.current_joint_index
        current_index = self.joint_names.index(current_joint) + 1

        # draw the point 
        the_point = QGraphicsEllipseItem(pos[0], pos[1], self.marker_size, self.marker_size)
        the_point.setData("joint name", current_joint)
        the_point.setData("joint index", current_index)
        the_point.setData("lines", []) 
        self.scene.addItem(the_point)

        # draw the line, need current joints plot: self.markers 
        for index, (i, j) in enumerate(self._joints_idx):
            the_color = self._color[index]      # maybe need some transform to fit the qt format
            if current_index == i:
                # and next point exist, next point is the other joint
                if self._joint_names[j] in self.exist_markers:
                    marker = self.joints2markers[self._joint_names[j]]
                    # draw the line
                    the_line = Connection(the_point, marker, the_color)
                    the_point.data("lines").append(the_line)
                    self.scene.addItem(the_line)

            elif current_index == j:
                if self._joint_names[i] in self.exist_markers:
                    marker = self.joints2markers[self._joint_names[i]]
                    the_line = Connection(marker, the_point, the_color)
                    the_point.data("lines").append(the_line)
                    self.scene.addItem(the_line)
            else:
                return

    def delete_the_point(self, item: QGraphicsItem):
        # the QGraphicsItem could be deleted by the item and also delete the lines 
        # get the item joint
        
        joint_name = item.data("joint name")
        joint_index = item.data("joint index")
        self.scene.removeItem(item)
        self.markers.remove(joint_name)

        # the lines could be deleted by the item
        lines = item.data("lines")
        for line in lines:
            self.scene.removeItem(line)

        # also change the self.markers state(exist or not)

        

    # drugging the points, the events happen on the QGraphicsScene level
    def mousePressEvent(self, event: QMouseEvent):
        # get the current joint, like no need, because the current joint could get in the draw function
        if event.buttons() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:

            self.current_marker(event.pos())
            # if the point is in the current joint, then drag it
            # if not, do nothing
            pass

        # get the item at the mouse cursor position
        item = self.scene.itemAt(event.pos())       # to scene position
        if item and event.modifiers() == Qt.ControlModifier:
            self.delete_the_point(item)
        
        # and draggable 
        pass

    def mouseMoveEvent(self, event):
        item = self.scene.itemAt(event.scenePos())      # check 
        for line in item.data("lines"):
            line.updateLine(item)

        super().mouseMoveEvent(event)

    # do i need this function? test it
    def mouseReleaseEvent(self, event):
        # reset the marker position

        pass 

    # used for dragging the points, do not need to be draggable
    def reset_marker(self, item, pos):
        # reset the marker position
        item.setPos(pos)        
        

class HeatMapAnimator(Animator):
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
            self._line.setP1(source.scenePos())
        elif source == self.end_point:
            self._line.setP2(source.scenePos())
        else:
            raise ValueError("source should be the start or end point")
        self.setLine(self._line)

        
