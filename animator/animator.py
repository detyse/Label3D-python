# animator for video display
# NOTE: the data stored in the item is immutable, so the data should reset after changing the stored data

# NOTE: set the z-value for each item, the z-value of pix set to 0, lines for 1, points for 2

from PySide6.QtGui import QEnterEvent, QMouseEvent, QWheelEvent
import cv2
import numpy as np
import os

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class Animator(QWidget):    # inherit from QWidget
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

        # 没有 大跨度翻页的需求 暂时
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
        self._joints_num = len(self._joint_names)        # total number of joints
        # NOTE: THESE PROPERTIES SHOULD NOT BE CHANGED IN THE CLASS, JUST FOR READ
        
        # on this frame, f for only in this frame
        self.f_current_joint_idx = None         # the current joint index
        # this property only changed by the set_joint method
        self.f_exist_markers = []           # a list of joints idx indicating the joints on this frame       
        self.f_joints2markers = {}            # a dict map the joint name to the marker on this frame
        # will these change after reprojection? if not we could update the frame after the reprojection

        # 
        self.frames_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)
        self.original_markers = np.full((self.nFrames, len(self._joint_names), 2), np.nan)
        # here markers are all position, not the items

        # d for data, constant for item data saving
        # the index for save the data in the item
        # NOTE: constant variable 
        self.d_joint_index = 0
        self.d_lines = 1

        # trivial properties
        self.marker_size = 10       # the size of the point

        self.preview_mode = False       # the preview mode for the animator, if true, the animator could not be edited  

        self.initUI()
        # self.update_frame()         # not call in this class, for a better control  # called by the load_labels in label3d
        self._initView()


    # also just consider the npy file, do not consider the index frame
    # build the index
    # load the frame files from output folder
    def load_videos(self, video_folder, label_num):
        # file_list = [f for f in os.listdir(video_folder) if os.path.isfile(os.path.join(video_folder, f))]
        file_list = os.listdir(video_folder)

        # if there is npy file, load the npy file
        for file in file_list:
            if file == "frames.npy":
                frames = np.load(os.path.join(video_folder, file))
                return frames
        
        # i think we are not using this right now, all the video will be saved as npy file
        # because we want to solve the GUI block problem
        # if there is no npy file, load the video file
        # the no use
        for file in file_list:
            if file == "0.mp4" or file == "0.avi":
                frame_num_list = []
                frame_index_list = []
                frames = []

                video_file = os.path.join(video_folder, file)

                # 
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

                frames = np.array(frames)
                return frames
            
        # error situation
        raise ValueError("No video file could load!")


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
        self.update_frame()         # update the frame after the labels are loaded


    def clear_marker_2d(self, ):        # clear the current joint marker
        if self.f_current_joint_idx is None:
            return
        
        # full the marker with nan
        # self.frames_markers[self.frame, self.f_current_joint_idx, ...] = np.nan
        self.frames_markers[self.frame, self.f_current_joint_idx, ...] = np.nan
        self.original_markers[self.frame, self.f_current_joint_idx, ...] = np.nan
        self.delete_marker()


    ## only called by label3d, to sync rt
    def set_joint(self, joint_idx):
        # print(f"animator - set_joint function called, joint index: {joint_idx}")
        self.f_current_joint_idx = joint_idx

        # highlight the joint marker
        for joint in self.f_joints2markers.values():
            self.highlight_marker(joint)
        
        # normal the other markers
        for idx in self.f_exist_markers:
            if idx != joint_idx:
                self.normal_marker(self.f_joints2markers[idx])
        return True


    def set_marker_2d(self, pos, reprojection=False):       # the params are unnecessary
        self.plot_marker_and_lines(pos, reprojection=reprojection)
        return True


    # NOTE: 
    def get_marker_2d(self, frame=None, joint_idx=None):       
        if frame is None and joint_idx is None:
            return self.frames_markers[self.frame, self.f_current_joint_idx]
        elif frame is None or joint_idx is None:
            raise ValueError("frame and joint index should be both set or not set")
        else:
            return self.frames_markers[frame, joint_idx]
    

    def get_all_original_marker_2d(self, ):         # return the original labeled markers of current frame
        return self.original_markers[self.frame]


    # FIXME: load the joint3d and hand labeled data
    def update_frame(self, frame_ind=None):       # this function should be use after the frame change, also used to init the scene        
        # frame_ind: the index of video frames, 
        # update frame and using self.frame as current frame
        if frame_ind is not None:
            if self.frame == frame_ind:
                return  # do nothing to reduce the computation cost
            
            self.frame = frame_ind
        
        # clear the scene, but the view would not change
        self.scene.clear()

        # make sure all the f start properties are reset
        # update the frame params in the f properties
        self.f_current_joint_idx = None         # NOTE: could not reset to None, keep the last frame joint
        self.f_exist_markers = []
        self.f_joints2markers = {}
        
        current_frame = self.video_frames[self.frame]
        height, width, channels = current_frame.shape       # the frame shape would change
        # print(f"frame shape: {height}, {width}, {channels}")
        bytesPerLine = channels * width
        q_image = QImage(current_frame.data, width, height, bytesPerLine, QImage.Format_RGB888)         # should we set up a class property for frame image? for change the contrast?  and init?
        self.pixmap = QPixmap.fromImage(q_image)            # only define when the frame is defined, use to change the frame contrast 
        # define a contrast factor here for contrast adjustment
        self.contrast_factor = 1.0

        self.pixmap_item = self.scene.addPixmap(self.pixmap)

        # get the scene rect 
        the_rect = self.scene.sceneRect()
        self.view.fitInView(the_rect, Qt.KeepAspectRatio)       # fit the image to the view

        # plot the labeled joint markers and lines on this frame
        for i in range(self._joints_num):
            if not np.isnan(self.frames_markers[self.frame, i]).all():
                pos = self.frames_markers[self.frame, i]
                self.plot_marker_and_lines(pos, i, reprojection=False)          # update the exist markers

        self.scene.update()


    # NOTE: add at the 240628
    # here are two thoughts, just change the image QPixmap, 
    # this function includes two process, one is change the contrast, the other is change the frame
    def change_frame_contract(self, ):
        # get the current frame pix object
        print("change frame contrast function called")
        if self.pixmap is None:
            print("no frame image loaded")
            return
        
        else:
            print(f"contrast factor: {self.contrast_factor}")
            image = self.pixmap.toImage()
            image = image.convertToFormat(QImage.Format_RGB888)

            width = image.width()
            height = image.height()
            ptr = image.constBits()
            arr = np.array(ptr).reshape(height, width, 3)

            mean = np.mean(arr)
            adjusted_array = (arr - mean) * self.contrast_factor + mean
            adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)

            # update the image
            q_image = QImage(adjusted_array.data, adjusted_array.shape[1], adjusted_array.shape[0], adjusted_array.shape[1]*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # update the pixmap
            if hasattr(self, 'pixmap_item') and self.pixmap_item is not None:
                self.pixmap_item.setPixmap(pixmap)
            else:
                self.pixmap_item = self.scene.addPixmap(pixmap)

            self.scene.update()         # NOTE: try to directly update the pixmap, do not clear the scene, hope all is right
            # otherwise could only call the update_frame method which have bug
        return 


    # NOTE: this function should be called by the label3d for sync contrast update
    def contrast_change(self, contrast_factor):
        if contrast_factor is None:
            return
        
        self.contrast_factor = contrast_factor
        
        if self.pixmap is None:
            print("no frame image loaded")
            return 
        
        else:
            print(f"contrast factor: {self.contrast_factor}")
            image = self.pixmap.toImage()
            image = image.convertToFormat(QImage.Format_RGB888)

            width = image.width()
            height = image.height()
            ptr = image.constBits()
            arr = np.array(ptr).reshape(height, width, 3)

            print("Original array shape:", arr.shape)
            print("Original array mean:", np.mean(arr))

            mean = np.mean(arr)
            adjusted_array = (arr - mean) * self.contrast_factor + mean
            adjusted_array = np.clip(adjusted_array, 0, 255).astype(np.uint8)

            print("Adjusted array mean:", np.mean(adjusted_array))

            # update the image
            q_image = QImage(adjusted_array.data, adjusted_array.shape[1], adjusted_array.shape[0], adjusted_array.shape[1]*3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # update the pixmap
            if hasattr(self, 'pixmap_item') and self.pixmap_item is not None:
                self.pixmap_item.setPixmap(pixmap)
            else:
                self.pixmap_item = self.scene.addPixmap(pixmap)

            self.scene.update()
        return


    # rewrite the functions
    def plot_marker_and_lines(self, pos, joint_idx=None, reprojection=False):
        # print("animator - plot_marker_and_lines called")
        if joint_idx is None:
            if self.f_current_joint_idx is None:
                return
            else:
                joint_idx = self.f_current_joint_idx

        marker = self.f_joints2markers.get(joint_idx)
        if marker:
            self.reset_marker(self.f_joints2markers[joint_idx], pos, reprojection)
            # print("animator - plot_marker_and_lines called -- reset marker")
            self.scene.update()
            return
        
        else:
            # print("animator - plot_marker_and_lines called -- create marker")
            marker = QGraphicsEllipseItem(-self.marker_size//2, -self.marker_size//2, self.marker_size, self.marker_size)
            marker.setPos(pos[0], pos[1])
            marker.setZValue(2)         # the z-value of the marker is 2
            brush = QBrush(color2QColor(self._color[joint_idx+1]))
            brush.setStyle(Qt.SolidPattern)
            marker.setBrush(brush)
            marker.setData(self.d_joint_index, joint_idx)
            marker.setData(self.d_lines, [])
            self.scene.addItem(marker)
            self.f_joints2markers[joint_idx] = marker
            self.f_exist_markers.append(joint_idx)

            # potential bugs?
            if joint_idx == self.f_current_joint_idx:
                self.highlight_marker(marker)

            self.frames_markers[self.frame, joint_idx] = pos
            if not reprojection:
                self.original_markers[self.frame, joint_idx] = pos

            for index, (i, j) in enumerate(self._joints_idx):
                if joint_idx == i - 1 or joint_idx == j - 1:
                    other_id = j - 1 if joint_idx == i - 1 else i - 1
                    other_marker = self.f_joints2markers.get(other_id)
                    if other_marker:
                        self.update_or_create_connection(marker, other_marker, self._color[index])

        self.scene.update()


    # called in label3d or animator
    def highlight_marker(self, marker_item):
        effect = QGraphicsDropShadowEffect()
        effect.setColor(QColor())
        effect.setBlurRadius(5)
        marker_item.setGraphicsEffect(effect)


    def normal_marker(self, marker_item):
        marker_item.setGraphicsEffect(None)


    def update_or_create_connection(self, marker1, marker2, color):
        connection = next((c for c in marker1.data(self.d_lines) if marker2 == c.theOtherPoint(marker1)), None)
        # the connection should always be None
        
        if connection:
            connection.updateLine()
        else:
            connection = Connection(marker1, marker2, color)
            self.scene.addItem(connection)
            
            # here should be called t3 times
            # print("append lines")
            marker1_connections = marker1.data(self.d_lines)
            marker1_connections.append(connection)
            marker1.setData(self.d_lines, marker1_connections)
            marker2_connections = marker2.data(self.d_lines)
            marker2_connections.append(connection)
            marker2.setData(self.d_lines, marker2_connections)


    # 
    def reset_marker(self, item, new_pos, reprojection=False):
        print("reset marker called")
        item.setPos(new_pos[0], new_pos[1])

        joint_idx = item.data(self.d_joint_index)
        self.frames_markers[self.frame, joint_idx] = new_pos
        if not reprojection:
            self.original_markers[self.frame, joint_idx] = new_pos

        for connection in item.data(self.d_lines):
            connection.updateLine()
    

    # 
    def delete_marker(self, joint_idx=None):
        print("delete marker called")
        joint_idx = joint_idx or self.f_current_joint_idx
        if joint_idx is None:
            return
        
        marker = self.f_joints2markers.pop(joint_idx, None)
        # pop the marker from the f_
        if marker:
            for connection in list(marker.data(self.d_lines)):
                other_marker = connection.theOtherPoint(marker)
                other_connections = other_marker.data(self.d_lines)
                other_connections.remove(connection)
                other_marker.setData(self.d_lines, other_connections)
                # print(other_marker.data(self.d_lines))
                self.scene.removeItem(connection)

            self.scene.removeItem(marker)
            self.frames_markers[self.frame, joint_idx] = np.nan
            if joint_idx in self.f_exist_markers:
                self.f_exist_markers.remove(joint_idx)
            
            self.scene.update()
    

    # ## NOTE: contrast change for single frame 
    # def keyPressEvent(self, event):
    #     # ignore all the key press event, leave it to the parent widget
    #     # except sevel key press event
    #     # only for contrast adjust other events will be ignored
    #     if event.key() == Qt.Key_BracketLeft:
    #         print("key press event (animator): bracket left")
    #         self.contrast_factor -= 0.1
    #         if self.contrast_factor < 0.1:
    #             self.contrast_factor = 0.1
    #         self.change_frame_contract()        # hope not need to use the update_frame method
    #     elif event.key() == Qt.Key_BracketRight:
    #         print("key press event (animator): bracket right")
    #         self.contrast_factor += 0.1
    #         if self.contrast_factor > 5.0:
    #             self.contrast_factor = 5.0
    #         self.change_frame_contract()
    #     else:
    #         event.ignore()


    def keyPressEvent(self, event):
        event.ignore()

    
    def mousePressEvent(self, event):
        if not self.preview_mode:
            if event.button() == Qt.LeftButton and event.modifiers() == Qt.ControlModifier:
                # pos = event.pos()           # the event pos relative to the widget, there is a position shift between the widget and the view
                # get the true position in the view
                view_pos = self.view.mapFromGlobal(self.mapToGlobal(event.pos()))
                scene_pos = self.view.mapToScene(view_pos)
                self.plot_marker_and_lines([scene_pos.x(), scene_pos.y()], reprojection=False)                  # plot the marker       # 

            # FIXME: the delete function on the first joint could not work properly
            elif event.button() == Qt.RightButton and event.modifiers() == Qt.ControlModifier:
                view_pos = self.view.mapFromGlobal(self.mapToGlobal(event.pos()))
                scene_pos = self.view.mapToScene(view_pos)
                item = self.scene.itemAt(scene_pos, self.view.transform())
                if isinstance(item, QGraphicsEllipseItem):
                    joint_idx = item.data(self.d_joint_index)
                    if joint_idx is not None:
                        self.delete_marker(joint_idx)
                    else:
                        print("joint index is None")

        else:
            event.ignore()


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
            # print("drag start position: ", self._drag_start_pos.x(), self._drag_start_pos.y())
            self._transform_start = self.transform()    # save the transform                
            # print("drag transform: ", self._transform_start)
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


    def keyPressEvent(self, event):
        event.ignore()


# NOTE: meet some error when using the auto triangle, check if the error comes from here
class Connection(QGraphicsLineItem):        # the line is not necessarily combined with the points, you do not return, so the 
    def __init__(self, start_point, end_point, color):          # , shift=5
        super().__init__()
        
        self.start_point = start_point
        self.end_point = end_point
        self.updateLine()

        the_color = QColor(color2QColor(color))
        self.setPen(QPen(the_color, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))


    def updateLine(self,):
        # self._line = QLineF(self.start_point.scenePos(), self.end_point.scenePos())
        start_pos = self.start_point.scenePos()
        end_pos = self.end_point.scenePos()
        self.setLine(QLineF(start_pos, end_pos))
        self.setZValue(1)         # the z-value of the line is 1
        # print(f"update line: {start_pos.x()}, {start_pos.y()} to {end_pos.x()}, {end_pos.y()}")


    # get the orther point of the line
    def theOtherPoint(self, item):
        if item == self.start_point:
            return self.end_point
        elif item == self.end_point:
            return self.start_point
        else:
            raise ValueError("Provided item is not an endpoint of the line.")


def color2QColor(color):
    return QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255), int(color[3]*255))

