# FIXME: there will be mismatch between the frame and the joint, check the sorting
# TODO: ignore the prediction path, skeleton path and params path which is 

import os
import sys
import cv2
import numpy as np
import yaml

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from utils.utils import read_json_skeleton

from label3d import Label3D
from utils.videomaker import VideoMaker

from scipy.io import loadmat
import pickle as pkl
import traceback

import json
import pandas as pd


class VideoGenerationWorker(QThread):
    """视频生成工作线程"""
    progress_update = Signal(str, str)  # message, color
    finished = Signal(bool, str)  # success, message
    
    def __init__(self, video_maker):
        super().__init__()
        self.video_maker = video_maker
    
    def run(self):
        try:
            self.progress_update.emit("Generating video... This may take a while.", "orange")
            self.video_maker.make_video()
            self.finished.emit(True, "Video generation completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error generating video: {str(e)}")


class DataLoadingWorker(QThread):
    """数据加载工作线程"""
    progress_update = Signal(str, str)  # message, color
    finished = Signal(bool, object, object, str)  # success, params, kwargs, message
    
    def __init__(self, loader_instance):
        super().__init__()
        self.loader = loader_instance
    
    def run(self):
        try:
            # 重用现有的load_data逻辑，但在线程中运行
            params, kwargs = self.loader._load_data_internal()
            if params is not None:
                self.finished.emit(True, params, kwargs, "Data loaded successfully!")
            else:
                self.finished.emit(False, None, None, "Failed to load data")
        except Exception as e:
            self.finished.emit(False, None, None, f"Error loading data: {str(e)}")


class ViewerLoader(QWidget):
    def __init__(self):
        super().__init__()
        
        # 预初始化UI组件引用
        self.progress_bar = None
        self.status_label = None
        
        # 工作线程相关
        self.video_worker = None
        self.data_worker = None
        
        # 初始化UI
        self.initUI()
        self.setWindowTitle("3D Pose Viewer")
        self.setGeometry(100, 100, 600, 300)

    # TODO add the video generation function
    def initUI(self):
        outer_layout = QVBoxLayout()

        layout = QGridLayout()

        self.video_path_label = QLabel("Video Path: ")
        self.video_path = QLineEdit()
        self.video_path_browse = QPushButton("...")
        self.video_path_browse.clicked.connect(lambda: self.dir_dialog(self.video_path))
        layout.addWidget(self.video_path_label, 0, 0)
        layout.addWidget(self.video_path, 0, 1)
        layout.addWidget(self.video_path_browse, 0, 2)

        self.prediction_path_label = QLabel("Predicted Path: ")
        self.prediction_path = QLineEdit()
        self.prediction_path_browse = QPushButton("...")
        self.prediction_path_browse.clicked.connect(lambda: self.file_dialog(self.prediction_path))
        layout.addWidget(self.prediction_path_label, 1, 0)
        layout.addWidget(self.prediction_path, 1, 1)
        layout.addWidget(self.prediction_path_browse, 1, 2)

        # do we need skeleton file and the param file?
        self.skeleton_path_label = QLabel("Skeleton Path: ")
        self.skeleton_path = QLineEdit()
        self.skeleton_path_browse = QPushButton("...")
        self.skeleton_path_browse.clicked.connect(lambda: self.file_dialog(self.skeleton_path))
        layout.addWidget(self.skeleton_path_label, 2, 0)
        layout.addWidget(self.skeleton_path, 2, 1)
        layout.addWidget(self.skeleton_path_browse, 2, 2)

        self.param_path_label = QLabel("Param Path: ")
        self.param_path = QLineEdit()
        self.param_path_browse = QPushButton("...")
        self.param_path_browse.clicked.connect(lambda: self.file_dialog(self.param_path))
        layout.addWidget(self.param_path_label, 3, 0)
        layout.addWidget(self.param_path, 3, 1)
        layout.addWidget(self.param_path_browse, 3, 2)
        
        # the frame to check load a one dimansion numpy array
        # have the highest priority, if the frame2check is set the frame_num2check will be ignored
        self.frame2check_label = QLabel("Indexes List: (.xlsx)")
        self.frame2check = QLineEdit()
        self.frame2check_browse = QPushButton("...")
        self.frame2check_browse.clicked.connect(lambda: self.index_dialog(self.frame2check))
        layout.addWidget(self.frame2check_label, 4, 0)
        layout.addWidget(self.frame2check, 4, 1)
        layout.addWidget(self.frame2check_browse, 4, 2)

        self.frame_num2check_label = QLabel("Frame Range: ")
        self.frame_num2check = QLineEdit()
        # set the input as integer
        self.frame_num2check.setValidator(QIntValidator())
        layout.addWidget(self.frame_num2check_label, 5, 0)
        layout.addWidget(self.frame_num2check, 5, 1)

        # choose ways to show the results, 下拉菜单选择
        self.show_results_label = QLabel("Show Results: ")
        self.show_results = QComboBox()
        self.show_results.addItems(["p ascending", "p descending", "frame order"])
        # the default is frame order
        self.show_results.setCurrentText("frame order")
        layout.addWidget(self.show_results_label, 6, 0)
        layout.addWidget(self.show_results, 6, 1)

        # 添加状态显示区域
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_bar)
        status_group.setLayout(status_layout)

        # add the group box for loading and video generation

        # change to V layout 
        video_group_layout = QHBoxLayout()
        video_generation_group = QGroupBox("Generate Videos")

        frame_rate_label = QLabel("Frame Rate: ")
        self.frame_rate = QLineEdit()
        self.frame_rate.setFixedSize(100, 20)
        
        video_format_label = QLabel("Video Format: ")
        self.video_format = QComboBox()
        self.video_format.addItems(["mp4", "avi"])
        self.video_format.setCurrentText("mp4")
        
        # 添加内存管理设置
        cache_frames_label = QLabel("Cache Frames: ")
        self.cache_frames = QComboBox()
        self.cache_frames.addItems(["500 (低内存)", "1000 (默认)", "2000 (中等内存)", "5000 (高内存)", "10000 (超高内存)"])
        self.cache_frames.setCurrentText("1000 (默认)")
        self.cache_frames.setToolTip("控制一次缓存的帧数。根据您的内存大小选择：\n• 500: 适合8GB以下内存\n• 1000: 适合8-16GB内存\n• 2000: 适合16GB以上内存\n• 5000: 适合32GB以上内存")
        
        self.generate_video_button = QPushButton("Generate")
        self.generate_video_button.clicked.connect(self.generate_video)

        video_group_layout.addWidget(frame_rate_label)
        video_group_layout.addWidget(self.frame_rate)
        video_group_layout.addStretch()
        video_group_layout.addWidget(video_format_label)
        video_group_layout.addWidget(self.video_format)
        video_group_layout.addStretch()
        video_group_layout.addWidget(cache_frames_label)
        video_group_layout.addWidget(self.cache_frames)
        video_group_layout.addStretch()
        video_group_layout.addWidget(self.generate_video_button)

        video_generation_group.setLayout(video_group_layout)

        self.load_prediction_button = QPushButton("View")
        self.load_prediction_button.clicked.connect(self.view_data)
        # button layout
        view_button_layout = QHBoxLayout()
        view_button_layout.addStretch()
        view_button_layout.addWidget(self.load_prediction_button)
        view_button_layout.addStretch()

        outer_layout.addLayout(layout)
        outer_layout.addWidget(status_group)
        outer_layout.addLayout(view_button_layout)
        outer_layout.addWidget(video_generation_group)
        self.setLayout(outer_layout)

    def dir_dialog(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory")
        if dir_path:
            line_edit.setText(dir_path)

    def file_dialog(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Configuration File", "", "Config Files (*.yaml *.json *.mat *.npy *.pkl);;All Files (*)")
        if file_path:
            line_edit.setText(file_path)

    def index_dialog(self, line_edit):
        index_path, _ = QFileDialog.getOpenFileName(self, "Choose Frame Indexes", "", "Frame Indexes (*.txt *.npy *.csv *.xlsx);;All Files (*)")
        if index_path:
            line_edit.setText(index_path)

    def update_status(self, message, color="black", show_progress=False):
        """更新状态显示"""
        # 总是在命令行显示状态
        print(f"[Status] {message}")
        
        # 检查UI组件是否已初始化
        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.setText(message)
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        if hasattr(self, 'progress_bar') and self.progress_bar is not None:
            self.progress_bar.setVisible(show_progress)
        
        # 只在UI已经创建后才处理事件
        if hasattr(self, 'status_label') and self.status_label is not None:
            QApplication.processEvents()  # 立即更新UI

    # 内部数据加载方法（在工作线程中使用）
    def _load_data_internal(self):
        # 注意：这个方法在工作线程中运行，不能直接调用UI更新
        
        params = {}
        kwargs = {}

        if not os.path.exists(self.prediction_path.text()):
            raise FileNotFoundError("Prediction file does not exist!")

        if not os.path.exists(self.video_path.text()):
            raise FileNotFoundError("Video folder does not exist!")
        
        if not os.path.exists(self.param_path.text()):
            raise FileNotFoundError("Camera parameter file does not exist!")

        # this part for mat file
        if self.prediction_path.text().endswith(".mat"):
            try:
                # load the mat file
                pred, p_max, sampleID = self.load_mat_file(self.prediction_path.text())
                total_frame_num = sampleID.shape[-1]

                # Determine which frames to check
                if self.frame2check.text() and os.path.exists(self.frame2check.text()):
                    try:
                        index2check = self._read_index_file(self.frame2check.text()).astype(int)
                    except Exception as e:
                        raise Exception(f"Could not read index file: {e}")
                else:
                    index2check = np.arange(total_frame_num)

                frame_num2check = int(self.frame_num2check.text()) if self.frame_num2check.text() else None
                # sort the data by p
                if self.show_results.currentText() == "p ascending":
                    view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=True)
                elif self.show_results.currentText() == "p descending":
                    view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=False)
                else:
                    view_indexes = index2check[:frame_num2check] if frame_num2check else index2check

                # load the viewer window
                joints3d = pred[view_indexes, :, :]
                joints3d = joints3d.transpose(0, 2, 1)
                kwargs['joints3d'] = joints3d
                kwargs['p_max'] = p_max[view_indexes]

                params['frame_indexes'] = view_indexes
                params['frame_num2label'] = frame_num2check

            except Exception as e:
                raise Exception(f"Error loading MAT file: {str(e)}")

        elif self.prediction_path.text().endswith(".pkl"):
            try:
                # load the pickle file
                with open(self.prediction_path.text(), "rb") as f:
                    data = pkl.load(f)

                # get the index to check
                pred = data["keypoints_3d_pred"]
                p_max = data["pred_max"]
                frame_indexes = data["frames_idx"]
                skeleton = data["skeleton"]
                selected_joints = data["select_joints"]
                
                if not self.skeleton_path.text():
                    # make the skeleton file
                    skeleton_path = os.path.join(self.video_path.text(), "generated_skeleton.json")
                    os.makedirs(os.path.dirname(skeleton_path), exist_ok=True)
                    with open(skeleton_path, "w") as f:
                        json.dump(skeleton, f)
                    params['skeleton_path'] = skeleton_path
                else:
                    if not os.path.exists(self.skeleton_path.text()):
                        raise FileNotFoundError("Skeleton file does not exist!")
                    # check the skeleton file is valid
                    with open(self.skeleton_path.text(), "r") as f:
                        skeleton_given = json.load(f)
                    if skeleton_given != skeleton:
                        raise ValueError("The provided skeleton file does not match the data!")
                    params['skeleton_path'] = self.skeleton_path.text()

                filled_pred = self.fill_pred(pred, skeleton, selected_joints)
                # print(f"filled_pred: {filled_pred.shape}")

                # make view_indexes
                # NOTE the frame2check should be in the frame_indexes
                if self.frame2check.text() and os.path.exists(self.frame2check.text()):
                    try:
                        index2check = self._read_index_file(self.frame2check.text()).astype(int)
                    except Exception as e:
                        raise Exception(f"Could not read index file: {e}")
                    # check if the frames2check is in the frame_indexes
                    if not np.all(np.isin(index2check, frame_indexes)):
                        raise ValueError("The frames to check are not in the frame_indexes!")

                else:
                    index2check = frame_indexes

                frame_num2check = int(self.frame_num2check.text()) if self.frame_num2check.text() else None
                # sort the data by p
                if self.show_results.currentText() == "p ascending":
                    view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=True)
                elif self.show_results.currentText() == "p descending":
                    view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=False)
                else:
                    view_indexes = index2check[:frame_num2check] if frame_num2check else index2check

                # load the viewer window
                joints3d = filled_pred[view_indexes, :, :]

                # fill the p_max value here
                p_max = self.fill_p_max(p_max, skeleton, selected_joints)

                kwargs['joints3d'] = joints3d
                kwargs['p_max'] = p_max[view_indexes]

                params['frame_indexes'] = view_indexes
                params['frame_num2label'] = frame_num2check

                # FIXME check works
                if 'cam_names' in data.keys():
                    params['cam_names'] = data['cam_names']
                else:
                    params['cam_names'] = None

                params['joints_selected'] = selected_joints

            except Exception as e:
                raise Exception(f"Error loading PKL file: {str(e)}")
        else:
            raise ValueError("Unsupported file format! Only .mat and .pkl files are supported.")

        # add other params
        params['video_folder'] = self.video_path.text()
        if not 'skeleton_path' in params:
            params['skeleton_path'] = self.skeleton_path.text()

        if not 'joints_selected' in params:
            params['joints_selected'] = None

        params['cam_params'] = self.unpack_cam_params(self.param_path.text())
        
        return params, kwargs

    def load_data(self, callback=None):
        """线程化的数据加载方法"""
        if self.data_worker is not None and self.data_worker.isRunning():
            return  # 如果已经在加载数据，忽略新的请求
        
        self.update_status("Starting data loading...", "blue", True)
        self.data_worker = DataLoadingWorker(self)
        self.data_worker.progress_update.connect(self.update_status)
        
        if callback:
            self.data_worker.finished.connect(callback)
        
        self.data_worker.start()

    def on_data_loading_finished(self, success, params, kwargs, message):
        """数据加载完成的回调函数"""
        if success:
            self.update_status(message, "green")
            self.progress_bar.setVisible(False)
            # 可以在这里处理加载完成后的逻辑
            return params, kwargs
        else:
            self.update_status(f"Error: {message}", "red")
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", message)
            return None, None

    def view_data(self):
        """启动查看数据的流程"""
        def on_data_loaded(success, params, kwargs, message):
            if success:
                self.update_status("Opening viewer window...", "blue")
                self.viewer_window = ViewerWindow(self, params, **kwargs)
                self.viewer_window.show()
                # dismiss the current window
                self.hide()
                self.update_status("Viewer opened successfully!", "green")
            else:
                self.update_status(f"Error: {message}", "red")
                QMessageBox.critical(self, "Error", message)
        
        self.load_data(callback=on_data_loaded)

    def generate_video(self):
        """启动视频生成流程"""
        if self.video_worker is not None and self.video_worker.isRunning():
            return  # 如果已经在生成视频，忽略新的请求
        
        # 禁用生成按钮防止重复点击
        self.generate_video_button.setEnabled(False)
        
        def on_data_loaded_for_video(success, params, kwargs, message):
            if not success:
                self.update_status(f"Error: {message}", "red")
                QMessageBox.critical(self, "Error", message)
                self.generate_video_button.setEnabled(True)
                return

            # Check if frame rate is provided
            if not self.frame_rate.text():
                self.update_status("Error: Please provide a frame rate!", "red")
                QMessageBox.critical(self, "Error", "Please provide a frame rate!")
                self.generate_video_button.setEnabled(True)
                return

            try:
                fps = float(self.frame_rate.text())
            except ValueError:
                self.update_status("Error: Frame rate must be a number!", "red")
                QMessageBox.critical(self, "Error", "Frame rate must be a number!")
                self.generate_video_button.setEnabled(True)
                return

            # Get video format
            video_format = self.video_format.currentText()

            try:
                self.update_status("Initializing video maker...", "blue")
                print(f"[Info] Generating video with {len(params['frame_indexes'])} frames at {fps} FPS")
                
                # 解析缓存帧数设置
                cache_text = self.cache_frames.currentText()
                max_cache_frames = int(cache_text.split()[0])  # 提取数字部分
                print(f"[Info] Using max cache frames: {max_cache_frames}")
                
                # Create VideoMaker instance with necessary parameters
                video_maker = VideoMaker(
                    camParams=params['cam_params'],
                    video_folder=params['video_folder'],
                    skeleton=params['skeleton_path'],
                    joints3d=kwargs['joints3d'],
                    save_path=params['video_folder'],
                    frame_indexes=params['frame_indexes'],
                    cam_names=params['cam_names'],
                    video_format=video_format,
                    joints_selected=params['joints_selected'],
                    fps=fps,
                    max_cache_frames=max_cache_frames
                )
                
                # 启动视频生成工作线程
                self.video_worker = VideoGenerationWorker(video_maker)
                self.video_worker.progress_update.connect(self.update_status)
                self.video_worker.finished.connect(self.on_video_generation_finished)
                self.video_worker.start()
                
            except Exception as e:
                self.update_status(f"Error initializing video maker: {str(e)}", "red")
                QMessageBox.critical(self, "Error", f"Error initializing video maker: {str(e)}")
                self.generate_video_button.setEnabled(True)
        
        # 首先加载数据，然后在回调中启动视频生成
        self.update_status("Preparing video generation...", "blue", True)
        self.load_data(callback=on_data_loaded_for_video)

    def on_video_generation_finished(self, success, message):
        """视频生成完成的回调函数"""
        self.generate_video_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success:
            self.update_status("Video generation completed!", "green")
            print(f"[Success] Video generation completed successfully!")
            QMessageBox.information(self, "Success", "Video generation completed!")
        else:
            self.update_status(f"Error: {message}", "red")
            print(f"[Error] Video generation failed: {message}")
            QMessageBox.critical(self, "Error", f"Error generating video: {message}")
            
        # 清理工作线程
        if self.video_worker:
            self.video_worker.deleteLater()
            self.video_worker = None

    # check the pred shape, also fill the p_max value
    def fill_pred(self, pred ,skeleton, selected_joints):
        # transpose the pred 
        pred = pred.transpose(0, 2, 1)      # (N, 3, K) -> (N, K, 3)

        joints_num = len(skeleton["joint_names"])
        if len(selected_joints) == joints_num:
            return pred

        # print(f"debug 20250310 {joints_num}")
        selected_joints = np.array(selected_joints)

        # fill the pred with nan on the missing joints
        pred_filled = np.full((pred.shape[0], joints_num, 3), np.nan)
        
        for i in range(joints_num):
            # print(f"debug 20250310 {i}")
            if i in selected_joints:
                # find the index of the selected joint
                index = np.where(selected_joints == i)[0][0]
                pred_filled[:, i, :] = pred[:, index, :]

        return pred_filled


    def fill_p_max(self, p_max, skeleton, selected_joints):
        # p_max shape is (N, K), K is the number of joints
        p_max_filled = np.full((p_max.shape[0], len(skeleton["joint_names"])), np.nan)

        selected_joints = np.array(selected_joints)

        for i in range(len(skeleton["joint_names"])):
            if i in selected_joints:
                index = np.where(selected_joints == i)[0][0]
                p_max_filled[:, i] = p_max[:, index]

        return p_max_filled


    def on_viewer_window_close(self, event):
        self.show()
        event.accept()

    def cleanup_threads(self):
        """清理所有活动的工作线程"""
        if self.video_worker and self.video_worker.isRunning():
            self.video_worker.terminate()
            self.video_worker.wait()
        if self.data_worker and self.data_worker.isRunning():
            self.data_worker.terminate()
            self.data_worker.wait()

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        self.cleanup_threads()
        event.accept()

    def load_mat_file(self, mat_path):
        mat_data = loadmat(mat_path)
        ''' mat_data
        data: 0s, same shape with pred
        pred: 3D pose, N x 3 x K
        p_max: probability, N x k
        sampleID: the index of the frame in microseconds
        '''
        pred = mat_data['pred']
        p_max = mat_data['p_max']
        sampleID = mat_data['sampleID']

        return pred, p_max, sampleID

    def _read_index_file(self, file_path):
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path, header=None).values.flatten()
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, header=None).values.flatten()
        elif file_path.endswith('.txt'):
            return np.loadtxt(file_path, dtype=int)
        else:
            raise ValueError(f"Unsupported index file format: {file_path}")

    # load the cam params with this function
    def unpack_cam_params(self, calibration_path):
        cam_params = loadmat(calibration_path)['params']

        load_camParams = []
        for i in range(len(cam_params)):
            load_camParams.append(cam_params[i][0])

        return load_camParams

    def sort_by_p(self, p, index2check, frame_num2check, ascending=True):
        p = p[index2check]

        if p.ndim > 1:
            p = p.mean(axis=1)

        if ascending:
            sorted_p_index = np.argsort(p)
        else:
            sorted_p_index = np.argsort(p)[::-1]

        index = sorted_p_index[:frame_num2check]
        return index

# the main layout is a label3d widget
# this is only used for showing the prediction
class ViewerWindow(QMainWindow):
    def __init__(self, parent, params, *args, **kwargs):
        super().__init__()
        self.parent = parent
        self.params = params

        self.frame_indexes = params['frame_indexes']
        print(f"frame_indexes: {self.frame_indexes}")

        self.frame_num2label = params['frame_num2label']    # the index, which is quite important
        print(f"frame_num2label: {self.frame_num2label}")

        self.save_path = None
        self.qc_mode = False

        self.camParams = params['cam_params']
        self.video_folder = params['video_folder']
        self.skeleton_path = params['skeleton_path']        # you'd better have this to show the skeleton connection

        self.args = args
        self.kwargs = kwargs

        self.initUI()

    def initUI(self):
        self.setWindowTitle("3D Pose Viewer")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.label3d = Label3D(
            camParams=self.camParams,
            video_folder=self.video_folder,
            skeleton=self.skeleton_path,
            frame_num2label=self.frame_num2label,       # what this for?
            save_path=self.save_path,
            qc_mode=self.qc_mode,
            frame_indexes=self.frame_indexes,
            view_mode=True,
            **self.kwargs)

        layout.addWidget(self.label3d)


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.label3d.close()
            if self.parent:
                self.parent.show()
            event.accept()
        else:
            event.ignore()

    @Slot(str)
    def updateStatusBar(self, new_text):
        self.statusBar.showMessage(new_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ViewerLoader()
    viewer.show()
    sys.exit(app.exec())

