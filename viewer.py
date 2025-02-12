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
from animator.videomaker import VideoMaker

from scipy.io import loadmat
import pickle as pkl
import traceback

import json


class ViewerLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("3D Pose Viewer")
        self.setGeometry(100, 100, 600, 300)

    # TODO add the video generation function
    def initUI(self):
        outer_layout = QVBoxLayout()

        layout = QGridLayout()

        self.video_path_label = QLabel("Video Path: ")
        self.video_path = QLineEdit(r"D:/YL_Wang/Pose_dta/250108_comparison/M3_datasets_0109/test_2/frames")
        self.video_path_browse = QPushButton("...")
        self.video_path_browse.clicked.connect(lambda: self.dir_dialog(self.video_path))
        layout.addWidget(self.video_path_label, 0, 0)
        layout.addWidget(self.video_path, 0, 1)
        layout.addWidget(self.video_path_browse, 0, 2)

        self.prediction_path_label = QLabel("Predicted Path: ")
        self.prediction_path = QLineEdit(r"D:/YL_Wang/Pose_dta/250108_comparison/M3_datasets_0109/test_2/frames/spose.pkl")
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
        self.param_path = QLineEdit(r"D:/YL_Wang/Pose_dta/250108_comparison/M3_datasets_0109/test_2/frames/cam_params.mat")
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
        self.frame2check_browse.clicked.connect(lambda: self.file_dialog(self.frame2check))
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
        self.generate_video_button = QPushButton("Generate")
        self.generate_video_button.clicked.connect(self.generate_video)

        video_group_layout.addWidget(frame_rate_label)
        video_group_layout.addWidget(self.frame_rate)
        video_group_layout.addStretch()
        video_group_layout.addWidget(video_format_label)
        video_group_layout.addWidget(self.video_format)
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

    # the entry to the data view window, and load all the data here
    def load_data(self):
        params = {}
        kwargs = {}

        if not os.path.exists(self.prediction_path.text()):
            QMessageBox.critical(self, "Error", "Prediction file does not exist!")
            return

        if not os.path.exists(self.video_path.text()):
            QMessageBox.critical(self, "Error", "Video folder does not exist!")
            return
        
        if not os.path.exists(self.param_path.text()):
            QMessageBox.critical(self, "Error", "Camera parameter file does not exist!")
            return

        if self.prediction_path.text().endswith(".mat"):
            try:
                # load the mat file
                pred, p_max, sampleID = self.load_mat_file(self.prediction_path.text())
                total_frame_num = sampleID.shape[-1]

                # Determine which frames to check
                if self.frame2check.text() and os.path.exists(self.frame2check.text()):
                    index2check = np.arange(total_frame_num)[np.load(self.frame2check.text())]
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
                QMessageBox.critical(self, "Error", f"Error loading MAT file: {str(e)}")
                return

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
                        QMessageBox.critical(self, "Error", "Skeleton file does not exist!")
                        return
                    # check the skeleton file is valid
                    with open(self.skeleton_path.text(), "r") as f:
                        skeleton_given = json.load(f)
                    if skeleton_given != skeleton:
                        QMessageBox.critical(self, "Error", "The provided skeleton file does not match the data!")
                        return
                    params['skeleton_path'] = self.skeleton_path.text()

                filled_pred = self.fill_pred(pred, skeleton, selected_joints)
                # print(f"filled_pred: {filled_pred.shape}")

                # make view_indexes
                # NOTE the frame2check should be in the frame_indexes
                if self.frame2check.text() and os.path.exists(self.frame2check.text()):
                    index2check = np.load(self.frame2check.text())
                    # check if the frames2check is in the frame_indexes
                    if not np.all(np.isin(index2check, frame_indexes)):
                        QMessageBox.critical(self, "Error", "The frames to check are not in the frame_indexes!")
                        return

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
                QMessageBox.critical(self, "Error", f"Error loading PKL file: {str(e)}")
                traceback.print_exc()
                return
        else:
            QMessageBox.critical(self, "Error", "Unsupported file format! Only .mat and .pkl files are supported.")
            return

        # add other params
        params['video_folder'] = self.video_path.text()
        if not 'skeleton_path' in params:
            params['skeleton_path'] = self.skeleton_path.text()

        if not 'joints_selected' in params:
            params['joints_selected'] = None

        params['cam_params'] = self.unpack_cam_params(self.param_path.text())
        return params, kwargs

    def view_data(self, ):
        params, kwargs = self.load_data()

        self.viewer_window = ViewerWindow(params, **kwargs)
        self.viewer_window.show()

        # dismiss the current window
        self.close()

    ### generate video part... use the video generator class 
    def generate_video(self):
        # load the params and kwargs
        params, kwargs = self.load_data()

        # Check if frame rate is provided
        if not self.frame_rate.text():
            QMessageBox.critical(self, "Error", "Please provide a frame rate!")
            return

        try:
            fps = float(self.frame_rate.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Frame rate must be a number!")
            return

        # Get video format
        video_format = self.video_format.currentText()

        # Create VideoMaker instance with necessary parameters
        # and have nothing with the viewer_window, just using the same params with the label3d class
        try:
            video_maker = VideoMaker(
                camParams=params['cam_params'],
                video_folder=params['video_folder'],
                skeleton=params['skeleton_path'],
                joints3d=kwargs['joints3d'],
                save_path=params['video_folder'],
                frame_indexes=params['frame_indexes'],
                cam_names=params['cam_names'],
                video_format=video_format,
                joints_selected=params['joints_selected'],          # key error, and we do not need this, we fill the joints3d data already
                fps=fps
            )

            # Create save directory if it doesn't exist
            # os.makedirs(os.path.join(params['video_folder'], "generated_videos"), exist_ok=True)

            # Generate the video
            video_maker.make_video()
            QMessageBox.information(self, "Success", "Video generation completed!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating video: {str(e)}\n{traceback.format_exc()}")
            return

    # check the pred shape
    def fill_pred(self, pred, skeleton, selected_joints):
        # transpose the pred 
        pred = pred.transpose(0, 2, 1)      # (N, 3, K) -> (N, K, 3)

        joints_num = len(skeleton["joint_names"])
        if len(selected_joints) == joints_num:
            return pred

        # fill the pred with nan on the missing joints
        pred_filled = np.full((pred.shape[0], joints_num, 3), np.nan)
        
        for i in range(joints_num):
            if i in selected_joints:
                # find the index of the selected joint
                index = np.where(selected_joints == i)[0][0]
                pred_filled[:, i, :] = pred[:, index, :]

        return pred_filled

    def on_viewer_window_close(self, event):
        self.show()
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
    def __init__(self, params, *args, **kwargs):
        super().__init__()
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