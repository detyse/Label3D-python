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

from scipy.io import loadmat
import traceback


class ViewerLoader(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setWindowTitle("Load config for 3D Pose Viewer")
        self.setGeometry(100, 100, 600, 300)

    def initUI(self):
        layout = QGridLayout()

        self.video_path_label = QLabel("Video Path: ")
        self.video_path = QLineEdit()
        self.video_path_browse = QPushButton("...")
        self.video_path_browse.clicked.connect(lambda: self.dir_dialog(self.video_path))
        layout.addWidget(self.video_path_label, 0, 0)
        layout.addWidget(self.video_path, 0, 1)
        layout.addWidget(self.video_path_browse, 0, 2)

        self.mat_path_label = QLabel("Mat Path: ")
        self.mat_path = QLineEdit()
        self.mat_path_browse = QPushButton("...")
        self.mat_path_browse.clicked.connect(lambda: self.file_dialog(self.mat_path))
        layout.addWidget(self.mat_path_label, 1, 0)
        layout.addWidget(self.mat_path, 1, 1)
        layout.addWidget(self.mat_path_browse, 1, 2)

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
        self.frame2check_label = QLabel("Frame to Check: ")
        self.frame2check = QLineEdit()
        self.frame2check_browse = QPushButton("...")
        self.frame2check_browse.clicked.connect(lambda: self.file_dialog(self.frame2check))
        layout.addWidget(self.frame2check_label, 4, 0)
        layout.addWidget(self.frame2check, 4, 1)
        layout.addWidget(self.frame2check_browse, 4, 2)

        self.frame_num2check_label = QLabel("Frame Num to Check: ")
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
        self.load_button = QPushButton("Load")
        self.load_button.clicked.connect(self.view_data)
        layout.addWidget(self.load_button, 6, 2)

        self.setLayout(layout)


    def dir_dialog(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory")
        if dir_path:
            line_edit.setText(dir_path)


    def file_dialog(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Configuration File", "", "Config Files (*.yaml *.json *.mat *.npy);;All Files (*)")
        if file_path:
            line_edit.setText(file_path)


    # the entry to the data view window, and load all the data here
    def view_data(self):
        params = {}
        kwargs = {}

        # load the mat file
        # get the index to check
        pred, p_max, sampleID = self.load_mat_file(self.mat_path.text())
        total_frame_num = sampleID.shape[-1]
        if self.frame2check.text():
            index2check = np.arange(total_frame_num)[np.load(self.frame2check.text())]
        else:
            index2check = np.arange(total_frame_num)

        if self.frame_num2check.text():
            frame_num2check = int(self.frame_num2check.text())
        else:
            frame_num2check = None

        # sort the data by p
        if self.show_results.currentText() == "p ascending":
            view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=True)
        elif self.show_results.currentText() == "p descending":
            view_indexes = self.sort_by_p(p_max, index2check, frame_num2check, ascending=False)
        else:
            view_indexes = index2check[:frame_num2check]

        # load the viewer window
        joints3d = pred[view_indexes, :, :]
        joints3d = joints3d.transpose(0, 2, 1)
        kwargs['joints3d'] = joints3d

        # pass the p value into the viewer window
        kwargs['p_max'] = p_max[view_indexes]

        # TODO: double check the params
        params['frame_indexes'] = view_indexes
        params['frame_num2label'] = frame_num2check

        # load cam params
        params['cam_params'] = self.unpack_cam_params(self.param_path.text())

        # add other params
        params['video_folder'] = self.video_path.text()
        params['skeleton_path'] = self.skeleton_path.text()

        self.viewer_window = ViewerWindow(params, **kwargs)
        self.viewer_window.show()

        # dismiss the current window
        self.close()

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
        # print(f"pred shape: {pred.shape}")
        # print(f"pred type: {type(pred)}")
        p_max = mat_data['p_max']
        # print(f"p_max shape: {p_max.shape}")
        # print(f"p_max type: {type(p_max)}")
        sampleID = mat_data['sampleID']
        # print(f"sampleID shape: {sampleID.shape}")
        # print(f"sampleID type: {type(sampleID)}")

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