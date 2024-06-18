# here is the plan, time is 2024-05-26
# we use the yaml to manage the config file for the 3D label, instead of a GUI
# and we add a multi video method to label multiple videos at once (which requires properlly handling the animator loading )
# and we add the original label saving funtion 
# also add the video 
# add we test our videos and the params
# update: load the video frames at the first place, do not nest the loader in to deeper layer

import os
import sys
import cv2
import numpy as np
import yaml

from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *

from utils.utils import read_json_skeleton, LoadYaml

# from label3d_v2 import Label3D
from label3d import Label3D

from scipy.io import loadmat
import traceback


class LoadConfigDialog(QDialog):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self, ):
        layout = QVBoxLayout(self)

        path_layout = QHBoxLayout()
        self.label = QLabel("Config File: ")
        self.path = QLineEdit()
        self.browse = QPushButton("...")

        path_layout.addWidget(self.label)
        path_layout.addWidget(self.path)
        path_layout.addWidget(self.browse)

        self.browse.clicked.connect(self.file_dialog)

        layout.addLayout(path_layout)

        self.load = QPushButton("Load Config")
        self.load.clicked.connect(self.load_config)

        layout.addWidget(self.load)

        self.setLayout(layout)


    def file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Configuration File", "", "Config Files (*.yaml *.json);;All Files (*)")
        if file_path:
            self.path.setText(file_path)
        else:
            self.path.setText('File not selected')


    def load_config(self, ):
        file_path = self.path.text()
        if file_path:
            self.accept()
        else:
            QMessageBox.warning(self, self, "Error", "Please select a valid directory.", QMessageBox.Ok)
    

    def getConfigPath(self, ):
        return self.path.text()


class ConfigWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()


    def initUI(self, ):
        layout = QVBoxLayout(self)
        self.setLayout(layout)
        
        # TODO: change to grid layout for a better look
        # 

        # video folder path
        video_path_layout = QHBoxLayout()
        self.video_path_label = QLabel("Video Folder: ")
        self.video_path = QLineEdit()
        self.video_path_browse = QPushButton("...")
        self.video_path_browse.clicked.connect(lambda: self.dir_dialog(self.video_path))
        video_path_layout.addWidget(self.video_path_label)
        video_path_layout.addWidget(self.video_path)
        video_path_layout.addWidget(self.video_path_browse)

        # cam params file path
        cam_params_layout = QHBoxLayout()
        self.cam_params_label = QLabel("Camera Params File: ")
        self.cam_params = QLineEdit()
        self.cam_params_browse = QPushButton("...")
        self.cam_params_browse.clicked.connect(lambda: self.file_dialog(self.cam_params))
        cam_params_layout.addWidget(self.cam_params_label)
        cam_params_layout.addWidget(self.cam_params)
        cam_params_layout.addWidget(self.cam_params_browse)

        # skeleton file path
        skeleton_path_layout = QHBoxLayout()
        self.skeleton_path_label = QLabel("Skeleton File: ")
        self.skeleton_path = QLineEdit()
        self.skeleton_path_browse = QPushButton("...")
        self.skeleton_path_browse.clicked.connect(lambda: self.file_dialog(self.skeleton_path))
        skeleton_path_layout.addWidget(self.skeleton_path_label)
        skeleton_path_layout.addWidget(self.skeleton_path)
        skeleton_path_layout.addWidget(self.skeleton_path_browse)

        # set the frame_num2label param
        frame_num2label_layout = QHBoxLayout()
        self.frame_num2label_label = QLabel("Frame Number to Label: ")
        self.frame_num2label = QLineEdit()
        self.frame_num2label.setValidator(QIntValidator())      # validate the input is an integer
        frame_num2label_layout.addWidget(self.frame_num2label_label)
        frame_num2label_layout.addWidget(self.frame_num2label)
        # frame number to label could be empty

        # set save path
        save_path_layout = QHBoxLayout()
        self.save_path_label = QLabel("Save Path: ")
        self.save_path = QLineEdit()
        self.save_path_label_browse = QPushButton("...")
        self.save_path_label_browse.clicked.connect(lambda: self.dir_dialog(self.save_path))
        save_path_layout.addWidget(self.save_path_label)
        save_path_layout.addWidget(self.save_path)
        save_path_layout.addWidget(self.save_path_label_browse)

        # set the config file path 
        config_path_layout = QHBoxLayout()
        self.config_path_label = QLabel("Config File: ")
        self.config_path = QLineEdit()
        self.config_path_browse = QPushButton("...")
        self.config_path_browse.clicked.connect(lambda: self.file_dialog(self.config_path))
        # NOTE: if the config file is set, using the config file to load the params
        config_path_layout.addWidget(self.config_path_label)
        config_path_layout.addWidget(self.config_path)
        config_path_layout.addWidget(self.config_path_browse)
        # the config_path could be empty
        
        # the load button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.write_config)

        layout.addLayout(video_path_layout)
        layout.addLayout(cam_params_layout)
        layout.addLayout(skeleton_path_layout)
        layout.addLayout(frame_num2label_layout)
        layout.addLayout(save_path_layout)

        layout.addLayout(QHBoxLayout())
        layout.addLayout(config_path_layout)
        layout.addWidget(self.load_button)


    # file dialog
    def file_dialog(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Configuration File", "", "Config Files (*.yaml *.json);;All Files (*)")
        if file_path:
            line_edit.setText(file_path)
        else:
            line_edit.setText('File not selected')

    # folder dialog
    def dir_dialog(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory")
        if dir_path:
            line_edit.setText(dir_path)
        else:
            line_edit.setText('Directory not available')

    # write the selected config into yaml file for reference and load
    # connect to the load button
    def write_config(self, ):           # should read all the config then write into a yaml file
        config_path = self.config_path.text()
        if config_path:
            self.yaml_path = config_path
            self.load_config()
        
        else:
            video_folder = self.video_path.text()
            cam_params = self.cam_params.text()
            skeleton_path = self.skeleton_path.text()
            save_path = self.save_path.text()

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # if any is empty, then show a warning
            if not save_path or not video_folder or not cam_params or not skeleton_path:
                QMessageBox.warning(self, "Warning", "Please fill the config", QMessageBox.Ok)
                return

            if not self.frame_num2label.text():
                frame_num2label = 0
            else:
                frame_num2label = int(self.frame_num2label.text())

            # write the config into a yaml file
            config = {
                "video_folder": video_folder,
                "cam_params": cam_params,
                "skeleton_path": skeleton_path,
                "frame_num2label": frame_num2label,
                "save_path": save_path,
            }

            yaml_path = os.path.join(save_path, "config.yaml")
            with open(yaml_path, 'w') as f:
                yaml.dump(config, f)

            self.yaml_path = yaml_path
            
            # load the config file
            self.load_config()
    
    # 
    def load_config(self, ):
        yaml_path = self.yaml_path
        if yaml_path:
            loader = LoadYaml(yaml_path)
            params = loader.get_all_params()
            if params:
                self.parent().startMainWindow(params)
                self.close()
            else:
                QMessageBox.warning(self, "Load Error", "Failed to load configuration.", QMessageBox.OK)
        else:
            QMessageBox.warning(self, "Error", "The config file is not valid.", QMessageBox.Ok)


class MainWindow(QMainWindow):
    def __init__(self, params, *args, **kwargs):
        super().__init__() 
        self.params = params
        self.camParams = params['cam_params']
        self.videos = params['video_folder']
        print(self.videos)
        self.skeleton_path = params['skeleton_path']
        self.frame_num2label = params['frame_num2label']
        self.save_path = params['save_path']
        
        self.args = args
        self.kwargs = kwargs
        self.initUI()
   

    def initUI(self, ):
        self.setWindowTitle("3D Labeling Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.label3d = Label3D(camParams=self.camParams, videos=self.videos, skeleton=self.skeleton_path, frame_num2label=self.frame_num2label, save_path=self.save_path)
        layout.addWidget(self.label3d)
        
        self.setLayout(layout)
        self.setCentralWidget(self.label3d)

        self.menuBar = self.menuBar()
        helpMenu = self.menuBar.addMenu("Help")
        self.showManual = QAction("User Manual", self)
        helpMenu.addAction(self.showManual)

        self.showManual.triggered.connect(self.user_manual)
        
        self.statusBar = self.statusBar()
        self.statusBar.showMessage("Ready") 


    def user_manual(self, ):
        # show a message box to show the show the user manual
        self.manualBox = QMessageBox()
        self.manualBox.setWindowTitle("User Manual")
        self.manualBox.setText("User Manual: \n\n"
                                "1. Using Ctrl + LeftMouse to label on the view if the joint is select \n"
                                "2. Push Tab move to next joint, or click the joint on left bar to select a joint \n"
                                "3. Push T to auto label the reset points of the joint \n"
                                "...")
        self.manualBox.exec()


    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message', "Are you sure to quit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.label3d.close()
            event.accept()
        else:
            event.ignore()


# define the widget in the Application?
# This object holds the event loop of your application - the core loop which governs all user interaciton with the GUI
class MainApplication(QApplication):
    def __init__(self, args):
        super().__init__(args)
        self.configWidget = ConfigWidget()
        self.configWidget.show()
        self.configWidget.parent = lambda: self     # set parent of the configwidget as the application 
    
    def startMainWindow(self, params):
        self.mainWindow = MainWindow(params)
        self.mainWindow.show()


def main():
    try:
        app = MainApplication(sys.argv)
        sys.exit(app.exec())
    except Exception as e:
        traceback.point_exc()
        print(e)


if __name__ == "__main__":
    main()

