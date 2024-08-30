# update 20240716 
# 1. find the bug of cannot delete the first joint 
# 2. add the function that the qc are not require to label all the frames
# 3. change the storage place of sampled frames
# 4. save the qc passed indexes for usage

# new update 20240717
# 1. load cam params add the mat choose
# 2. change the "load config" button to "start" button
# 3. add load defined index option, the index is not required to be depulicated
# 4. change the "load config" button to "load history" button (the config file is the highest priority, should we change that?)
# 5. add preview module, which could repidly browse the video frames
# 6. put the frames npy and the indexes into the output 
# 7. the quality control is allowed to implement when the frames are not all labeled
# 8. show the current frame index 
# 9. save the qc passed indexes for usage
# 10. concat the frames of all views into a single file
# 11. GUI update, change the GUI: 
#       the config window, add the load method to load the defined indexes  
#       also do not block when loading the frames
#       main window, add preview button at the bottom
# 12. change some button
# 13. change the video loading in the load config, instead of animator

# a temp stable vision

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


class ConfigWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    # a new grid layout for better look and add the quality control option, as a radio button i think
    # alsowide enough to show the path
    def initUI(self):
        # Create a grid layout
        layout = QGridLayout()

        # video folder path
        self.video_path_label = QLabel("Video Folder: ")
        self.video_path = QLineEdit()
        self.video_path_browse = QPushButton("...")
        self.video_path_browse.clicked.connect(lambda: self.dir_dialog(self.video_path))
        layout.addWidget(self.video_path_label, 0, 0)  # row 0, column 0
        layout.addWidget(self.video_path, 0, 1)        # row 0, column 1
        layout.addWidget(self.video_path_browse, 0, 2) # row 0, column 2

        # cam params file path
        self.cam_params_label = QLabel("Camera Params File: ")
        self.cam_params = QLineEdit()
        self.cam_params_browse = QPushButton("...")
        self.cam_params_browse.clicked.connect(lambda: self.file_dialog(self.cam_params))
        layout.addWidget(self.cam_params_label, 1, 0)  # row 1, column 0
        layout.addWidget(self.cam_params, 1, 1)        # row 1, column 1
        layout.addWidget(self.cam_params_browse, 1, 2) # row 1, column 2

        # skeleton file path
        self.skeleton_path_label = QLabel("Skeleton File: ")
        self.skeleton_path = QLineEdit()
        self.skeleton_path_browse = QPushButton("...")
        self.skeleton_path_browse.clicked.connect(lambda: self.file_dialog(self.skeleton_path))
        layout.addWidget(self.skeleton_path_label, 2, 0)  # row 2, column 0
        layout.addWidget(self.skeleton_path, 2, 1)        # row 2, column 1
        layout.addWidget(self.skeleton_path_browse, 2, 2) # row 2, column 2
        
        # set the frame_num2label param
        self.frame_num2label_label = QLabel("Frame Number to Label: ")
        self.frame_num2label = QLineEdit()
        self.frame_num2label.setValidator(QIntValidator())
        layout.addWidget(self.frame_num2label_label, 3, 0) # row 3, column 0
        layout.addWidget(self.frame_num2label, 3, 1)       # row 3, column 1
        # Frame number to label could be empty; no column 2 widget needed here

        # add a load index, the index have higher priority then the frame number to label
        self.load_index_label = QLabel("Load Index: ")
        self.load_index = QLineEdit()
        self.load_index_browse = QPushButton("...")
        self.load_index_browse.clicked.connect(lambda: self.file_dialog(self.load_index))
        layout.addWidget(self.load_index_label, 4, 0)     # row 4, column 0
        layout.addWidget(self.load_index, 4, 1)           # row 4, column 1
        layout.addWidget(self.load_index_browse, 4, 2)    # row 4, column 2

        # set save path
        self.save_path_label = QLabel("Save Path: ")
        self.save_path = QLineEdit()
        self.save_path_label_browse = QPushButton("...")
        self.save_path_label_browse.clicked.connect(lambda: self.dir_dialog(self.save_path))
        layout.addWidget(self.save_path_label, 5, 0)       # row 4, column 0
        layout.addWidget(self.save_path, 5, 1)             # row 4, column 1
        layout.addWidget(self.save_path_label_browse, 5, 2) # row 4, column 2

        # set the config file path
        self.config_path_label = QLabel("Config File: ")
        self.config_path = QLineEdit()
        self.config_path_browse = QPushButton("...")
        self.config_path_browse.clicked.connect(lambda: self.file_dialog(self.config_path))
        layout.addWidget(self.config_path_label, 6, 0)     # row 5, column 0
        layout.addWidget(self.config_path, 6, 1)           # row 5, column 1
        layout.addWidget(self.config_path_browse, 6, 2)    # row 5, column 2
        # The config_path could be empty; thus, handle it accordingly

        # Set initial row and column stretches as needed
        layout.setColumnStretch(1, 1)  # Give more space to the middle column where line edits are

        # set a main layout and a button layout for quality control button and the load button
        button_layout = QHBoxLayout()

        # on a single row, add a radio button for quality control 
        self.quality_control = QRadioButton("quality control on")
        button_layout.addWidget(self.quality_control)

        # the load button
        self.load_button = QPushButton("Load Config")
        self.load_button.clicked.connect(self.write_config)     # update the config and load the config
        button_layout.addWidget(self.load_button)

        # add the loading indicator for the label3d widget
        self.loading_indicator = QLabel("Input the config and load the config")
        indicator_layout = QVBoxLayout()
        indicator_layout.addWidget(self.loading_indicator)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(indicator_layout)

        # set the default window size
        self.setGeometry(100, 100, 600, 300)
        self.setLayout(main_layout)

    # file dialog
    def file_dialog(self, line_edit):
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose Configuration File", "", "Config Files (*.yaml *.json *.mat *.npy);;All Files (*)")
        if file_path:
            line_edit.setText(file_path)
        # else:
        #     line_edit.setText('File not selected')

    # folder dialog
    def dir_dialog(self, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, "Choose Directory")
        if dir_path:
            line_edit.setText(dir_path)
        # else:
            # line_edit.setText('Directory not available')

    # write the selected config into yaml file for reference and load
    # connect to the load button
    # TODO: change the message box indicator to the QLabel indicator
    def write_config(self, ):           # should read all the config then write into a yaml file
        # unable the load button
        self.load_button.setEnabled(False)
        self.loading_indicator.setText("Loading...")
        self.loading_indicator.repaint()
        
        try: 
            config_path = self.config_path.text()

            # if the config path is exist, then just load the config file
            if config_path:
                self.yaml_path = config_path

                with open(config_path, 'r') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                self.load_config_thread()

            else:
                video_folder = self.video_path.text()
                cam_params = self.cam_params.text()
                skeleton_path = self.skeleton_path.text()
                save_path = self.save_path.text()
                frame_indexes = self.load_index.text()
                
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

                qc_mode = self.quality_control.isChecked()

                # write the config into a yaml file
                config = {
                    "video_folder": video_folder,           # the video list is defined in the video_folder
                    "cam_params": cam_params,
                    "skeleton_path": skeleton_path,
                    "frame_num2label": frame_num2label,
                    "save_path": save_path,
                    "quality_control_on": qc_mode,
                    "given_frame_indexes": frame_indexes,
                }

                yaml_path = os.path.join(save_path, "config.yaml")
                with open(yaml_path, 'w') as f:
                    yaml.dump(config, f)
                
                self.yaml_path = yaml_path
                # load the config file
                self.load_config_thread()
        
        except Exception as e:
            self.load_button.setEnabled(True)
            self.loading_indicator.setText("Error: " + str(e) + " === Please check the config file.")
            raise e
    
    # 
    # def load_config(self, ):
    #     yaml_path = self.yaml_path
    #     if yaml_path:
    #         loader = LoadYaml(yaml_path)
    #         params = loader.get_all_params()
    #         if params:
    #             self.loading_indicator.setText("Loading...")
    #             self.loading_indicator.repaint()
    #             # self.parent().startMainWindow(params)     # here is the loading function
    #             self.close()
    #         else:
    #             QMessageBox.warning(self, "Load Error", "Failed to load configuration.", QMessageBox.OK)
    #     else:
    #         QMessageBox.warning(self, "Error", "The config file is not valid.", QMessageBox.Ok)
    

    # a new load config running in another thread that would not block the main thread
    def load_config_thread(self, ):
        yaml_path = self.yaml_path
        if yaml_path:
            self.load_worker = LoadConfigWorker(yaml_path)
            self.load_thread = QThread()
            self.load_worker.moveToThread(self.load_thread)
            self.load_thread.started.connect(self.load_worker.load_config)
            self.load_worker.finished.connect(self.on_load_finished)
            self.load_worker.error.connect(self.on_load_error)
            self.load_thread.start()
            self.loading_indicator.setText("Loading...")
            self.loading_indicator.repaint()
        else:
            QMessageBox.warning(self, "Error", "The config file is not valid.", QMessageBox.Ok)


    # a load finish function
    @Slot(dict)
    def on_load_finished(self, params):
        self.parent().startMainWindow(params)
        self.load_thread.quit()
        self.load_thread.wait()
        self.close()


    @Slot(str)
    def on_load_error(self, error):
        QMessageBox.warning(self, "Load Error", error, QMessageBox.Ok)
        self.load_thread.quit()
        self.load_thread.wait()


# define the widget in the Application?
# This object holds the event loop of your application - the core loop which governs all user interaciton with the GUI
class MainApplication(QApplication):
    def __init__(self, args):
        super().__init__(args)
        self.configWidget = ConfigWidget()
        self.configWidget.show()
        self.configWidget.parent = lambda: self     # set parent of the configwidget as the application 
    
    # NOTE: where the params comes from?
    # this function will be called by the configWidget
    def startMainWindow(self, params):
        self.mainWindow = MainWindow(params)
        self.mainWindow.show()


# the worker for config load
# NOTE: if not qc mode, the frame is loaded in the animator class, will may still block the main thread
class LoadConfigWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, yaml_path):
        super().__init__()
        self.yaml_path = yaml_path

    @Slot()
    def load_config(self, ):
        loader = LoadYaml(self.yaml_path)
        params = loader.get_params()
        if params:
            self.finished.emit(params)
        else:
            self.error.emit("Failed to load configuration.")


class MainWindow(QMainWindow):
    def __init__(self, params, *args, **kwargs):
        super().__init__() 
        self.params = params
        self.camParams = params['cam_params']
        self.video_folder = params['video_folder']    # 
        self.skeleton_path = params['skeleton_path']
        self.frame_num2label = params['frame_num2label']
        self.save_path = params['save_path']
        self.qc_mode = params['quality_control_on']
        self.frame_indexes = params['frame_indexes']    # the frame indexes are not required to be depulicated
        
        # if the qc_mode is on, this index is a random sampled index with depulication
        # else the index is uniformly sampled
        # not set the default value for the frame_index, if the qc mode is on, the frame index should be read from the video folder
        # self.frame_index = params['frame_index']

        self.args = args
        self.kwargs = kwargs
        self.initUI()
    

    def initUI(self, ):
        self.setWindowTitle("3D Labeling Tool")
        self.setGeometry(100, 100, 800, 600)
        
        layout = QVBoxLayout()
        self.label3d = Label3D(camParams=self.camParams, video_folder=self.video_folder, skeleton=self.skeleton_path, frame_num2label=self.frame_num2label, save_path=self.save_path,
                               frame_indexes=self.frame_indexes, qc_mode=self.qc_mode)        # newly added params
        # the frame index will generate automatically in the video folder
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

        self.label3d.update_status.connect(self.updateStatusBar)


    def user_manual(self, ):
        # show a message box to show the show the user manual
        self.manualBox = QMessageBox()
        self.manualBox.setWindowTitle("User Manual")
        self.manualBox.setText("User Manual: \n\n"
                                "1. Using \"Ctrl + Left\" to label on the view if the joint is select \n"
                                "2. Push \"Q / E\" to select the joint \n"
                                "3. \"Ctrl + Right\" could delete the markers \n"
                                "4. Push \"D / A\" for next frame or last frame \n"
                                "5. Push S for label saving \n"
                                "6. Push \"Ctrl + R\" to clear create joint markers \n"
                                "7. Push \"Up / Down\" to change the jump speed \n"
                                "8. Push \"[ / ]\" to change the contrast of the view image \n"
                                "...")
        self.manualBox.exec()


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


def main():
    try:
        app = MainApplication(sys.argv)
        sys.exit(app.exec())
    except Exception as e:
        traceback.point_exc()       # to get the traceback error
        print(e)


if __name__ == "__main__":
    main()
