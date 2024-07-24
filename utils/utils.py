# helper functions for label 3d
# load skeleton and camera parameters into json format
import sys
import json
import csv
import cv2
import os

import numpy as np
from scipy.io import loadmat
from dataclasses import dataclass, field

import yaml


# transform mat to json to load 
def transform_mat_to_json_skeleton(mat_file, save_file):
    mat = loadmat(mat_file)
    color = mat['color']
    joint_names = [item.item() for item in mat['joint_names'][0]]
    joints_idx = mat['joints_idx']
    with open(save_file, 'w') as f:
        json.dump({'color': color.tolist(), 'joint_names': joint_names, 'joints_idx': joints_idx.tolist()}, f, indent=4)
    # return mat


def read_json_skeleton(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


# TODO: the config load should be fine if the the quality control is not in the yaml file, 
@dataclass
class LoadYaml:
    yaml_file: str = ""
    data: dict = field(default_factory=dict, init=False)


    def __post_init__(self, ):
        self.load_yaml()


    def load_yaml(self):
        try:
            with open(self.yaml_file, 'r') as f:
                self.data = yaml.safe_load(f)
            print(f"YAML data loaded: {self.data}")
        except Exception as e:
            print(f"Error loading YAML file: {e}")


    def get_all_params(self):
        params = {}
        
        params['quality_control_on'] = self.data["quality_control_on"]
    
        if params.get('quality_control_on'):
            self.build_up_frames_npy()
        else:
            self.build_uniform_sample_indexes()
            
        params['cam_params'] = self.unpack_cam_params()
        # 
        params['video_folder'] = self.data["video_folder"]            # just one layer of folder
        params['skeleton_path'] = self.data["skeleton_path"]
        params['frame_num2label'] = self.data["frame_num2label"]
        params['save_path'] = self.data["save_path"]

        # the frame indexes should be the indexes file in the save path
        save_folder = self.data["save_path"]
        index_file = os.path.join(save_folder, 'indexes.npy')
        params['frame_indexes'] = index_file        # here the frame index have a higher priority than the frame_num2label
        return params


    def unpack_cam_params(self, ):
        cam_params = loadmat(self.data["cam_params"])['params']
        
        load_camParams = []
        for i in range(len(cam_params)):
            load_camParams.append(cam_params[i][0])
        
        return load_camParams
    

    # no use now
    def get_video_folder(self, ):
        video_folder = self.data["video_folder"]        # the folder path
        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        # join the path
        view_folders = [os.path.join(video_folder, f) for f in view_folders]
        return view_folders
    

    def build_up_frames_npy(self, ):
        # could get/save the index in video folder
        video_folder = self.data["video_folder"]
        save_folder = self.data["save_path"]

        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        # get the index file
        index_file = os.path.join(save_folder, 'indexes.npy')
        if os.path.exists(index_file):
            print("Already have the index file")
            return
        
        # get the video path of the first view
        view_folder = view_folders[0]
        video_path = os.path.join(video_folder, view_folder, '0.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, view_folder, '0.avi')

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        cap.release()

        # if did not set the frame index, build the index, else use the given index
        if self.data["given_frame_indexes"] is None:
            frame_index = np.random.choice(total_frames, self.data["frame_num2label"], replace=False)
            np.random.shuffle(frame_index)

        else:
            frame_index = np.load(self.data["given_frame_indexes"])
            # check the index avaliable, if the max and the min index is out of the total frames
            if frame_index.max() >= total_frames or frame_index.min() < 0:
                print("The frame index is out of the total frames")
                raise ValueError("The frame index is out of the total frames")

            # TODO: add other frame check here, like the index could not have duplicated values

        # TODO: add a check for npy file
        frame_index = frame_index.repeat(2)
        # shuffle the index
        np.random.shuffle(frame_index)
        np.save(index_file, frame_index)
        
        for view_folder in view_folders:
            video_path = os.path.join(video_folder, view_folder, '0.mp4')
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, view_folder, '0.avi')

            frames = frame_sampler(video_path, frame_index)
            npy_folder = os.path.join(save_folder, "frames", view_folder)
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)
            npy_file = os.path.join(npy_folder, 'frames.npy')
            np.save(npy_file, frames)
        
        return


    # 方便后续处理
    def build_uniform_sample_indexes(self, ):
        video_folder = self.data["video_folder"]
        save_folder = self.data["save_path"]

        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        index_file = os.path.join(video_folder, 'uniform_indexes.npy')
        if os.path.exists(index_file):
            print("Already have the index file")
            return

        # get the video path from the first view
        view_folder = view_folders[0]
        video_path = os.path.join(video_folder, view_folder, '0.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, view_folder, '0.avi')

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.release()

        # if there is the given indexes
        if self.data["given_frame_indexes"] is None:
            label_num = self.data["frame_num2label"]
            if label_num == 0 or label_num > total_frames:
                label_num = total_frames

            indexes = np.linspace(0, total_frames-1, label_num, dtype=int)
            np.save(index_file, indexes)

        else: 
            indexes = np.load(self.data["given_frame_indexes"])
            if indexes.max() >= total_frames or indexes.min() < 0:
                print("The frame index is out of the total frames")
                raise ValueError("The frame index is out of the total frames")

            # TODO: could also add other check here
            np.save(index_file, indexes)
        
        # also save the frames
        for view_folder in view_folders:
            video_path = os.path.join(video_folder, view_folder, '0.mp4')
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, view_folder, '0.avi')

            frames = frame_sampler(video_path, indexes)
            npy_folder = os.path.join(save_folder, "frames", view_folder)
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)
            npy_file = os.path.join(npy_folder, 'frames.npy')
            np.save(npy_file, frames)

        return 
    

import cv2
import numpy as np
# function or class? function
# this function for building a frames npy for labeling
# NOTE: do not need to take care of the multi folder problem
def frame_sampler(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)

    # # get the frame size of the video
    # frame_width = int(cap.get(3))       # no use
    # frame_height = int(cap.get(4))      # no use
    # # print(frame_width, frame_height)

    # # get the total number of frames
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # no use
    # index_length = len(frame_index)                             # no use
    
    frames = []
    # write the frames into the npy file
    # the frame index is not sorted, should 
    for index in frame_index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)     # 反复横跳 

        ret = cap.grab()
        if not ret:
            break
        
        ret, frame = cap.retrieve()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(rgb_frame)
    cap.release()
    
    frames = np.array(frames)

    # ? should save the npy file for future analysis ?  should not be here
    # dir_path = os.path.dirname(video_path)
    # if save:
        # np.save(os.path.join(dir_path, 'frames.npy'), frames)
    
    return frames
