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
        if "quality_control_on" in self.data:
            params['quality_control_on'] = self.data["quality_control_on"]
            # print("Quality control is on")
            self.build_up_frames_npy()
        else:
            params['quality_control_on'] = False
            print("Quality control is off")
            self.build_uniform_sample_indexes()

        params['cam_params'] = self.unpack_cam_params()
        # 
        params['video_folder'] = self.data["video_folder"]            # just one layer of folder
        params['skeleton_path'] = self.data["skeleton_path"]
        params['frame_num2label'] = self.data["frame_num2label"]
        params['save_path'] = self.data["save_path"]
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
        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        # get the index file
        index_file = os.path.join(video_folder, 'indexes.npy')
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

        frame_index = np.random.choice(total_frames, len(self.data["frame_num2label"])//2, replace=False)
        frame_index = frame_index.repeat(2)
        np.random.shuffle(frame_index)

        np.save(index_file, frame_index)

        cap.release()
        
        # build the frames npy for all the views
        frame_index = np.load(index_file)
        for view_folder in view_folders:
            video_path = os.path.join(video_folder, view_folder, '0.mp4')
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, view_folder, '0.avi')

            frames = frame_sampler(video_path, frame_index)
            npy_file = os.path.join(video_folder, view_folder, 'frames.npy')
            np.save(npy_file, frames)
        
        return


    # 方便后续处理
    def build_uniform_sample_indexes(self, ):
        video_folder = self.data["video_folder"]
        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        index_file = os.path.join(video_folder, 'uniform_indexes.npy')
        if os.path.exists(index_file):
            return

        # get the video path from the first view
        view_folder = view_folders[0]
        video_path = os.path.join(video_folder, view_folder, '0.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, view_folder, '0.avi')

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        label_num = self.data["frame_num2label"]
        if label_num == 0 or label_num > total_frames:
            label_num = total_frames

        indexes = np.linspace(0, total_frames-1, label_num, dtype=int)
        np.save(index_file, indexes)
        
        return 
    

    ## NOTE: no use now
    # # how to make sure the alignment of the order?
    # # use integer index 
    # # here is the function to get the video paths list from all views
    # # NOTE: do not need to change for the new dataset
    # def get_view_subfolders_from_video_folder(self, ):       
    #     video_folder = self.data["video_folder"]
    #     # get the subfolders 
    #     video_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]

    #     # get the video paths of all the videos
    #     video_paths = []        # add all the video then reverse the order
    #     for folder in video_folders:        # sort the list to align the order
    #         video_files = []
    #         view_dirnames = [f for f in os.listdir(os.path.join(video_folder, folder)) if os.path.isdir(os.path.join(video_folder, folder, f))]
    #         view_dirnames.sort()    # sort the list to align the order of files

    #         # print(view_dirnames)
    #         for view in view_dirnames:
            
    #             video_file = os.listdir(os.path.join(video_folder, folder, view))[0]
    #             video_file = os.path.join(video_folder, folder, view, video_file)
    #             video_files.append(video_file)

    #         video_paths.append(video_files)
    
    #     # transpose the list of list            
    #     trans_video_paths = list(map(list, zip(*video_paths)))
    #     return trans_video_paths


    # # build the frames numpy and save the index in the same folder when the quality control mode is on
    # # NOTE: need the video files, and if the npy is exist and the index is exist, skip the process for all the views in the subfolder
    # # 对于一些 video folder 中 views 不一致的情况（有的有index，有的没有），没有做任何处理
    # def build_up_frames_npy(self, ):
    #     # for each video path, each view, for the save folder, the index should keep the same
    #     video_folder = self.data["video_folder"]        # NOTE: this is a path, other are just folder name
    #     # iterate the subfolders of each subfolder
    #     sub_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]

    #     # for each subfolder, generate a frames index according to the frame_num2label
    #     for exp_folder in sub_folders:
    #         # get view folders in the exp
    #         view_folders = [f for f in os.listdir(os.path.join(video_folder, exp_folder)) if os.path.isdir(os.path.join(video_folder, exp_folder, f))]
    #         # sort the view folders
    #         view_folders.sort()
            
    #         current_frame_index = None
    #         for i, view_folder in enumerate(view_folders):
    #             # if there is the npy file and the index file, skip the process
    #             npy_file = os.path.join(video_folder, exp_folder, view_folder, 'frames.npy')
    #             index_file = os.path.join(video_folder, exp_folder, view_folder, 'frames_index.npy')

    #             if os.path.exists(npy_file) and os.path.exists(index_file):
    #                 break

    #             # FIXME temp code for loading the previous index
    #             if os.path.exists(npy_file):
    #                 break

    #             elif os.path.exists(index_file) and i == 0:     # given the index file, load the index
    #                 current_frame_index = np.load(index_file)
    #                 # NOTE: an dangerous operation, if the index not follow the format

    #             # else, get the video file(0.mp4 / 0.avi) build the frames npy and save the index 
    #             video_path = os.path.join(video_folder, exp_folder, view_folder, '0.mp4')
    #             if not os.path.exists(video_path):
    #                 video_path = os.path.join(video_folder, exp_folder, view_folder, '0.avi')
                
    #             if i == 0:
    #                 # read video
    #                 cap = cv2.VideoCapture(video_path)
    #                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #                 # build the frames index
    #                 frame_index = np.random.choice(total_frames, len(self.data["frame_num2label"])//2, replace=False)
    #                 frame_index = frame_index.repeat(2)
    #                 np.random.shuffle(frame_index)

    #                 current_frame_index = frame_index
    #             else:
    #                 frame_index = current_frame_index

    #             # save the index
    #             np.save(index_file, frame_index)

    #             # build the frames npy
    #             frames = frame_sampler(video_path, frame_index)

    #             np.save(npy_file, frames)

    #             cap.release()
    #     return 


import cv2
import numpy as np
# function or class? function
# this function for building a frames npy for labeling
# NOTE: do not need to take care of the multi folder problem
def frame_sampler(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)

    # get the frame size of the video
    frame_width = int(cap.get(3))       # no use
    frame_height = int(cap.get(4))      # no use
    # print(frame_width, frame_height)

    # get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # no use
    index_length = len(frame_index)                             # no use
    
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