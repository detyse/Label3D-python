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


# struct helper classes
# @dataclass
# class CameraParams:
#     '''
#     r: 3 x 3
#     t: 3 x 1 or 1 x 3
#     K: 3 x 3
#     RDistort: 1 x 3
#     TDistort: 1 x 2
#     '''
#     r: np.ndarray = np.zeros((3, 3))
#     t: np.ndarray = np.zeros((3, 1))
#     K: np.ndarray = np.zeros((3, 3))
#     RDistort: np.ndarray = np.zeros((1, 3))
#     TDistort: np.ndarray = np.zeros((1, 2))

#     def __post_init__(self, mat_file):
#         mat_data = loadmat(mat_file)
#         self.r = mat_data['r']
#         self.t = mat_data['t'].transpose()  # check bug
#         self.K = mat_data['K']
#         self.RDistort = mat_data['RDistort']
#         self.TDistort = mat_data['TDistort']


# @dataclass
# class Skeleton:
#     '''
    
#     '''
#     a = 1

# @dataclass
# # to organize the parameters all together
# # the function to read the yaml file and load the parameters
# class Loader():

#     index: list = field(default_factory=list)
#     # ERROR: 不能为字段赋予一个可变的默认值，比如列表或字典 []
#     # change to: list = field(default_factory=list)
    
#     video_paths: list = field(default_factory=list)
#     cam_params: list = field(default_factory=list)
#     skeleton: str = ""  # json file path 

#     frame_list: list = ""       # a list to store wanted frames index

#     def __post_init__(self, index, video_paths, cam_params, skeletion_path):
#         assert len(index) == len(video_paths) == len(cam_params), "Length of index, video_paths, and cam_params must be the same."
#         self.index = index
#         self.video_paths = video_paths
#         self.cam_params = cam_params
#         self.skeleton = read_json_skeleton(skeletion_path)


# a class to load the yaml file in to parameters
# @dataclass
# class LoadYaml():
#     yaml_file: str = ""

#     def __post_init__(self, yaml_file):
#         with open(yaml_file, 'r') as f:
#             data = yaml.safe_load(f)
#         self.data = data

#     def unpack_cam_params(self):
#         cam_params = loadmat(self.data["cam_params_path"])['params']

#         load_camParams = []
#         for i in range(len(cam_params)):
#             load_camParams.append(cam_params[i][0])

#         self.data['camParams'] = load_camParams

#     def get_params(self):
#         return self.data
    

# add a 
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
        except Exception as e:
            print(f"Error: {e}")


    def get_all_params(self):
        params = {}
        params['cam_params'] = self.unpack_cam_params()
        params['video_folder'] = self.get_videos_from_video_folder()
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
    
    # how to make sure the alignment of the order?
    # use integer index 
    def get_videos_from_video_folder(self, ):
        video_folder = self.data["video_folder"]
        # get the subfolders 
        video_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        # print(video_folders)

        # get the video paths of all the videos
        video_paths = []        # add all the video then reverse the order
        for folder in video_folders:        # sort the list to align the order
            video_files = []
            view_dirnames = [f for f in os.listdir(os.path.join(video_folder, folder)) if os.path.isdir(os.path.join(video_folder, folder, f))]
            view_dirnames.sort()

            # print(view_dirnames)
            for view in view_dirnames:
                video_file = os.path.join(video_folder, folder, view, "0.mp4")
                video_files.append(video_file)
            
            video_paths.append(video_files)
    
        # transpose the list of list
        trans_video_paths = list(map(list, zip(*video_paths)))
        return trans_video_paths