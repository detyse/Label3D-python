# helper functions for label 3d
# load skeleton and camera parameters into json format
import sys
import json
import csv
import cv2

import numpy as np
from scipy.io import loadmat
from dataclasses import dataclass, field

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

@dataclass
class Loader():

    index: list = field(default_factory=list)
    # ERROR: 不能为字段赋予一个可变的默认值，比如列表或字典 []
    # change to: list = field(default_factory=list)
    
    video_paths: list = field(default_factory=list)
    cam_params: list = field(default_factory=list)
    skeleton: str = ""  # json file path 

    frame_list: list = ""       # a list to store wanted frames index

    def __post_init__(self, index, video_paths, cam_params, skeletion_path):
        assert len(index) == len(video_paths) == len(cam_params), "Length of index, video_paths, and cam_params must be the same."
        self.index = index
        self.video_paths = video_paths
        self.cam_params = cam_params
        self.skeleton = read_json_skeleton(skeletion_path)
