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

# TODO: check the yaml load and add the error handling
# in the loading stage, the order is frames/videos -> index -> quality control
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

    
    # FIXME: consider the reload, if the indexes and frames are already stored, we could just load them directly
    def get_params(self, ):
        params = {}
        params['cam_params'] = self.unpack_cam_params()
        params['skeleton_path'] = self.data["skeleton_path"]
        params['save_path'] = self.data["save_path"]
        params['video_folder'] = self.data["video_folder"]

        # create the index and the frames here
        save_folder = self.data["save_path"]

        # NOTE: if the frames is stored in npy file, just load the npy file?
        # no, we should handle the npy and the mp4 in the same way
        # build the frames file, if the index is given,
        # NOTE: if the index is given, the frame_num2label is not used, 
        # and the index should be stored in a numpy array and should be unduplicated and in the range of the total frames
        index_file = self.data["given_frame_indexes"]
        
        params['frame_num2label'] = self.data["frame_num2label"]
        sample_num = self.data["frame_num2label"]
        
        params['quality_control_on'] = self.data["quality_control_on"]

        # not the first loading, use the indexes file to check that? 
    
        the_index_file = os.path.join(save_folder, "indexes.npy")
        if os.path.exists(the_index_file):
            # load the index file
            indexes = np.load(the_index_file)
        else:
            # get index, no duplicate
            if params.get('frame_indexes'):
                indexes = np.load(index_file)
            else: 
                # check the index file 
                indexes, total_frames = self.get_index(sample_num)

            # assert index
            assert indexes.max() < total_frames and indexes.min() >= 0, "The frame index is out of the total frames"

        if params.get('quality_control_on'):
            final_indexes = self.build_qc_index(indexes)
            # save the index file
        else:
            final_indexes = indexes

        # build the frames
        self.build_frames(final_indexes)

        # save the index file should be the last step to load the params
        np.save(the_index_file, final_indexes)
        params['frame_indexes'] = the_index_file
            
        return params


    def unpack_cam_params(self, ):
        cam_params = loadmat(self.data["cam_params"])['params']
        
        load_camParams = []
        for i in range(len(cam_params)):
            load_camParams.append(cam_params[i][0])
        
        return load_camParams
    

    # function to build up the index, when the index is not given
    # and the index should uniform sampled from the total frames
    # before quality control, 
    def get_index(self, sample_num=None):
        video_folder = self.data["video_folder"]

        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        the_folder = view_folders[0]
        # the video path accept '.mp4' and '.avi' and '.npy'
        video_path = os.path.join(video_folder, the_folder, '0.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, the_folder, '0.avi')
        
        # if the video path is exist, read the video using cv2
        if os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            video_path = os.path.join(video_folder, the_folder, '0.npy')
            total_frames = np.load(video_path).shape[0]     # the frames stacked in the first dimention
            if not os.path.exists(video_path):
                raise ValueError("The video path is not available, please check the video path")
            
        # 
        if sample_num is None or sample_num == 0 or sample_num > total_frames:
            indexes = np.arange(total_frames)
        else:
            indexes = np.linspace(0, total_frames-1, sample_num, dtype=int)
    
        return indexes, total_frames

    
    # need shuffle and duplicate the index
    def build_qc_index(self, indexes):      # and the index is required
        # duplicate the index and shuffle
        indexes = indexes.repeat(2)
        indexes = np.random.shuffle(indexes)
        return indexes
    

    # the function to build the frames, indexes is required
    # assert the video not aligned 
    # TODO: if the frames are already built, we could just load the frames
    def build_frames(self, indexes):
        video_folder = self.data["video_folder"]
        save_folder = self.data["save_path"]

        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        # check if the frames are already built
        # TODO: add CHECK for the avaliable frames
        if os.path.exists(os.path.join(save_folder, "frames")):
            print("The frames are already built")
            return

        # TODO: assert the video frames number not aligned, a time cost function 
        frames_num_list = []
        for view_folder in view_folders:
            video_path = os.path.join(video_folder, view_folder, '0.mp4')
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, view_folder, '0.avi')
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, view_folder, '0.npy')
            
            frames = frame_sampler(video_path, indexes)
            npy_folder = os.path.join(save_folder, "frames", view_folder)
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder)
            npy_file = os.path.join(npy_folder, 'frames.npy')
            np.save(npy_file, frames)
            
        return


    # NOTE: receive the index and get the frames stored into the npy file
    def build_up_frames_npy(self, ):
        # could get/save the index in video folder
        video_folder = self.data["video_folder"]        
        # the video folder format is:
        # video_folder
        #   - view1
        #       - 0.mp4
        #   - view2
        #       - 0.mp4
        #   - ...

        save_folder = self.data["save_path"]

        view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
        view_folders.sort()

        # get the index file, read the index file from save folder
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
# TODO: COULD WE SPEED THIS UP?
def frame_sampler(video_path, frame_index):
    # if the video is npy file
    if video_path.endswith('.npy'):
        frames = np.load(video_path)
        frames = frames[frame_index]
        return frames

    else: 
        cap = cv2.VideoCapture(video_path)
        frames = []
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

    return frames
