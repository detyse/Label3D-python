# helper functions for label 3d
# load skeleton and camera parameters into json format

# TODO: add the 

import sys
import json
import csv
import cv2
import os
import shutil
import pandas as pd
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

    # also check avaliable here
    def load_index_file(self, index_file, total_frames):    
        # index_file: path of the file
        if index_file.endswith('.npy'):
            indexes = np.load(index_file)
        # else if the index file is a csv/excel file, load the index file
        # TODO test the index loading function
        elif index_file.endswith('.csv'):
            df = pd.read_csv(index_file, header=None)
            indexes = df.iloc[:, 0].values      # the first column is the index
            indexes = np.array(indexes, dtype=int)
        elif index_file.endswith('.xlsx'):
            df = pd.read_excel(index_file, header=None)
            indexes = df.iloc[:, 0].values
            indexes = np.array(indexes, dtype=int)
        elif index_file.endswith('.txt'):
            # Read the file content
            with open(index_file, 'r') as f:
                content = f.read().strip()
            # Check if comma-separated
            if ',' in content:
                numbers = content.split(',')
            else:
                numbers = content.split()
            # Convert to integers
            indexes = np.array([int(x.strip()) for x in numbers if x.strip()], dtype=int)
        else:
            raise ValueError("The index file is not supported")
        
        print(f"the indexes: {indexes}")
        # print the type of the indexes
        print(f"the type of the indexes: {type(indexes)}")
        # print the type of the indexes[0]
        print(f"the type of the indexes[0]: {type(indexes[0])}")
        # all the elements should be int
        assert all(isinstance(index, int) or isinstance(index, np.int32) or isinstance(index, np.int64) for index in indexes), "The index file should be a list of integers"

        # assert if there is duplicate and out of range
        assert len(indexes) == len(set(indexes)), "There is duplicate in the index file"
        assert indexes.max() < total_frames and indexes.min() >= 0, "The frame index is out of the total frames"

        print(f"the indexes: {indexes}")
        return indexes

    # TODO: divide new loading or old loading here, if there is joints3d, then it is old loading
    # and old loading could  laskdjf lljdo not consider the QC mode
    def get_params_new(self, ):
        params = {}

        save_folder = self.data["save_path"]
        params["save_path"] = save_folder

        if os.path.exists(os.path.join(save_folder, "joints3d.npy")):
            load_old = True
        else:
            load_old = False

        if load_old:        # load old, we do not consider the QC mode
            if os.path.exists(os.path.join(save_folder, "frames")):
                params["video_folder"] = os.path.join(save_folder, "frames")
            else:
                raise ValueError("There should be built frames in the save folder")
            
            if os.path.exists(os.path.join(save_folder, "cam_params.mat")):
                cam_params_path = os.path.join(save_folder, "cam_params.mat")
            else:
                cam_params_path = self.data["cam_params"]
            
            if os.path.exists(os.path.join(save_folder, "skeleton.json")):
                params["skeleton_path"] = os.path.join(save_folder, "skeleton.json")
            else:
                params["skeleton_path"] = self.data["skeleton_path"]

            # if there is the index, load the index, else build the index
            if os.path.exists(os.path.join(save_folder, "indexes.npy")):
                print(f"the index file exists")
                params["frame_indexes"] = np.load(os.path.join(save_folder, "indexes.npy"))
            else:
                print(f"the index file does not exist")
                params["frame_indexes"], _ = self.get_index()      # FIXME this will be fine
                print(f"the frame_indexes: {params['frame_indexes']}, type: {type(params['frame_indexes'])}")

            params["total_frame_num"] = params["frame_indexes"].shape[0]
            params["frame_num2label"] = None
            params["quality_control_on"] = False
            params["cam_params"] = self.unpack_cam_params(cam_params_path)
            return params

        else:
            cam_params_path = self.data["cam_params"]
            skeleton_path = self.data["skeleton_path"]
            video_folder = self.data["video_folder"]
            index_file = self.data["given_frame_indexes"]
            frame_num2label = self.data["frame_num2label"]
            quality_control_on = self.data["quality_control_on"]

            # if the index file is not given, then we need to build the index
            if index_file == '':
                if frame_num2label == None or frame_num2label == '':
                    params["frame_num2label"] = None
            
                else:
                    params["frame_num2label"] = frame_num2label

                # than make the index list
                indexes, total_frames = self.get_index(frame_num2label, None, )

                if quality_control_on:
                    indexes = self.build_qc_index(indexes)

                params["frame_indexes"] = indexes

                # than build the frames
                self.build_frames(indexes)

            else:
                total_frames = self.get_frame_number(video_folder)
                indexes = self.load_index_file(index_file, total_frames)

                if quality_control_on:
                    indexes = self.build_qc_index(indexes)

                params["frame_indexes"] = indexes

                # build the frames
                self.build_frames(indexes)

            params["total_frame_num"] = total_frames        # also not used in label3d
            params["quality_control_on"] = quality_control_on
            params["video_folder"] = video_folder
            params["cam_params"] = self.unpack_cam_params(cam_params_path)
            params["skeleton_path"] = skeleton_path
            params["frame_num2label"] = None 

            return params

    # FIXME: consider the reload, if the indexes and frames are already stored, we could just load them directly
    # so the priority is saved-index-file > given_frame_indexes > frame_num2label
    # def get_params(self, ):
    #     # show self.data here
    #     print(f"show self.data: {self.data}")       # wondering if the data is not given, what will return?

    #     params = {}
    #     cam_params_path = self.data["cam_params"]
    #     params['cam_params'] = self.unpack_cam_params(cam_params_path)
    #     params['skeleton_path'] = self.data["skeleton_path"]
    #     params['save_path'] = self.data["save_path"]

    #     # the video folder could be None if the output folder exists
    #     params['video_folder'] = self.data.get("video_folder", None)

    #     # create the index and the frames here
    #     save_folder = self.data["save_path"]

    #     given_index_file = self.data["given_frame_indexes"]
    #     # check the index file

    #     if given_index_file is not None:
    #         params['frame_num2label'] = None
    #     else:
    #         params['frame_num2label'] = self.data["frame_num2label"]
    #         sample_num = self.data["frame_num2label"]
        
    #     params['quality_control_on'] = self.data["quality_control_on"]          # NOTE: may have bug when the video folder is not given

    #     # not the first loading, use the indexes file to check that? 
    #     the_index_file = os.path.join(save_folder, "indexes.npy")
    #     # NOTE the saved frame indexes are have higher priority
    #     if os.path.exists(the_index_file):
    #         # load the index file
    #         indexes = np.load(the_index_file)
    #         total_frames = indexes.shape[0]
        
    #     else:
    #         # get index, no duplicate
    #         if given_index_file is not None:
    #             # if the index file is npy file, load the npy file
    #             if given_index_file.endswith('.npy'):
    #                 indexes = np.load(given_index_file)
    #             # else if the index file is a csv/excel file, load the index file
    #             # TODO test the index loading function
    #             elif given_index_file.endswith('.csv'):
    #                 df = pd.read_csv(given_index_file, header=None)
    #                 indexes = df.iloc[:, 0].values      # the first column is the index
    #             elif given_index_file.endswith('.xlsx'):
    #                 df = pd.read_excel(given_index_file, header=None)
    #                 indexes = df.iloc[:, 0].values
    #             elif given_index_file.endswith('.txt'):
    #                 text = np.loadtxt(given_index_file)
    #                 # split by space
    #                 if "," in text:
    #                     indexes = text.split(",").strip()
    #                 else:
    #                     indexes = text.split(" ").strip()      
    #                 indexes = np.array(indexes, dtype=int)
    #             else:
    #                 raise ValueError("The index file is not supported")

    #         else: 
    #             # check the index file 
    #             indexes, total_frames = self.get_index(sample_num)

    #         # assert index
    #         assert indexes.max() < total_frames and indexes.min() >= 0, "The frame index is out of the total frames"

    #     if params.get('quality_control_on'):
    #         final_indexes = self.build_qc_index(indexes)
    #         # save the index file
    #     else:
    #         final_indexes = indexes

    #     if os.path.exists(os.path.join(save_folder, "frames")):
    #         print("The frames are already built")
    #         # build the frames
    #     else:    
    #         self.build_frames(final_indexes)

    #     # save the index file should be the last step to load the params
    #     np.save(the_index_file, final_indexes)
        
    #     print(f"the frame indexes: {final_indexes}")
    #     params['frame_indexes'] = final_indexes
    #     if 'total_frames' not in locals():          # at what condition the total frame is not defined? when the index is loaded...
    #         _, total_frames = self.get_index(sample_num)
    #     params['total_frame_num'] = total_frames

    #     return params

    def unpack_cam_params(self, mat_path):
        cam_params = loadmat(mat_path)['params']
        
        load_camParams = []
        for i in range(len(cam_params)):
            load_camParams.append(cam_params[i][0])
        
        return load_camParams
    
    def get_frame_number(self, video_folder):
        if video_folder is None or video_folder == "":
            frames_path = self.data["save_path"]
            frames_path = os.path.join(frames_path, "frames")

        # NOTE: so the video folder should only have view folders... means that if you do the prediction, you could not use the folder to label...
        # remove the logger folder in prediction...
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

        return total_frames

    # function to build up the index, when the index is not given
    # and the index should uniform sampled from the total frames
    # before quality control, 
    def get_index(self, sample_num=None, total_frames=None):
        video_folder = self.data["video_folder"]

        # if video folder is not given, and the index file is not given
        # 这是一个补丁 only used in spacial case: which the video folder is ignored
        if video_folder is None or video_folder == "":
            frames_path = self.data["save_path"]
            frames_path = os.path.join(frames_path, "frames")

            try:
                view_folders = [f for f in os.listdir(frames_path) if os.path.isdir(os.path.join(frames_path, f))]
                view_folders.sort()
            except:
                raise ValueError("The frames are not built, please build the frames first")

            # get the total frames
            total_frames = np.load(os.path.join(frames_path, view_folders[0], 'frames.npy')).shape[0]
            return np.arange(total_frames), total_frames

        if total_frames is None:
            total_frames = self.get_frame_number(video_folder)

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
        # TODO: add the storage of other params
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

    # this function is not used...
    def save_other_params(self, ):
        save_folder = self.data["save_path"]

        # just rename the params file and the skeleton file
        params_file = self.data["cam_params"]
        skeleton_file = self.data["skeleton_path"]

        # rename the files.. TODO: WE'D BETTER MAKE A NOTE ON THE FILE NAME RESAVED, and better add the exp name as the prefix
        params_save = os.path.join(save_folder, "cam_params.mat")
        skeleton_save = os.path.join(save_folder, "skeleton.json")

        # if the file is exist, skip
        if os.path.exists(params_save):
            print(f"The params file {params_save} already exists, skip")
        else:
            shutil.copy(params_file, params_save)

        if os.path.exists(skeleton_save):
            print(f"The skeleton file {skeleton_save} already exists, skip")
        else:
            shutil.copy(skeleton_file, skeleton_save)

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
            # Convert index to integer and ensure it's a valid frame number
            frame_pos = int(index)
            if frame_pos < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret = cap.grab()
            if not ret:
                break
            ret, frame = cap.retrieve()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        cap.release()
        frames = np.array(frames)
        if len(frames) == 0:
            raise ValueError("No valid frames were extracted from the video")
    return frames
