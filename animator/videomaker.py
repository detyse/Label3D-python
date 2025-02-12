# class and function to make frames
# choose using matplotlib to plot the scatter (could change the style easily)
# WE DO NOT HAVE THE CAM_NAMES HERE...? 

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

# and do not pick views here
class VideoMaker:   
    def __init__(self, 
                camParams, 
                video_folder, 
                skeleton, 
                joints3d, 
                save_path, 
                frame_indexes,          # make sure the order is correct
                cam_names,
                video_format,
                joints_selected,        # put this here for scatter plot
                fps,
            ):
        
        """ Using the same params with label3d class
        camParams: a packed data just load from a mat file
        video_folder: 
        """
        # some properties
        self.cam_names = cam_names
        self.save_path = save_path
        self.video_format = video_format
        self.fps = fps

        # joints3d = kwargs['joints3d']       # like the label3d classs
        
        cam_params = self._unpack_camParams(camParams)
        frames, cam_names = self.get_frames(video_folder, cam_names,frame_indexes)     # error here
        the_skeleton = self.read_json_skeleton(skeleton)

        # pick the selected joints
        # picked_joints3d = self.pick_joints(joints3d, joints_selected)

        # NOTE: the views is already picked in the params...
        self._init_properties(
            frames=frames,             # a list of frames data align with the cam_names order
            joints3d=joints3d,
            skeleton=the_skeleton,
            joints_selected=joints_selected,    # this for the scatter plot
            cam_params=cam_params,
            save_folder=save_path,          # do not need to process
            save_format=video_format,       # do not need to process
            fps=fps,                # do not need to process
            cam_names=cam_names           # do not need to process, for the output name
        )

    def _init_properties(self, 
                 frames, 
                 joints3d, 
                 skeleton, 
                 joints_selected, 
                 cam_params, 
                 save_folder,
                 save_format,
                 fps,
                 cam_names):
        """
        frames: a list of frames, (or path?) frames data here
        skeleton: the skeleton dict, load from the json file
        joints3d: the 3d joints data, load from the json file
        joints_selected: the joints selected to make the video, could be a list of joints or a list of joint names
        cam_params: the camera parameters, also loaded data
        save_folder: the folder to save the video, actually the frames folder
        save_format: the format of the video "mp4" or "avi", default is mp4
        cam_names: a list of the camera names
        """
        self.frames = np.array(frames)
        self.joints3d = joints3d
        self.skeleton = skeleton
        self.joints_selected = joints_selected
        self.cam_params = cam_params
        self.save_folder = save_folder
        self.save_format = save_format
        self.fps = fps
        self.cam_names = cam_names      # here is a dict
    
        # print(f"frames: {type(frames)}")
        # print(f"frames: {len(frames)}")
        # print(f"frames: {frames[0].shape}")

        # this could used to decide which views used to make the video
        # F**K validation checks
        # if cam_names is not None:
            # assert len(frames) == len(cam_params) == len(cam_names), "The number of frames, camera parameters and camera names must be the same"
        # else:
            # assert len(frames) == len(cam_params), "The number of frames and camera parameters must be the same"
        # FIXME: 1 or 2 here... we will see
        # assert joints3d.shape[1] == len(joints_selected), "The number of joints and joints selected must be the same"

    def _unpack_camParams(self, camParams):
        r = []
        t = []
        K = []
        RDist = []
        TDist = []

        for cam in camParams:
            r.append(cam["r"][0][0].T)
            t.append(cam["t"][0][0])
            K.append(cam["K"][0][0].T)
            RDist.append(cam["RDistort"][0][0])
            TDist.append(cam["TDistort"][0][0])

        cam_params = {}
        cam_params["r"] = np.array(r)
        cam_params["t"] = np.array(t)
        cam_params["K"] = np.array(K)
        cam_params["RDist"] = np.array(RDist)
        cam_params["TDist"] = np.array(TDist)

        return cam_params
        
    # also need some functions to do the loading
    def read_json_skeleton(self, skeleton_path):
        with open(skeleton_path, 'r') as f:
            data = json.load(f)
        return data

    def get_frames(self, video_folder, cam_names, frame_indexes):
        frames = []

        if cam_names is not None:
            for cam_name in cam_names:
                # the video path
                for _name in ["0.mp4", "0.npy", "frames.npy"]:
                    video_path = os.path.join(video_folder, cam_name, _name)
                    if os.path.exists(video_path):
                        if _name == "0.mp4":
                            frames_data = self.frames_from_video(video_path)
                            frames.append(frames_data[frame_indexes, ...])
                        elif _name == "0.npy":
                            frames_data = np.load(video_path)
                            frames.append(frames_data[frame_indexes, ...])
                        elif _name == "frames.npy":
                            frames_data = np.load(video_path)
                            frames.append(frames_data[frame_indexes, ...])
        
        # we get the cam_names here
        else:   # get the views from the save folder
            view_folders = [f for f in os.listdir(video_folder) if os.path.isdir(os.path.join(video_folder, f))]
            view_folders.sort()
            print(f"view_folders: {view_folders}")
            cam_names = view_folders
            views = [os.path.join(video_folder, f) for f in view_folders]
            
            for view in views:
                for _name in ["0.mp4", "0.npy", "frames.npy"]:
                    video_path = os.path.join(view, _name)
                    if os.path.exists(video_path):
                        if _name == "0.mp4":
                            frames_data = self.frames_from_video(video_path)
                            frames.append(frames_data[frame_indexes, ...])
                        elif _name == "0.npy" or _name == "frames.npy":
                            frames_data = np.load(video_path)
                            frames.append(frames_data[frame_indexes, ...])
        return frames, cam_names

    def frames_from_video(self, video_path):
        # read the video with opencv
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        frames = np.array(frames)
        return frames

    # def pick_joints(self, joints3d, joints_selected):
    #     return joints3d[:, joints_selected, :]      # FIXME if the shape is [n_frames, n_joints, 3], we will see.

    def make_video(self, ):
        num_views, frame_num, height, width, channel_num = self.frames.shape        # the frames is a list of frames
        
        frame_size = (height, width)

        # get the projected 2d points at once
        points2d = project_3d_to_2d(self.joints3d, self.cam_params)

        for v in range(num_views):
            if self.save_format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.save_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                raise ValueError(f"Unsupported save format: {self.save_format}")

            # FIXME one of the params of the video writer is None
            # print(f"frame_size: {frame_size}")
            # print(f"fps: {self.fps}")
            # print(f"fourcc: {fourcc}")
            # print(f"save_folder: {self.save_folder}")
            # print(f"save_format: {self.save_format}")
            # print(f"cam_names: {self.cam_names}")
            # print(f"v: {v}")

            video = cv2.VideoWriter(os.path.join(self.save_folder, f"{self.cam_names[v]}_with_joints.{self.save_format}"), fourcc, self.fps, (frame_size[1], frame_size[0]))

            # print(f" ===== points2d shape: {points2d.shape}")
            # the shape of the points2d is (n_views, n_frames, n_joints, 2)

            for f in range(frame_num):
                the_frame = scatter_frame(self.frames[v, f, ...], points2d[v, f, ...], self.skeleton, self.joints_selected)

                the_frame = cv2.cvtColor(the_frame, cv2.COLOR_RGB2BGR)

                if the_frame.shape[:2][::-1] != frame_size:
                    the_frame = cv2.resize(the_frame, (frame_size[1], frame_size[0]))

                video.write(the_frame)

            video.release()
        return 

#### utils functions ####
def scatter_frame(frame, points2d, skeleton, joints_selected, **kwargs):
    # the **kwargs will be used to set the style of the scatter plot
    # frame = cv2.imread(frame)         # imread except a path of the image

    height, width = frame.shape[:2]
    dpi = 1000
    figsize = (width / dpi, height / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(frame, interpolation='bilinear')
    ax.set_axis_off()

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # joints -> index map, points2d maybe not full with all joints
    joints_idx = {}
    sorted_joints = sorted(joints_selected)
    for i, joint in enumerate(sorted_joints):
        joints_idx[joint] = i

    # NOTE: here set the scatter style
    for joint in joints_selected:
        ax.scatter(points2d[joint, 0], points2d[joint, 1], color=tuple(skeleton['color'][joint+1]), s=0.5)

    for i, bones in enumerate(skeleton["joints_idx"]):
        if bones[0] in joints_selected and bones[1] in joints_selected:
            x_value = [points2d[bones[0]-1, 0], points2d[bones[1]-1, 0]]
            y_value = [points2d[bones[0]-1, 1], points2d[bones[1]-1, 1]]
            # NOTE: here set the line style
            ax.plot(x_value, y_value, color=tuple(skeleton["color"][i]), linewidth=0.3)

    # convert the matplotlib figure to a numpy array
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    cv2.imshow("frame", img)
    cv2.waitKey(1)
    # close the figure
    plt.close(fig)
    return img


### projection functions ###
# not pick views here
def frame_adjust(frame, **kwargs):
    return frame

# FIXME: hope this works well
def project_3d_to_2d(joints_3d, cam_params):
    # input shape: (n_frames, n_joints, 3)
    # output shape: (n_frames, n_joints, num_views, 2)
    # and here the "r", "t", "K", "RDist", "TDist" are all list of arrays

    r = cam_params["r"]
    t = cam_params["t"]
    K = cam_params["K"]
    RDist = cam_params["RDist"]
    TDist = cam_params["TDist"]

    n_frames, n_joints, _ = joints_3d.shape
    n_views = len(r)

    print(f"n_views in function 'project_3d_to_2d' : {n_views}")

    reprojected_points = np.full((n_views, n_frames, n_joints, 2), np.nan)

    # handle the nan data here, 
    for k, frame_points in enumerate(joints_3d):
        for j in range(n_joints):
            if np.isnan(frame_points[j]).any():
                reprojected_points[:, k, j] = np.nan
                continue
            point3d = frame_points[j]
            # the reprojectToViews function handle all the views at once...
            reprojected_points[:, k, j] = reprojectToViews(point3d, r, t, K, RDist, TDist, n_views)

    return reprojected_points


# only for one point projection
# handle the nan data here
# FIXME: the view_num is 5?
def reprojectToViews(points3d, r, t, K, RDist, TDist, view_num):
    # points3d shape: (n_frames, n_joints, 3)
    rvec = []
    for i in range(view_num):
        rvec.append(cv2.Rodrigues(r[i])[0])

    reprojected_points = []
    for i in range(view_num):
        the_K = K[i]
        dist_coef = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], the_K, dist_coef)
        reprojected_points.append(reprojected_point)
    
    reprojected_points = np.array(reprojected_points).squeeze()
    # get the shape reprojected data
    # print("the shape of reprojected points: ", reprojected_points.shape)
    return reprojected_points
