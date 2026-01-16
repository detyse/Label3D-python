# class and function to make frames
# choose using matplotlib to plot the scatter (could change the style easily)
# WE DO NOT HAVE THE CAM_NAMES HERE...? 

import os
import cv2
import numpy as np
import json
import time

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
                joints_selected,
                fps,
                max_cache_frames=1000,  # 新增参数：最大缓存帧数，用户可根据内存大小调整
            ):
        
        """ Using the same params with label3d class
        camParams: a packed data just load from a mat file
        video_folder: 
        max_cache_frames: 最大缓存帧数，默认1000帧。用户可根据内存大小调整：
                         - 内存充足（16GB+）: 2000-5000
                         - 中等内存（8-16GB）: 1000-2000  
                         - 内存有限（<8GB）: 500-1000
        """
        # some properties
        self.cam_names = cam_names
        self.save_path = save_path
        self.video_format = video_format
        self.fps = fps
        self.video_folder = video_folder
        self.frame_indexes = frame_indexes
        self.joints_selected = joints_selected
        self.max_cache_frames = max_cache_frames

        # joints3d = kwargs['joints3d']       # like the label3d classs
        
        self.cam_params = self._unpack_camParams(camParams)
        self.skeleton = self.read_json_skeleton(skeleton)       # this should be the generated skeleton
        self.joints3d = joints3d

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

    def get_video_paths(self):
        video_paths = []
        if self.cam_names:
            for cam_name in self.cam_names:
                # the video path
                for _name in ["0.mp4", "0.avi"]: # Add other potential video formats
                    video_path = os.path.join(self.video_folder, cam_name, _name)
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
                        break # Found video for this camera view
        
        else: # get the views from the save folder
            view_folders = [f for f in os.listdir(self.video_folder) if os.path.isdir(os.path.join(self.video_folder, f))]
            view_folders.sort()
            self.cam_names = view_folders
            views = [os.path.join(self.video_folder, f) for f in view_folders]
            
            for view in views:
                for _name in ["0.mp4", "0.avi"]:
                    video_path = os.path.join(view, _name)
                    if os.path.exists(video_path):
                        video_paths.append(video_path)
                        break
        return video_paths


    def make_video(self, ):
        start_time = time.time()
        print("[Info] Starting optimized video generation...")

        # 1. 投影3D点到2D
        points2d = project_3d_to_2d(self.joints3d, self.cam_params)
        video_paths = self.get_video_paths()

        if not video_paths:
            print("[Error] No video files found.")
            return

        total_videos = len(video_paths)
        unique_frame_indices = sorted(list(set(self.frame_indexes)))
        total_unique_frames = len(unique_frame_indices)
        
        # 计算需要多少批次来处理所有帧
        num_batches = (total_unique_frames + self.max_cache_frames - 1) // self.max_cache_frames
        
        print(f"[Info] Processing {total_unique_frames} unique frames in {num_batches} batch(es)")
        print(f"[Info] Max cache frames per batch: {self.max_cache_frames}")
        
        for v_idx, video_path in enumerate(video_paths):
            view_start_time = time.time()
            print(f"[Info] Processing video {v_idx + 1}/{total_videos}: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[Error] Could not open video {video_path}")
                continue

            # 获取视频属性
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)
            
            if self.video_format == "mp4":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            elif self.video_format == "avi":
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                raise ValueError(f"Unsupported save format: {self.video_format}")

            out_path = os.path.join(self.save_path, f"{self.cam_names[v_idx]}_with_joints.{self.video_format}")
            video_writer = cv2.VideoWriter(out_path, fourcc, self.fps, frame_size)

            print(f"[Info] Writing frames to: {out_path}")
            
            # 不需要预先分组，直接在处理过程中按需处理
            
            # 创建帧到序列位置的映射
            frame_to_sequence_map = {frame_idx: i for i, frame_idx in enumerate(self.frame_indexes)}
            
            # 存储等待写入的帧数据（按序列顺序）
            pending_frames = {}
            next_frame_to_write = 0  # 下一个应该写入的序列位置
            
            # 分批处理：缓存->处理->写入->清理内存
            for batch_idx in range(num_batches):
                batch_start = batch_idx * self.max_cache_frames
                batch_end = min((batch_idx + 1) * self.max_cache_frames, total_unique_frames)
                batch_frame_indices = unique_frame_indices[batch_start:batch_end]
                
                print(f"[Info] Batch {batch_idx + 1}/{num_batches}: Caching frames {batch_start + 1}-{batch_end}")
                
                # 1. 缓存当前批次的帧
                batch_cache = self._cache_frames_batch(cap, batch_frame_indices)
                
                # 2. 处理当前批次中的帧，按照self.frame_indexes的顺序准备数据
                for frame_idx in batch_frame_indices:
                    if frame_idx in frame_to_sequence_map:
                        sequence_pos = frame_to_sequence_map[frame_idx]
                        frame = batch_cache[frame_idx]
                        the_frame = scatter_frame(frame, points2d[v_idx, sequence_pos, ...], self.skeleton, joints_selected=self.joints_selected)
                        
                        if the_frame.shape[1] != width or the_frame.shape[0] != height:
                            the_frame = cv2.resize(the_frame, frame_size)
                        
                        pending_frames[sequence_pos] = the_frame
                
                # 3. 按顺序写入所有可以写入的帧
                while next_frame_to_write in pending_frames:
                    video_writer.write(pending_frames[next_frame_to_write])
                    del pending_frames[next_frame_to_write]  # 立即释放内存
                    next_frame_to_write += 1
                    
                    if next_frame_to_write % 50 == 0:
                        print(f"[Progress] Written {next_frame_to_write}/{len(self.frame_indexes)} frames for video {v_idx+1}")
                
                # 4. 清理当前批次的缓存
                del batch_cache
                print(f"[Info] Batch {batch_idx + 1} processed, memory cleared. Pending: {len(pending_frames)} frames")
            
            # 5. 写入剩余的帧（如果有的话）
            while next_frame_to_write < len(self.frame_indexes) and next_frame_to_write in pending_frames:
                video_writer.write(pending_frames[next_frame_to_write])
                del pending_frames[next_frame_to_write]
                next_frame_to_write += 1
            
            cap.release()
            
            video_writer.release()
            view_end_time = time.time()
            print(f"[Success] Completed video {v_idx + 1}/{total_videos} in {view_end_time - view_start_time:.2f} seconds.")

        end_time = time.time()
        print(f"[Success] All videos generated successfully in {end_time - start_time:.2f} seconds.")
        return

    def _cache_frames_batch(self, cap, frame_indices_to_cache):
        """缓存指定的帧索引"""
        cached_frames = {}
        max_frame_to_cache = max(frame_indices_to_cache)
        
        frame_count = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到开头
        
        while cap.isOpened() and frame_count <= max_frame_to_cache:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count in frame_indices_to_cache:
                cached_frames[frame_count] = frame.copy()
                
            frame_count += 1
            
            # 如果已经缓存了所有需要的帧，提前退出
            if len(cached_frames) == len(frame_indices_to_cache):
                break
        
        return cached_frames

#### utils functions ####
def scatter_frame(frame, points2d, skeleton, joints_selected=None, **kwargs):
    """
    在单帧图像上用OpenCV绘制骨骼点和连接，参考animator.py中的方法来美化绘制效果。
    使用skeleton中保存的color信息为每个关节点和连接线上色。
    """
    img = frame.copy()
    
    joint_names = skeleton['joint_names']
    joints_idx = skeleton['joints_idx']
    colors = skeleton['color']
    num_joints = len(joint_names)
    
    # 1. 先绘制骨骼连接线（在关节点下方）
    for i, bones in enumerate(joints_idx):
        start_joint_idx, end_joint_idx = bones[0] - 1, bones[1] - 1  # 转换为0-based索引
        
        # 检查关节选择
        if joints_selected is not None and (start_joint_idx not in joints_selected or end_joint_idx not in joints_selected):
            continue

        # 检查关节点是否有效（包括NaN和inf检查）
        if (start_joint_idx < num_joints and end_joint_idx < num_joints and
            not np.isnan(points2d[start_joint_idx, 0]) and not np.isnan(points2d[end_joint_idx, 0]) and
            not np.isinf(points2d[start_joint_idx, 0]) and not np.isinf(points2d[end_joint_idx, 0]) and
            not np.isnan(points2d[start_joint_idx, 1]) and not np.isnan(points2d[end_joint_idx, 1]) and
            not np.isinf(points2d[start_joint_idx, 1]) and not np.isinf(points2d[end_joint_idx, 1])):
            
            # 安全地转换为整数坐标
            try:
                start_x = int(np.round(float(points2d[start_joint_idx, 0])))
                start_y = int(np.round(float(points2d[start_joint_idx, 1])))
                end_x = int(np.round(float(points2d[end_joint_idx, 0])))
                end_y = int(np.round(float(points2d[end_joint_idx, 1])))
                start_point = (start_x, start_y)
                end_point = (end_x, end_y)
            except (ValueError, OverflowError):
                continue  # 跳过无法转换的点
            
            # 使用skeleton中保存的连接线颜色（RGBA -> BGR格式）
            if i < len(colors):
                color_rgba = colors[i]
                # 转换RGBA到BGR，OpenCV使用BGR格式
                color_bgr = (
                    int(color_rgba[2] * 255),  # B
                    int(color_rgba[1] * 255),  # G  
                    int(color_rgba[0] * 255)   # R
                )
                # 使用alpha值来调整线条粗细（可选）
                alpha = color_rgba[3] if len(color_rgba) > 3 else 1.0
                thickness = max(1, int(3 * alpha))  # 根据alpha调整线条粗细
            else:
                color_bgr = (255, 255, 255)  # 默认白色
                thickness = 2
            
            cv2.line(img, start_point, end_point, color_bgr, thickness=thickness)

    # 2. 再绘制关节点（在连接线上方）
    for joint_idx in range(num_joints):
        # 检查关节选择
        if joints_selected is not None and joint_idx not in joints_selected:
            continue
        
        # 检查点是否有效（包括NaN和inf检查）
        if (not np.isnan(points2d[joint_idx, 0]) and not np.isnan(points2d[joint_idx, 1]) and
            not np.isinf(points2d[joint_idx, 0]) and not np.isinf(points2d[joint_idx, 1])):
            
            # 安全地转换为整数坐标
            try:
                point_x = int(np.round(float(points2d[joint_idx, 0])))
                point_y = int(np.round(float(points2d[joint_idx, 1])))
                point = (point_x, point_y)
            except (ValueError, OverflowError):
                continue  # 跳过无法转换的点
            
            # 使用skeleton中保存的关节点颜色
            if joint_idx < len(colors):
                color_rgba = colors[joint_idx]
                # 转换RGBA到BGR
                color_bgr = (
                    int(color_rgba[2] * 255),  # B
                    int(color_rgba[1] * 255),  # G
                    int(color_rgba[0] * 255)   # R
                )
                # 使用alpha值来调整点的大小（可选）
                alpha = color_rgba[3] if len(color_rgba) > 3 else 1.0
                radius = max(2, int(6 * alpha))  # 根据alpha调整点的大小
            else:
                color_bgr = (255, 255, 255)  # 默认白色
                radius = 4
            
            # 绘制关节点：先画一个稍大的黑色圆作为边框，再画彩色圆
            cv2.circle(img, point, radius + 1, (0, 0, 0), thickness=-1)  # 黑色边框
            cv2.circle(img, point, radius, color_bgr, thickness=-1)       # 彩色填充
            
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
    return reprojected_points

