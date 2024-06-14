# functions for 2d 3d transformation
# some code from DANNCE-pytorch
# rewrite the code for better reuse
'''
some description for DANNCE format calibrated camera params:
 - r: 3 x 3
 - t: 3 x 1 or 1 x 3
 - K: 3 x 3            # transposed instrinsic matrix
 - RDist: 3 x 1
 - TDist: 2 x 1         # 切向畸变 几乎没有
'''

import cv2
import numpy as np
# import torch

# undistort and transfer the pixel image to camera coordinate
def pixel2Camera(point2d, K, RDist, TDist):       # (point2d, camera)
    # for a singla point, and all the params should be in the right shape
    # undistort the point 
    # point shape: [m, 1, 2]
    dcoef = RDist[:2].tolist() + TDist.tolist()
    if len(RDist) == 3:
        dcoef.append(RDist[2])
    else:
        dcoef.append(0)
    dcoef = np.array(dcoef, dtype=np.float32)

    point2d = np.array(point2d, dtype=np.float32).reshape(-1, 1, 2)

    undist = cv2.undistortPoints(
        point2d, K, dcoef, P=K          # add P is the original camera matrix
    )

    # squeeze the undistort point
    undist = undist.squeeze(1)

    # already transfer the pixel to camnera coordinate
    return undist


def triangulateMultiview(points2d, r, t, K, RDist, TDist):
    undist = []
    for i in range(len(points2d)):
        the_K = K[i]
        the_K[0, 1] = 0         # set the skew to 0
        point = points2d[i]

        # TODO: check skip the undistort
        dist_vec = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        undistort_point = cv2.undistortPoints(point, the_K, dist_vec)
        undist.append(undistort_point)
        # check the undist format 

        print(points2d)

    # triangulate the points
    undist = np.array(undist)
    undist = undist.squeeze()
    print(f"undist shape: {undist.shape}")
    print(undist)

    undist = np.array(points2d)

    A = []
    for i in range(len(r)):
        P = np.hstack((r[i], t[i].T))       # shape: (3, 4)  # there is a transpose, stack the P matrix here
        A.append(np.vstack((undist[i][0] * P[2, :] - P[0, :],
                            undist[i][1] * P[2, :] - P[1, :])))
        
    A = np.array(A)
    A = np.concatenate(A, axis=0)

    # SVD method to solve the hyper define problem
    _, _, V = np.linalg.svd(A)
    point_3d = V[-1, :]             # find the largest eign value vector
    point_3d = point_3d / point_3d[-1]
    return point_3d[:3]


# reprojection
def reprojectToViews(points3d, r, t, K, RDist, TDist, view_num):
    rvec = []
    for i in range(view_num):
        rvec.append(cv2.Rodrigues(r[i])[0])        # transfer the rotation vector into rotation matrix

    reprojected_points = []
    for i in range(view_num):
        the_K = K[i]
        the_K[0, 1] = 0         # set the skew to 0
        # dist_coef = np.array([RDist[i][0][0], RDist[i][0][1], TDist[i][0][0], TDist[i][0][1], RDist[i][0][2]])
        reprojected_point, _ = cv2.projectPoints(points3d, rvec[i], t[i], the_K, )      # dist_coef
        reprojected_points.append(reprojected_point)

    reprojected_points = np.array(reprojected_points).squeeze()
    return reprojected_points


### function from DANNCE-pytorch ###
# since define a class for camera, just use the class to reduce the params
# but we need matrix calculation
# stack the camera params to use
class Camera:
    def __init__(self, R, t, K, tdist, rdist, name=""):
        self.R = np.array(R, dtype=np.float32)
        assert self.R.shape == (3, 3)

        self.t = np.array(t, dtype=np.float32)
        assert self.t.shape == (3, 1)

        self.K = np.array(K, dtype=np.float32)
        assert self.K.shape == (3, 3)

        self.extrinsics = np.concatenate([self.R, self.t], axis=1)  # ?? 

        # distortion
        self.tdist = np.array(tdist, dtype=np.float32).flatten()
        # 
        self.rdist = np.array(rdist, dtype=np.float32).flatten()
        # 
        self.name = name


    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox        # why the right and lower are not used.

        cx, cy = self.K[2, 0], self.K[2, 1]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[2, 0], self.K[2, 1] = new_cx, new_cy


    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_height, new_width = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[2, 0], self.K[2, 1]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[2, 0], self.K[2, 1] = new_fx, new_fy, new_cx, new_cy


    def camera_matrix(self):
        return self.M 
    

    def get_extrinsics(self):
        return self.extrinsics


def distortPoints(points, cameras):
    
    pass 


def undistortPoints(points, cameras):
    # points
    
    pass


def trianglute_multi_instance(points, cameras):
    
    pass 
