''' Helper class and functions for loading KITTI objects

Author: Ari Jin
Date: Jan. 2022
'''
from __future__ import print_function
import inwater_util as utils

import os
import sys
import math
import numpy as np
import geopy as gp
import cv2
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
NUM_SAMPLE = 21
split = "training"


def read_info_file(filepath):
    ''' Read in an information file and parse into a dictionary.
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def rectify_motion(x, y, yaw, linearx, angularz, timestamp, base_timestamp):
    deltaT = base_timestamp - timestamp
    yaw = yaw + angularz * deltaT
    x = x + np.cos(yaw) * linearx * deltaT
    y = y + np.sin(yaw) * linearx * deltaT
    return x[0], y[0], yaw[0]


def generate_label_file(index, split="training"):
    '''This function is for generate the lables in the KITTI style.
    原始标签存储了目标类别，目标坐标(x,y,z)，目标尺寸(-x,+x,-y,+y,z),目标yaw，自身坐标(x,y,z),自身(r,p,y)
    啊'''
    # split = "training" # or "testing"
    root_dir = os.path.join(ROOT_DIR, 'inWater/object')
    split_dir = os.path.join(root_dir, split)

    info_dir = os.path.join(split_dir, 'info')
    src_label_dir = os.path.join(split_dir, 'label_2_raw')
    des_label_dir = os.path.join(split_dir, 'label_2')

    # read info
    data = read_info_file(os.path.join(info_dir, '%06d.txt' % (index)))
    camera_timestamp = data["camera_timestamp"]
    velodyne_timestamp = data["velodyne_timestamp"]
    livox_timestamp = data["livox_timestamp"]
    motion_info = data["motion"]

    mx = motion_info[0]
    my = motion_info[1]
    mz = motion_info[2]

    mroll = motion_info[3]
    mpitch = motion_info[4]
    myaw = motion_info[5]

    mlinearx = motion_info[6]
    mangularz = motion_info[7]
    mtimestamp = motion_info[8]

    # timestamp alignment: my motion timestamp tp camera timestamp
    print(mx, my, myaw)
    mx, my, myaw = rectify_motion(
        mx, my, myaw, mlinearx, mangularz, mtimestamp, camera_timestamp)
    print(mx, my, myaw)
    # read raw label
    lines = [line.rstrip() for line in open(
        os.path.join(src_label_dir, '%06d.txt' % (index)))]
    f = open(os.path.join(des_label_dir, f"{index:06d}.txt"), 'w')

    for line in lines:
        # read target label
        data_src = line.split(' ')
        data_src[1:] = [float(x) for x in data_src[1:]]

        type = data_src[0]
        # print(type)
        truncation = 0
        occlusion = 0
        alpha = 0

        tx = data_src[1]
        ty = data_src[2]
        tz = data_src[3]

        troll = 0
        tpitch = 0
        tyaw = data_src[4]

        tlinearx = data_src[5]
        tangularz = data_src[6]

        pos_sx = data_src[7]
        neg_sx = data_src[8]
        pos_sy = data_src[9]
        neg_sy = data_src[10]
        h = data_src[11]

        ttimestamp = data_src[12]

        # timestamp alignment: target timestamp tp camera timestamp
        if type == "ship":
            tx, ty, tyaw = rectify_motion(
                tx, ty, tyaw, tlinearx, tangularz, ttimestamp, camera_timestamp)

        '''get 3d bbox corner points in my base_link coordinate'''
        '''
        qs: (8,3) array of vertices for the 3d box in following order:
            5  -------- 4
           /|               /|
          6  -------- 7 .
           | |              | |
           . 1 -------- 0
           |/               |/
           2 -------- 3
        '''
        # 3d bounding box corners
        x_corners = [pos_sx, pos_sx, neg_sx,
                     neg_sx, pos_sx, pos_sx, neg_sx, neg_sx]
        y_corners = [pos_sy, neg_sy, neg_sy,
                     pos_sy, pos_sy, neg_sy, neg_sy, pos_sy]
        z_corners = [0, 0, 0, 0, h, h, h, h]
        corners_3d = np.vstack([x_corners, y_corners, z_corners])

        wt_T = gp.Tmatrix(troll, tpitch, tyaw, tx, ty, tz)
        wm_T = gp.Tmatrix(mroll, mpitch, myaw, mx, my, mz)
        mt_T = np.dot(gp.inverse_rigid_trans(wm_T), wt_T)
        # rotate and translate 3d bounding box
        corners_3d = np.dot(mt_T, np.vstack((corners_3d, np.ones((1, 8)))))
        corners_3d = corners_3d[0:3, :]

        # center_3d in base link coord, which is the 3d bbox's the center of bottom plane.
        corners_3d = np.transpose(corners_3d)
        # print(corners_3d)
        new_tx = (corners_3d[0, 0] + corners_3d[2, 0])/2
        new_ty = (corners_3d[0, 1] + corners_3d[2, 1])/2
        new_tz = (corners_3d[0, 2] + corners_3d[2, 2])/2

        new_th = h
        new_tw = pos_sy - neg_sy
        new_tl = pos_sx - neg_sx

        euler = gp.Rmatrix_to_euler(mt_T[0:3, 0:3])
        new_r = euler[0]
        new_p = euler[1]
        new_y = euler[2]

        # get calibration parameters
        calib_dir = os.path.join(split_dir, 'calib')
        calib_filename = os.path.join(calib_dir, '%06d.txt' % (index))
        calib = utils.Calibration(calib_filename)
        # only draw 3d bounding box for objs in front of the camera
        # print(corners_3d)
        if np.any(corners_3d[:, 0] < 0.1):
            xmin, ymin, xmax, ymax = 0, 0, 0, 0
        else:
            # project the 3d bounding box into the image plane
            # print(tx, ty, mx, my)
            # print("corners_3d", corners_3d)
            n = corners_3d.shape[0]
            pts_3d_extend = np.hstack((corners_3d, np.ones((n, 1))))
            corners_3d = np.dot(pts_3d_extend, np.transpose(
                calib.B2C))  # 8x4 4x3 = 8x3
            corners_2d = utils.project_to_image(corners_3d, calib.P)
            # print("corners_2d", corners_2d)
            xmin = corners_2d[0:4].min(axis=0)[0]
            ymin = corners_2d.min(axis=0)[1]
            xmax = corners_2d[0:4].max(axis=0)[0]
            ymax = corners_2d.max(axis=0)[1]
            area = (xmax - xmin) * (ymax - ymin)
            xmin = min(max(xmin, 0), calib.imgW)
            xmax = min(max(xmax, 0), calib.imgW)
            ymin = min(max(ymin, 0), calib.imgH)
            ymax = min(max(ymax, 0), calib.imgH)
            cutted_area = (xmax - xmin) * (ymax - ymin)
            truncation = int(cutted_area) / (area + 1e-6)
            truncation = float(truncation)
            if truncation < 0.3:
                xmin, ymin, xmax, ymax = 0, 0, 0, 0

        data = [None]*17
        # extract label, truncation, occlusion
        data[0] = str(type)  # 'Car', 'Pedestrian', ...
        data[1] = truncation  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        data[2] = int(occlusion)
        data[3] = alpha  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        data[4] = int(xmin)  # left
        data[5] = int(ymin)  # top
        data[6] = int(xmax)  # right
        data[7] = int(ymax)  # bottom

        # extract 3d bounding box information
        data[8] = new_th  # box height
        data[9] = new_tw  # box width
        data[10] = new_tl  # box length (in meters)
        data[11] = new_tx  # location (x,y,z) in camera coord.
        data[12] = new_ty
        data[13] = new_tz

        data[14] = new_r  # roll angle [-pi..pi]
        data[15] = new_p  # pitch angle [-pi..pi]
        data[16] = new_y  # yaw angle [-pi..pi]

        label = f"{data[0] } "
        for i in range(1, len(data)):
            label += f"{data[i] } "
        label += f"{data[-1]}\n"
        f.write(label)

    f.close()


def prepare_data():
    # generate_calib_parames_file(0, split)
    for data_idx in range(NUM_SAMPLE):
        generate_label_file(data_idx, split)


if __name__ == '__main__':
    prepare_data()
