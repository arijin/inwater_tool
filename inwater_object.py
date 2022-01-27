''' Helper class and functions for loading KITTI objects

Author: Ari Jin
Date: Jan. 2022
'''
from __future__ import print_function
# 必须在有import cv2的库之前import有mayavi的库
from viz_utils.lidarplot import initialize_figure, adjust_view, save_fig, draw_lidar_simple, draw_lidar, draw_gt_boxes3d, draw_heading_arrow
from viz_utils.plots3d import draw_projected_box3d
from viz_utils.plots import Annotator, colors, save_one_box
import inwater_util as utils

import os
import time
import sys
import numpy as np
import cv2
import yaml
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
NUM_SAMPLE = 50
split = "training"

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class DetectAnnotator():
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, data=None):
        self.names = [f'class{i}' for i in range(1000)]  # assign defaults
        if data:  # data.yaml path (optional)
            with open(data, errors='ignore') as f:
                self.names = yaml.safe_load(f)['names']  # class names

    def draw_image(self, im0, type_name, xyxy):
        label = f'{type_name} 1.0'
        if type_name in self.names:
            c = self.names.index(type_name)
        else:
            c = 0

        annotator = Annotator(im0, line_width=1, example=str(self.names))
        annotator.box_label(xyxy, label, color=colors(c, True))
        # Stream results
        im0 = annotator.result()
        return im0


class inwater_object(object):
    '''Load and parse object data into a usable format.'''

    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = NUM_SAMPLE
        elif split == 'testing':
            self.num_samples = 0
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.livox_dir = os.path.join(self.split_dir, 'livox')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx < self.num_samples)
        img_filename = os.path.join(self.image_dir, '%06d.png' % (idx))
        img = cv2.imread(img_filename)
        return img

    # (x, y, z)
    def get_velo(self, idx):
        assert(idx < self.num_samples)
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin' % (idx))
        scan = np.fromfile(lidar_filename, dtype=np.float)  # read bin file.
        scan = scan.reshape((-1, 3))
        return scan

    # (x, y, z)
    def get_livox(self, idx):
        assert(idx < self.num_samples)
        lidar_filename = os.path.join(self.livox_dir, '%06d.bin' % (idx))
        scan = np.fromfile(lidar_filename, dtype=np.float)  # read bin file.
        scan = scan.reshape((-1, 3))
        return scan

    def get_calibration(self, idx):
        assert(idx < self.num_samples)
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        # assert(idx<self.num_samples and self.split=='training')
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return utils.read_label(label_filename)


# def viz_kitti_video():
#     video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
#     dataset = kitti_object_video(\
#         os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
#         os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
#         video_path)
#     print(len(dataset))
#     for i in range(len(dataset)):
#         img = dataset.get_image(0)
#         pc = dataset.get_lidar(0)
#         Image.fromarray(img).show()
#         draw_lidar(pc)
#         raw_input()
#         pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
#         draw_lidar(pc)
#         raw_input()
#     return

def show_image_with_boxes(img, objects, calib, annotator, show3d=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox

    for obj in objects:
        if obj.type == 'DontCare':
            continue
        # obj.print_object()
        # draw 2d bbox
        xyxy = [int(obj.xmin), int(obj.ymin), int(obj.xmax), int(obj.ymax)]
        img1 = annotator.draw_image(img1, obj.type, xyxy)
        # cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
        #     (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(
            obj, calib.P, calib.B2C)
        # print(box3d_pts_2d)
        # print(box3d_pts_3d)
        img2 = draw_projected_box3d(img2, box3d_pts_2d)
    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    # Image.fromarray(img1).show()
    # if show3d:
    #     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # Image.fromarray(img2).show()
    return img1, img2


def show_lidar_with_boxes_cv(pc_lidar, objects, calib,
                             img_fov=False, img_width=None, img_height=None):
    pc_lidar[:, 0] = pc_lidar[:, 0]
    pc_lidar = pc_lidar[:, 0:2] * 10
    pc_lidar = np.int16(pc_lidar)
    print(pc_lidar)
    pc_lidar = np.clip(pc_lidar, -990, 990)
    pc_lidar = pc_lidar + 1000

    img = np.zeros((2000, 2000, 3), dtype=np.uint8)
    rows = pc_lidar[:, 1]
    cols = pc_lidar[:, 0]
    img[rows, cols] = (255, 255, 255)
    img[rows + 1, cols] = (255, 255, 255)
    img[rows - 1, cols] = (255, 255, 255)
    img[rows, cols-1] = (255, 255, 255)
    img[rows, cols+1] = (255, 255, 255)
    img[rows - 1, cols - 1] = (255, 255, 255)
    img[rows - 1, cols + 1] = (255, 255, 255)
    img[rows+1, cols-1] = (255, 255, 255)

    for obj in objects:
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(
            obj, calib.P, calib.B2V)
        for i in range(4):
            x = box3d_pts_3d[i, 0] * 10 + 1000
            y = box3d_pts_3d[i, 1] * 10 + 1000
            point = np.array([x, y])
            point = np.int16(point)
            if point[0] < 0 or point[0] > 1999 or point[1] < 0 or point[1] > 1999:
                continue
            pointd = (point[0], point[1])
            cv2.circle(img, pointd, 1, (0, 0, 255), 2)
    img = cv2.resize(img, (1000, 1000))
    cv2.imshow("aaa", img)
    cv2.waitKey(5000)

    return img


def show_lidar_with_boxes(pc_lidar, objects, calib,
                          img_fov=False, img_width=None, img_height=None):
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''

    print(('All point num: ', pc_lidar.shape[0]))
    # fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
    #                   fgcolor=None, engine=None, size=(1000, 500))
    # if img_fov:
    #     pc_lidar = get_lidar_in_image_fov(pc_lidar, calib, 0, 0,
    #                                      img_width, img_height)
    #     print(('FOV point num: ', pc_lidar.shape[0]))
    fig = initialize_figure()

    for obj in objects:
        if obj.type == 'DontCare':
            continue
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(
            obj, calib.P, calib.B2V)  # calib.B2V把点变换到velodyne坐标系下
        if obj.type == "ship":
            color = (0.9, 1, 0.5)
        elif obj.type == "docking":
            color = (0.8, 0.35, 0)
        else:
            color = (0.7, 0.3, 0)
        draw_gt_boxes3d([box3d_pts_3d], fig=fig, color=color, draw_text=False)
        if obj.type == "ship":
            draw_heading_arrow([box3d_pts_3d], fig=fig, color=(1, 0, 0))

    draw_lidar(pc_lidar, fig=fig)
    # livox远景：azimuth=180, elevation=72, focalpoint=[85, 0, 0], distance=220
    # livox近景：azimuth=180, elevation=72, focalpoint=[35, 0, 0], distance=220
    # velodyne俯视：azimuth=90, elevation=0, focalpoint=[32, 0, 0], distance=600
    adjust_view(90, 0, [5, 0, 0], 220, fig)  # 16


def get_lidar_in_image_fov(pc_lidar, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = utils.project_pc_to_image(pc_lidar, calib.P, calib.V2C)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_lidar[:, 0] > clip_distance)  # 俯瞰图下距离自身2.0m开外。
    imgfov_pc_lidar = pc_lidar[fov_inds, :]
    if return_more:
        return imgfov_pc_lidar, pts_2d, fov_inds
    else:
        return imgfov_pc_lidar


def show_lidar_on_image(pc_lidar, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    img1 = np.copy(img)
    imgfov_pc_lidar, pts_2d, fov_inds = get_lidar_in_image_fov(pc_lidar,
                                                               calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = utils.project_pc_to_camera(imgfov_pc_lidar, calib.V2C)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 0]
        # print(depth)
        color = cmap[int(640.0/depth), :]
        cv2.circle(img1, (int(np.round(imgfov_pts_2d[i, 0])),
                          int(np.round(imgfov_pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    # Image.fromarray(img).show()
    return img1


def dataset_viz():
    # save mkdir
    gt_result_dir = os.path.join(ROOT_DIR, 'inWater/result/gt')
    time_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    new_gt_result_dir = os.path.join(gt_result_dir, time_label)
    os.makedirs(new_gt_result_dir)
    lidar_gt_savedir = os.path.join(new_gt_result_dir, 'lidar_seq')
    pointsimage_savedir = os.path.join(
        new_gt_result_dir, 'pointsimage_seq')
    obj3dimage_savedir = os.path.join(new_gt_result_dir, 'obj3dimage_seq')
    obj2dimage_savedir = os.path.join(new_gt_result_dir, 'obj2dimage_seq')
    os.makedirs(lidar_gt_savedir)
    os.makedirs(pointsimage_savedir)
    os.makedirs(obj3dimage_savedir)
    os.makedirs(obj2dimage_savedir)

    dataset = inwater_object(os.path.join(
        ROOT_DIR, 'inWater/object'), split=split)
    annotator = DetectAnnotator(os.path.join(
        ROOT_DIR, 'inwater_tool/inwater.yaml'))

    for data_idx in range(len(dataset)):  # len(dataset)
        print(data_idx)
        # sensor data
        img = dataset.get_image(data_idx)  # cv::Mat
        pc_velo = dataset.get_velo(data_idx)[:, 0:3]
        pc_livox = dataset.get_livox(data_idx)[:, 0:3]
        pc = pc_velo  # pc_velo or pc_livox
        calib = dataset.get_calibration(data_idx)
        img_height, img_width, img_channel = img.shape
        print(('Image shape: ', img.shape))
        print('lidar shape:', pc.shape)
        # Show lidar points on image.
        lidar_on_img = show_lidar_on_image(
            pc, img, calib, calib.imgW, calib.imgH)
        # lidar_on_img = cv2.cvtColor(lidar_on_img, cv2.COLOR_BGR2RGB)
        # Image.fromarray(lidar_on_img).show()
        # Load data from dataset.
        objects = dataset.get_label_objects(data_idx)
        # Show all LiDAR points. Draw 3d box in LiDAR point cloud.
        show_lidar_with_boxes(pc, objects, calib,
                              True, img_width, img_height)
        # test
        # img_lidar_cv = show_lidar_with_boxes_cv(pc, objects, calib,
        #                                         True, img_width, img_height)
        # Image.fromarray(img_lidar_cv).show()
        # raw_input()
        # Draw 2d, 3d box in image.
        for object in objects[:]:
            if object.is_in_image() == False:
                objects.remove(object)
        print("object number: ", len(objects))
        if len(objects) > 0:
            # objects[0].print_object()
            show3d = True
            img2d, img3d = show_image_with_boxes(
                img, objects, calib, annotator, show3d)
            # img2d = cv2.cvtColor(img2d, cv2.COLOR_BGR2RGB)
            # Image.fromarray(img2d).show()
            # if show3d:
            #     img3d = cv2.cvtColor(img3d, cv2.COLOR_BGR2RGB)
            # Image.fromarray(img3d).show()
        else:
            img2d = img
            img3d = img
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Image.fromarray(img).show()
        # raw_input()
        # Save
        # lidar_pc_gt
        lidar_gt_filename = os.path.join(
            lidar_gt_savedir, "%06d.png" % (data_idx))
        save_fig(lidar_gt_filename)
        # pointsimage
        pointsimage_filename = os.path.join(
            pointsimage_savedir, "%06d.png" % (data_idx))
        cv2.imwrite(pointsimage_filename, lidar_on_img)
        # object 3d in image
        obj3dimage_filename = os.path.join(
            obj3dimage_savedir, "%06d.png" % (data_idx))
        cv2.imwrite(obj3dimage_filename, img3d)
        # object 2d in image
        obj2dimage_filename = os.path.join(
            obj2dimage_savedir, "%06d.png" % (data_idx))
        cv2.imwrite(obj2dimage_filename, img2d)
    print("Finished!")


if __name__ == '__main__':
    # import mayavi.mlab as mlab
    # from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d
    dataset_viz()
