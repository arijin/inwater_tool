import numpy as np
import geopy as gp


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.occlusion = int(data[2])
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        # location (x,y,z) in camera coord.
        self.t = (data[11], data[12], data[13])
        self.rr = data[14]  # roll angle (around X-axis) [-pi..pi]
        self.rp = data[15]  # pitch angle (around Y-axis) [-pi..pi]
        self.ry = data[16]  # yaw angle (around Z-axis) [-pi..pi]

    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' %
              (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' %
              (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' %
              (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f, %f, %f' %
              (self.t[0], self.t[1], self.t[2], self.rr, self.rp, self.ry))

    def is_in_image(self):
        if self.xmin == 0 and self.ymin == 0 and self.xmax == 0 and self.ymax == 0:
            return False
        else:
            return True


class Calibration(object):
    def __init__(self, calib_filepath):
        # input the data corresponding to the undistorted images
        calibs = self.read_calib_file(calib_filepath)

        self.imgH = calibs['image_height']
        self.imgW = calibs['image_width']
        # Rigid transform from Velodyne coord to base link coord
        self.V2B = calibs['Tr_velo_to_base']  # bv_T
        self.V2B = np.reshape(self.V2B, [3, 4])
        self.B2V = gp.inverse_rigid_trans(self.V2B)  # vb_T
        # Rigid transform from Camera coord to base link coord
        self.C2B = calibs['Tr_cam_to_base']  # bc_T
        self.C2B = np.reshape(self.C2B, [3, 4])
        self.B2C = gp.inverse_rigid_trans(self.C2B)  # cb_T

        self.V2C = np.dot(gp.Tcart2hom(self.B2C),
                          gp.Tcart2hom(self.V2B))  # cv_T
        self.V2C = self.V2C[0:3, :]

        # Camera intrinsics and extrinsics
        self.P = calibs['P']
        self.P = np.reshape(self.P, [3, 4])

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
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


def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
             ExtrincT:      3x4 extrinsics
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_3d_extend = pts_3d_extend[:, [1, 2, 0, 3]]
    pts_3d_extend[:, 0] = -pts_3d_extend[:, 0]
    pts_3d_extend[:, 1] = -pts_3d_extend[:, 1]
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def project_pc_to_camera(pts_3d, ExtrincT):
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_3d = np.dot(pts_3d_extend, np.transpose(
        ExtrincT))  # nx4 4x3 = nx3
    return pts_3d


def project_pc_to_image(pts_3d, P, ExtrincT=None):
    if ExtrincT is not None:
        n = pts_3d.shape[0]
        pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
        pts_3d = np.dot(pts_3d_extend, np.transpose(
            ExtrincT))  # nx4 4x3 = nx3
    pts_2d = project_to_image(pts_3d, P)
    return pts_2d


def compute_box_3d(obj, P, ExtrincT=None):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = gp.euler_to_Rmatrix(obj.ry, obj.rp, obj.rr)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    '''
    qs: (8,3) array of vertices for the 3d box in following order:
          5 -------- 4
         /|              /|
        6 -------- 7 .
        |  |             |  |
        . 1 --------  0
        |/       o     |/
        2 --------  3
    '''
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [0, 0, 0, 0, h, h, h, h]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]

    corners_3d = np.transpose(corners_3d)  # 8x3

    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[:, 0] < 0.1):
        corners_2d = None
        return corners_2d, corners_3d

    if ExtrincT is not None:
        n = corners_3d.shape[0]
        pts_3d_extend = np.hstack((corners_3d, np.ones((n, 1))))
        corners_3d = np.dot(pts_3d_extend, np.transpose(
            ExtrincT))  # 8x4 4x3 = 8x3

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(corners_3d, P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, corners_3d
