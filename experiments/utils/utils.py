import torch.nn.functional as F
from torch.autograd import Variable
import torch
import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
import math
from scipy.spatial.transform import Rotation
import copy


def binarize(output, activation=None, threshold=0.5):
    """
    binarize an image by a given threshold.

    Parameters
    ----------
    output : np.array
        segmentation output
    activation : str
        activation function if necessary
    threshold : float
        threshold to classify as one or zero
    
    return: np.array
        binary image
    """
    # If desired add a elementwise none linearity to the output
    if activation=="sigmoid":
            output = torch.sigmoid(Variable(output)).data        
    elif activation=="softmax":
            output = F.softmax(Variable(output), dim=1).data
    elif activation is not None:
        raise NotImplementedError

    output[output>=threshold] = 1
    output[output<threshold] = 0
    return output


def compute_pointcloud(left_img, right_img, P_left, P_right, minDisparity = 0, numDisparities=512, blocksize=17):
    """ Generates pointcloud from stereo images by usage of SGBM

    Parameters
    ----------
    left_img : np.array
        left rectified image
    right_img : np.array
        right rectified image
    P_left : np.array
        projection matrix of the left camera
    P_right : np.array
        projection matrix of the right camera
    minDisparity : int
        minimum disparity
    numDisparities : int
        number of disparities
    block_size : int
        block size
    
    return: tuple
        point cloud and disparity map
    """ 
    gray_l = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
    gray_l = gray_l.astype("uint8")
    gray_r = gray_r.astype("uint8")
    stereo = cv2.StereoSGBM_create(minDisparity=minDisparity, numDisparities=numDisparities, blockSize=blocksize, preFilterCap=31, uniquenessRatio=15, P1=200, P2=400, speckleRange=4, speckleWindowSize=100)
    disp = stereo.compute(gray_l, gray_r)

    c_x = P_left[0,2]
    c_y = P_left[1,2]
    f = P_left[0,0]
    Tx = (P_right[:,3] / P_right[0,0])[0]
    c_x_ = P_right[0,2]
    Q = np.array([[1, 0, 0, -c_x], [0, 1, 0, -c_y], [0, 0, 0, f], [0, 0, -1/Tx, (c_x - c_x_)/Tx]])

    disp = disp / 16.0
    disp = disp.astype("float32")
    pc_img = cv2.reprojectImageTo3D(disp, Q)

    inf_mask_x = np.isfinite(pc_img[:,:,0])
    inf_mask_y = np.isfinite(pc_img[:,:,1])
    inf_mask_z = np.isfinite(pc_img[:,:,2])
    z_mask = pc_img[:,:,2] > 0

    inf_mask = inf_mask_x & inf_mask_y & inf_mask_z & z_mask
    pc = pc_img[inf_mask]
    rgb = left_img[inf_mask]
    pc = np.concatenate((pc, rgb), axis=1)
    pc = pc
    return pc, disp


def calc_plane(point_cloud, var_index=[0,1,2]):
    '''
    Calculates a regression plane in given point cloud using RANSAC with LinearRegression.
    var_index determines which columns contain the dependant and the independant variables.
    Last entry of the array is the index of the dependant variable.
    The axis of the dependant variable must not be parallel to the regression plane. 
    '''
    y = point_cloud[:,var_index[2]]
    X = -point_cloud[:,var_index[0:2]]
    lin_reg = LinearRegression()
    reg = RANSACRegressor(estimator=lin_reg, stop_probability=0.99, residual_threshold=0.2,max_trials=1000).fit(X, y)
    normal_vec, support_vec = ([0,0,0], [0,0,0])
    normal_vec[var_index[2]] = 1
    normal_vec[var_index[0]] = reg.estimator_.coef_[0]
    normal_vec[var_index[1]] = reg.estimator_.coef_[1]
    support_vec[var_index[2]] = reg.estimator_.intercept_

    return normal_vec, support_vec


def normal_to_euler(N, degrees=False):
    """
    Roll and pitch calculation.

    Parameters
    ----------
    N : np.array
        normal vector of the water surface
    degrees : bool
        if True == degrees else radians
    
    return: (float, float)
        roll and pitch
    """
    N /= np.linalg.norm(N)
    # opposite (Z-Komponente: Gegenkathete) / adjacent (Y-Komponente: Ankathete)
    pitch = math.atan2(N[2],N[1])
    # opposite (X-Komponente: Gegenkathete) / hypothenuse (XYZ-Komponente: Normiert = 1)
    roll = math.atan(N[0]/N[1])
    if degrees:
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
    return roll, pitch



def resize_img_and_projection(img_l, img_r, P_l, P_r, factor):
    """ resize image by factor and also the camera matrices

    Parameters
    ----------
    img_l : np.array
        left rectified image
    img_r : np.array
        right rectified image
    P_l : np.array
        projection matrix of the left camera
    P_r : np.array
        projection matrix of the right camera
    factor : float
        factor to resize images and projection matrices
    
    return: tuple
        scaled (left image, right image, left projection matrix, right projection matrix)
    """ 
    y_left, x_left, _ = img_l.shape
    y_right, x_right, _ = img_r.shape
    x_size = x_left * factor
    y_size = y_left * factor
    x_scale_left = x_size / x_left
    y_scale_left = y_size / y_left
    x_scale_right = x_size / x_right
    y_scale_right = y_size / y_right

    P_l_resize = copy.deepcopy(P_l)
    P_r_resize = copy.deepcopy(P_r)

    img_l = cv2.resize(img_l, None, fx= factor, fy= factor, interpolation= cv2.INTER_LINEAR)
    img_r = cv2.resize(img_r, None, fx= factor, fy= factor, interpolation= cv2.INTER_LINEAR)

    P_l_resize[0,:3] *= x_scale_left
    P_l_resize[1,:3] *= y_scale_left

    P_r_resize[0,:] *= x_scale_right
    P_r_resize[1,:3] *= y_scale_right    
    return img_l, img_r, P_l_resize, P_r_resize