#!/usr/bin/env python
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import copy
import math


def draw_points(img, points, color=(0,255,0)):
    """
    Draws points(x,y) in given image'''

    Parameters
    ----------
    img : np.array
        image
    points : np.array
        list of points
    return: np.array
        image with plotted points
    """
    scale = int(img.shape[0]/140)
    img_c = copy.deepcopy(img)
    points = np.squeeze(points)
    points = points[np.logical_and.reduce((points[:,0]<img_c.shape[1], points[:,1]<img_c.shape[0],
                              points[:,0]>=0, points[:,1]>=0))]
    for point in points:
        if point.shape[0] <3:
            img_c = cv2.circle(img_c, (point[0].astype(int),point[1].astype(int)), radius=scale, color=color, thickness=-1) 
        else:
            img_c = cv2.circle(img_c, (point[0].astype(int),point[1].astype(int)), radius=scale, color=get_color(point[2]), thickness=-1) 
    return img_c


def get_color(z, maxdist=30):
    """
    Return color, depending on the z value
    
    Parameters
    ----------
    z : float
        z value
    maxdist : int
        maximum distance
    return: tuple
        color tuple
    """

    scl = 50 / maxdist
    z *= scl
    # r = min(255, 305 -int(z*10))
    # g = min(255, int(z*10)-55)
    r =  min(255,-(z-5)**2/2+255)
    g = min(255,-(z-28)**2/2+255)
    b = min(255,- (z-46)**2/2+255)
    color = (max(0,r), max(0,g), max(0,b))
    return color


def calc_line(points, bounds=(1900)):
    """
    Calculates a line equation for given points
    The line connects the point with min(x) and the point with max(x)
    Only points in the defined bound are considered

    Parameters
    ----------
    points : np.array
        list of points
    bounds : tuple
        image bounds
    return: tuple
        intersect and slope
    """
    points = np.squeeze(points)
    points = points[np.logical_and(points[:,0]>=0, points[:,0]<bounds)]
    points = points[points[:, 0].argsort()]
    
    m =  (points[-1,1] - points[0,1])/(points[-1,0] - points[0,0])
    p = points[0]
    b = -m*p[0]+p[1]
    return b, m


def get_surface_points(roll, pitch, height, dist_horizon=5000):
    """
    This function returns three structured point clouds, representing the water surface in camera coordinates.
    Parameters are roll pitch and height of the camera system (in lidar/imu aligned camera coordinate system)
    
    Parameters
    ----------
    roll : float
        camera roll in imu aligned camera coordinates (z up)
    pitch : float
        camera pitch in imu aligned camera coordinates (z up)
    height : float
        camera height in imu aligned camera coordinates (z up)
    dist_horizon : int
        distance to horizon
    return: tuple
        tuple[0] : points lying on the water surface in a distance of 'dist_horizon'
        tuple[1] : points near to the camera lying on the water surface in a 1x1 meter grid
        tuple[2] : points lying on the water surface on a line, representing the x-Axis of the temporary world-coordinate system  
    """
    # Get the rotation of the surface in relation to the camera
    rot = Rotation.from_euler('YX',[pitch,roll]).as_matrix().T
    # Get the rotation between velodyne/imu aligned camera coordinate system and original camera coordinate system
    imu_to_cam = Rotation.from_euler('XZ', [1.5707963268, 1.5707963268]).as_matrix()
    # Combine both rotations. First rotate according to roll and pitch, than transfer in original camera coordinate system.
    rot = imu_to_cam @ rot
        
    ## Combine Rotation and Translation to a Projection Matrix
    plane_to_cam = np.zeros((4,4))
    plane_to_cam[:3,:3] = rot   
    plane_to_cam[1,3] = height
    plane_to_cam[3,3] = 1

    # Generate 3D Points in the 'plane coordinate' system
    obj_points1 = np.mgrid[dist_horizon:dist_horizon + 1, -int(dist_horizon/2):int(dist_horizon/2):100,0:1].reshape(3, -1).T.astype(float)
    obj_points2 = np.mgrid[3:23,-15:15,0:1].reshape(3, -1).T.astype(float)

    obj_points3 = np.mgrid[4:100, 0:1, 0:1].reshape(3, -1).T.astype(float)
        
    # Transfer the points from plane coordinate system into stereo-camera coordinate-system
    # to homogenous coordinates
    obj_points_stereo1 = np.insert(obj_points1, 3, 1, axis=1).astype(float)
    obj_points_stereo2 = np.insert(obj_points2, 3, 1, axis=1).astype(float)
    obj_points_stereo3 = np.insert(obj_points3, 3, 1, axis=1).astype(float)
    
    # to stereo coordinates
    obj_points_stereo1 = obj_points_stereo1 @ plane_to_cam.T
    obj_points_stereo1 = (obj_points_stereo1[:,:3].T / obj_points_stereo1[:,3]).T

    obj_points_stereo2 = obj_points_stereo2 @ plane_to_cam.T
    obj_points_stereo2 = (obj_points_stereo2[:,:3].T / obj_points_stereo2[:,3]).T
       
    obj_points_stereo3 = obj_points_stereo3 @ plane_to_cam.T
    obj_points_stereo3 = (obj_points_stereo3[:,:3].T / obj_points_stereo3[:,3]).T

    return obj_points_stereo1, obj_points_stereo2, obj_points_stereo3


def project_points(points, img, K, R= np.identity(3), T= np.array([[0.0,0.0,0.0]]), D=(0,0,0,0,0), horizon_color=None):
    """
    This function projects the points in a picture according to the given camera parameters

    Parameters
    ----------
    points : np.array
        list of points
    img : np.array
        image
    K : np.array
        camera matrix
    R : np.array
        extrinsic rotation
    T : np.array
        extrinsic translation
    D : tuple
        distortion parameters
    horizon_color : tuple
        color to plot the horizon
    return: np.array
        plotted image
    """

    # calculate the pixel positions of the 3D points
    pic_points, jacobian = cv2.projectPoints(points, R, T, K, D)
    img_c = copy.deepcopy(img)
    scale = int(img.shape[0]/200)
      
    # Add z information
    pic_points2 = np.squeeze(pic_points)
    pic_points2 = np.insert(pic_points2, 2, points[:,2],axis=1)
    # draw near field grid
    img_c = draw_points(img_c, pic_points2)

    return img_c


def project_lines(points, img, K, vlines=20, hlines=30, R=np.identity(3), T= np.array([[0.0,0.0,0.0]]), D=(0,0,0,0,0), color=(255,0,0), thickness=2):
    """
    This function projects lines in the image based on point positions

    Parameters
    ----------
    points : np.array
        list of points
    img : np.array
        image
    K : np.array
        camera matrix
    vlines : int
        number of vertical lines
    hlines : int
        number of horizontal lines
    R : np.array
        extrinsic rotation
    T : np.array
        extrinsic translation
    D : tuple
        distortion parameters
    color : tuple
        color tuple
    thickness : int
        thickness of lines
    return: np.array
        plotted image
    """
    img_c = copy.deepcopy(img)
    bounds = (0,0,img.shape[1], img.shape[0])
    if thickness is None:
        thickness = int(img.shape[0]/200)
    
    if vlines ==0:
        h_points = points[points[:, 0].argsort()].reshape([1, points.size//(3) ,3])
        h_points = np.expand_dims(h_points, 0)
    else:
        h_points = points[points[:, 2].argsort()].reshape([vlines, hlines,3])
    for row in h_points:
        pixels , _ = cv2.projectPoints(row, R, T, K, D)
        pixels = pixels.squeeze()
        left_px = pixels[pixels[:,0].argsort()][0]
        right_px = pixels[pixels[:,0].argsort()][-1]
        _, start, end = cv2.clipLine(bounds,(int(round(left_px[0])), int(round(left_px[1]))), (int(round(right_px[0])), int(round(right_px[1]))))
        cv2.line(img_c,start,end,color,thickness)

    if vlines != 0:
        v_points = points[points[:, 0].argsort()].reshape([hlines, vlines,3])
        for row in v_points:
            pixels , _ = cv2.projectPoints(row, R, T, K, D)
            pixels = pixels.squeeze()
            top_px = pixels[pixels[:,1].argsort()][0]
            bottom_px = pixels[pixels[:,1].argsort()][-1]
            _, start, end = cv2.clipLine(bounds,(int(round(top_px[0])), int(round(top_px[1]))), (int(round(bottom_px[0])), int(round(bottom_px[1]))))
            cv2.line(img_c,start,end,color,thickness)

    return img_c