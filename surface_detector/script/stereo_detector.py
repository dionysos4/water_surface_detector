#!/usr/bin/env python
import rospy
import cv2
try:
  import py_cpp_utils
except:
  print("bayer_rggb12 encoding not possible!")
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation 
import numpy as np
import surface_visualisation
import pc_utils
import os, sys
from sklearn.linear_model import RANSACRegressor, LinearRegression
import math
from masknet.vgg_encoder_decoder import VGGNet, FCN8s
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import tf2_ros
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
import time


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
    if activation=="sofmax":
            output = F.softmax(Variable(output), dim=1).data
    output[output>=threshold] = 1
    output[output<threshold] = 0
    return output


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
    pitch = math.atan2(N[2],N[1])
    roll = math.atan(N[0]/N[1])
    if degrees:
        pitch = math.degrees(pitch)
        roll = math.degrees(roll)
    return roll, pitch


def get_transform_msg(time, orientation, position, parent_frame, child_frame):
    """
    Creates a transform message

    Parameters
    ----------
    time : timestamp
        timestamp for header.stamp
    orientation : geometry_msgs/Quaternion
        rotation
    position : np.array
        translation
    parent_frame: str
      frame_id
    child_frame: str
      child_frame_id
    return: geometry_msgs/TransformStamped
    """
    tf_msg = TransformStamped()
    tf_msg.header.stamp = time
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.rotation = orientation
    tf_msg.transform.translation.x = position[0]
    tf_msg.transform.translation.y = position[1]
    tf_msg.transform.translation.z = position[2]
    return tf_msg


class estimator_stereo:
  """
  Ros node to estimate roll, pitch and camera height of a stereo camera system
  """
  def __init__(self, args):
    self.name = rospy.get_name()
    self.load_parameters()
    self.visualisation = False if "-novis" in args else True
    self.disp_estimator = "sgbm"
    self.bridge = CvBridge()
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.load_segmentation_model()

    # subscribers and publishers
    self.image_l_sub = Subscriber("/stereo/left/image_raw",Image)
    self.image_r_sub = Subscriber("/stereo/right/image_raw", Image)
    self.cam_l_info_sub = Subscriber("stereo/left/camera_info", CameraInfo)
    self.cam_r_info_sub = Subscriber("stereo/right/camera_info", CameraInfo)
    self.sychronizer = ApproximateTimeSynchronizer([self.image_l_sub, self.image_r_sub, self.cam_l_info_sub, self.cam_r_info_sub], queue_size=1, slop=0.001)
    if self.visualisation:
      self.image_pub = rospy.Publisher("stereo_imu/horizon", Image, queue_size=5)
      self.marker_pub = rospy.Publisher("stereo_imu/surface", Marker, queue_size=5)
      self.pc_pub = rospy.Publisher("stereo_imu/surface_points", PointCloud2, queue_size=5)
    self.pose_pub = rospy.Publisher("stereo_imu/pose",PoseStamped, queue_size=5)
    self.sychronizer.registerCallback(self.callback)
    self.tf_br = tf2_ros.TransformBroadcaster()
    rospy.loginfo("%s: Initialization complete, waiting for data", self.name)


  def callback(self, image_l, image_r, info_l, info_r):
    time_in = image_l.header.stamp
    # if callback is called for first time, camera paramters are set to speed up proceeding runs
    if self.K_l is None:
      self.set_camera_params(info_l, info_r)

    s1 = time.time()  
    # scale, undistort and rectify images
    img_l, img_r = self.preprocess_images(image_l, image_r)
    s2 = time.time()
    # segmentation of water in left image.
    mask = self.mask_water(img_l)
    s3 = time.time()
    # Use sgbm for disparity estimation
    pc_water, disp = pc_utils.stereo_to_pointcloud(img_l, img_r, self.Q, max_disp=int(576/self.scale_disp), block_size=self.sgbm_block_size, mask=mask)
    s4 = time.time()
    # Remove points which are far away (in dimension of optical axis)
    pc_water= pc_water[pc_water[:,2]<=self.max_z]

    # base_camera_left is the coordinate system of the rectified coordinate system

    # Calculate surface normal 
    normal, support = self.calc_plane(pc_water, var_index=[0,2,1])
    s5 = time.time()
    print("pre ", s2-s1, " seg ", s3-s2, " stereo", s4-s3, " ransac ", s5-s4)
    normal = np.asarray(normal)
    support = np.asarray(support)
    normal = 1 / np.sqrt(np.sum(normal**2)) * normal
  
    # Calculate roll and pitch in 'camera_in_velodyne_coords'
    roll, pitch = normal_to_euler(normal, degrees=False)
    rotation = Rotation.from_euler("YX", [pitch, roll])
    quat = rotation.as_quat()
    translation = np.array([0, 0, np.dot(normal, support)])

    R_cam_water = Rotation.from_quat(quat).as_matrix().T
    t_cam_water = -R_cam_water @ translation
    R_cam_water_quat = Rotation.from_matrix(R_cam_water).as_quat()
    Tr_water_cam = get_transform_msg(time_in, Quaternion(R_cam_water_quat[0], R_cam_water_quat[1], R_cam_water_quat[2], R_cam_water_quat[3]), t_cam_water, "camera_in_velodyne_coords", "water_surface")
    self.tf_br.sendTransform(Tr_water_cam)


    pose_msg = PoseStamped()
    pose_msg.header.stamp = time_in
    pose_msg.header.frame_id = "water_surface"
    pose_msg.pose.orientation = Quaternion(quat[0], quat[1], quat[2], quat[3])
    pose_msg.pose.position.x = translation[0]
    pose_msg.pose.position.y = translation[1]
    pose_msg.pose.position.z = translation[2]
    self.pose_pub.publish(pose_msg)

    if self.visualisation:
      col_img = np.copy(img_l)
      col_img[(mask>0)] = [255,0,0]
      col_img = cv2.addWeighted(col_img, 0.3, img_l, 0.7, 0, col_img)
      points1, points2, points3 = surface_visualisation.get_surface_points(roll, pitch, support[1], 5000)

      # Visualisation with Lines
      cam_horizon = surface_visualisation.project_lines(points2, col_img, self.P_l[:3,:3], color=(0,255,0), thickness=2)
      # Horizon Visualisation
      cam_horizon = surface_visualisation.project_lines(points1, cam_horizon, self.P_l[:3,:3], vlines=0, hlines=1, thickness=2)

      img_msg = self.bridge.cv2_to_imgmsg(cam_horizon, 'rgb8')
      img_msg.header.stamp = time_in
      img_msg.header.frame_id = image_l.header.frame_id
      self.image_pub.publish(img_msg)

      pc_msg = pc_utils.array_to_pointcloud2(pc_water)
      pc_msg.header.stamp = time_in
      pc_msg.header.frame_id = image_l.header.frame_id
      self.pc_pub.publish(pc_msg)

      # Maker of the surface for visualisation with rviz
      marker = Marker()
      marker.header.frame_id = 'water_surface'
      marker.id = 300
      marker.type = marker.CUBE
      marker.action = marker.ADD
      marker.pose.position.x = 0
      marker.pose.position.y = 0
      marker.pose.position.z = 0
      marker.pose.orientation = Quaternion(0, 0, 0, 1)
   
      marker.scale.x = 500
      marker.scale.y = 500
      marker.scale.z = 0.001
      marker.color.a = 1
      marker.color.r = 0.0
      marker.color.g = 0.1
      marker.color.b = 0.9

      self.marker_pub.publish(marker) 
          
    rospy.loginfo("%s: roll: %f  pitch: %f", self.name, roll, pitch)


  def mask_water(self, image):
    """
    Segmentation of water surface and background

    Parameters
    ----------
    image : np.array
        image of a maritime scenario
    return: np.array
      image mask
    """
    # Prepare Resize
    original_x, original_y = image.shape[1] , image.shape[0]
    scale = self.scale_mask / self.scale_disp 
    new_x, new_y = int(original_x/scale), int(original_y/scale)

    # Transformation to normalize input images
    norm = transforms.Normalize(self.segmentation_data_mean, self.segmentation_data_std)

    if (scale != 1): 
      image = cv2.resize(image, dsize=(new_x,new_y), interpolation=cv2.INTER_AREA)
    
    # Perform forward pass with resized image as input tensor
    trans = transforms.ToTensor()
    input_tensor = norm(trans(image)).to(self.device)
    with torch.no_grad():
        output = self.segmentation_model(input_tensor.unsqueeze(0).float())

    # Binarize output with a high threshold (less false positives (but also less true positives))
    mask = binarize(torch.squeeze(output.data.cpu()), activation="sigmoid", threshold=self.thres_mask).numpy() 

    if (scale != 1):
    # Resize mask back to original size
      mask = cv2.resize(mask, dsize=(original_x,original_y), interpolation=cv2.INTER_CUBIC)

    mask = cv2.addWeighted(mask, alpha=255, src2=0, beta=0, gamma=0, dtype=cv2.CV_8U)
    
    return mask


  def preprocess_images(self, image_l, image_r):
    """
    Preprocessing of a raw image (12 bit conversion, demosaicing, rectify)
    Supported encodings (rgb8 and bayer_rggb12 (pylon format))

    Parameters
    ----------
    image_l : sensor_msgs/Image
        image of left camera
    image_r : sensor_msgs/Image
        image of right camera
    return: (np.array, np.array)
        preprocessed left and right image
    """
    
    if image_l.encoding != "bayer_rggb12":
      try:
        img_l = self.bridge.imgmsg_to_cv2(image_l, image_l.encoding)
      except CvBridgeError as e:
        print(e)

      try:
        img_r = self.bridge.imgmsg_to_cv2(image_r, image_r.encoding)
      except CvBridgeError as e:
        print(e)

    else:
      #12 bit to 16 bit
      image_l = py_cpp_utils.unpack_12bit_packed(image_l.data, image_l.step, image_l.width)
      image_r = py_cpp_utils.unpack_12bit_packed(image_r.data, image_r.step, image_r.width)
      
      #Bayer to RGB
      img_l = cv2.cvtColor(image_l, cv2.COLOR_BayerBG2RGB)
      img_r = cv2.cvtColor(image_r, cv2.COLOR_BayerBG2RGB)

      #16 bit to 8 bit
      img_l = cv2.addWeighted(img_l, alpha=1/257, src2=0, beta=0, gamma=0, dtype=cv2.CV_8U)
      img_r = cv2.addWeighted(img_r, alpha=1/257, src2=0, beta=0, gamma=0, dtype=cv2.CV_8U)
    
    # Crop image to 1920 x 1152. First 48 rows arent important, because they wont contain water anyway.
    img_l_crop= img_l[48:,:]
    img_r_crop= img_r[48:,:]
    
    # Rectify and resize images
    img_l_rect = cv2.remap(img_l_crop, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
    img_r_rect = cv2.remap(img_r_crop, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
    
    return img_l_rect, img_r_rect


  def set_camera_params(self, info_l, info_r):
    """
    Read camera calibration from ros message and save to member variables

    Parameters
    ----------
    info_l : sensor_msgs/CameraInfo
        calibration matrices of left camera
    info_r : sensor_msgs/CameraInfo
        calibration matrices of right camera
    """
    K_l = np.array(info_l.K).reshape(3,3)
    K_r = np.array(info_r.K).reshape(3,3)
    P_l = np.array(info_l.P).reshape(3,4)
    P_r = np.array(info_r.P).reshape(3,4)
    self.R_l = np.array(info_l.R).reshape(3,3)
    self.R_r = np.array(info_r.R).reshape(3,3)
    self.new_shape = (int(1920//self.scale_disp), int(1152//self.scale_disp)) 
    P_l[1,2] -=48
    P_r[1,2] -=48
    K_l[1,2] -=48
    K_r[1,2] -=48
    P_l[:2] = P_l[:2] / self.scale_disp
    P_r[:2] = P_r[:2] / self.scale_disp
    self.P_l = P_l
    self.P_r = P_r
    self.K_l = K_l
    self.K_r = K_r
    self.D_l = info_l.D
    self.D_r = info_r.D

    # Construct Q
    c_x = P_l[0,2]
    c_y = P_l[1,2]
    f = P_l[0,0]
    Tx = (P_r[:,3] / P_r[0,0])[0]
    c_x_ = P_r[0,2]
    self.Q = np.array([[1, 0, 0, -c_x], [0, 1, 0, -c_y], [0, 0, 0, f], [0, 0, -1/Tx, (c_x - c_x_)/Tx]])
    self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(self.K_l, self.D_l, self.R_l, self.P_l, (self.new_shape), cv2.CV_32FC1)
    self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(self.K_r, self.D_r, self.R_r, self.P_r, (self.new_shape), cv2.CV_32FC1)


  def load_segmentation_model(self):
    """
    Initialized neural network for image segmentation and load pretrained weights
    """
    rospy.loginfo("%s: Loading Segmentation Model", self.name)
    # Load VGG16 Encoder Decoder Model
    vgg_model = VGGNet(requires_grad=True, remove_fc=True, pretrained=False)
    model = FCN8s(pretrained_net=vgg_model, n_class=1).to(self.device)

    # Load trained state in model
    pkg_path = rospy.get_param("~pkg_path")
    path = os.path.join(pkg_path, self.segmentation_model_path)
    model.load_state_dict(torch.load(path))
    self.segmentation_model = model.eval()
    rospy.loginfo("%s: Segmentation Model loaded on %s", self.name, self.device)


  def calc_plane(self, point_cloud, var_index=[0,1,2]):
    ''' 
    Calculates a regression plane in given point cloud using RANSAC with LinearRegression.
    var_index determines which columns contain the dependent and the independent variables.
    Last entry of the array is the index of the dependent variable.
    The axis of the dependant variable must not be parallel to the regression plane. 
    '''
    y = point_cloud[:,var_index[2]]
    X = -point_cloud[:,var_index[0:2]]
    lin_reg = LinearRegression()
    reg = RANSACRegressor(base_estimator=lin_reg,residual_threshold=self.ransac_res_tresh,
    max_trials=self.ransac_max_trials, stop_probability=self.ransac_stop_prob).fit(X, y)
    rospy.loginfo("%s: Used %d RANSAC runs until stop criterion was met", self.name, reg.n_trials_)
    normal_vec, support_vec = ([0,0,0], [0,0,0])
    normal_vec[var_index[2]] = 1
    normal_vec[var_index[0]] = reg.estimator_.coef_[0]
    normal_vec[var_index[1]] = reg.estimator_.coef_[1]
    support_vec[var_index[2]] = reg.estimator_.intercept_
    return normal_vec, support_vec


  def load_parameters(self):
    """
    Load parameters from params.yaml
    """
    if not rospy.has_param("~pkg_path"):
      rospy.logerr("%s: Absolute path to package is missing. Make sure to pass the path in the private 'pkg_path' parameter", self.name)
      rospy.signal_shutdown("error")
    if not rospy.has_param('stereo'):
      rospy.logerr("%s: Configuration parameters could not be found. Make sure the contents of config/params.yaml have been loaded to the parameter server", self.name)
      rospy.signal_shutdown("error")
    else:
      self.segmentation_data_mean = rospy.get_param("stereo/masking/dataset_mean")
      self.segmentation_data_std = rospy.get_param("stereo/masking/dataset_std")

      self.scale_mask = rospy.get_param("stereo/masking/scale")
      self.thres_mask = rospy.get_param("stereo/masking/threshold")
      self.scale_disp = rospy.get_param("stereo/estimation/scale")
      self.max_z = rospy.get_param("stereo/estimation/surface_max_z")
      self.ransac_max_trials = rospy.get_param("stereo/estimation/ransac_max_trials")
      self.ransac_res_tresh = rospy.get_param("stereo/estimation/ransac_res_threshold")
      self.ransac_stop_prob = rospy.get_param("stereo/estimation/ransac_stop_prob")

      self.segmentation_model_path = rospy.get_param("stereo/masking/model_path")

      self.sgbm_block_size = rospy.get_param("stereo/sgbm/block_size")

      self.K_l = None
    

def main(args):
  name = "estimator_stereo"
  rospy.init_node(name, anonymous=False)
  estimator = estimator_stereo(args)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print(name + ": Shutting down")

if __name__ == '__main__':
    main(sys.argv)