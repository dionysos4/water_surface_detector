#!/usr/bin/env python
import sys
import rospy
import pc_utils
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from visualization_msgs.msg import Marker
from nav_msgs.msg import OccupancyGrid
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scipy.spatial.transform import Rotation 
import numpy as np
import tf2_ros
import sys


class obstacle_detector:
  """
  Ros node to create an obstacle map with the detected water surface and lidar measurements
  """
  def __init__(self, args):
    self.name = rospy.get_name()
    self.load_parameters()
    self.visualisation = False if "-novis" in args else True
    self.stereo = True if "-stereo" in args else False

    self.imu_sub= Subscriber("/stereo_imu/pose", PoseStamped, queue_size=1)

    # Broadcaster for water surface frame
    self.tf_br = tf2_ros.TransformBroadcaster()
    self.tf_buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(self.tf_buffer, buff_size = 131072)

    self.pc_sub = Subscriber(self.lidar_topic , PointCloud2, queue_size=1)
    self.grid_pub = rospy.Publisher("obstacle/grid", OccupancyGrid, queue_size=1)
    
    self.sychronizer = ApproximateTimeSynchronizer([self.imu_sub, self.pc_sub], queue_size=10, slop=0.1)
    self.sychronizer.registerCallback(self.callback)

    rospy.loginfo("%s: Running, waiting for data", self.name)


  def callback(self,imu, pc_msg):
    time_in = pc_msg.header.stamp

    # Transform from 'velodyne' to recently published frame 'water surface' 
    try:
        velodyne_to_target= self.tf_buffer.lookup_transform("water_surface", pc_msg.header.frame_id, time_in, rospy.Duration(0.1))        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr("%s: Transform of velodyne to %s frame failed", self.name, "water_surface")
        return   
      
    # Get grid and obstacle pc
    grid_msg = self.get_grid(pc_msg, velodyne_to_target)
    grid_msg.header.stamp = time_in
    grid_msg.header.frame_id = "water_surface"
    self.grid_pub.publish(grid_msg)
    rospy.loginfo("%s: Grid updated", self.name)


  def get_grid(self, pc_msg, transformation):
    """
    Transforms pointcloud message according to transformation. Maps points to a 2D Occupancy Grid
    
    Parameters
    ----------
    pc_msg : sensor_msgs/PointCloud2
        lidar point cloud
    transformation : geometry_msgs/TransformStamped
        transformation to map lidar points into new frame
    return: nav_msgs/OccupancyGrid
        occupancy grid
    """

    pc = pc_utils.pointcloud2_to_array(pc_msg)
    
    x = pc['x']
    y = pc['y']
    z = pc['z']
                
    pc = np.zeros((x.flatten().shape[0], 3))
    pc[:,0] = x.flatten()
    pc[:,1] = y.flatten()
    pc[:,2] = z.flatten()

    # remove invalid points
    pc = pc[np.isfinite(pc).any(axis=1)]

    quat = transformation.transform.rotation
    R = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()
    translation = transformation.transform.translation

    # R_describes rotation from Lidar-System to water surface frame
    pc  = pc @ R.T
    pc = pc  + [translation.x, translation.y, translation.z]

    # Remove points to high or to low
    pc = pc[pc[:,2]>=self.min_height,:]
    pc = pc[pc[:,2]<=self.max_height,:]
        
    # Remove reflection from boat
    pc = pc[np.logical_or.reduce((pc[:,1]>=self.dist_l, pc[:,1]<=-self.dist_r, pc[:,0]>=self.dist_f, pc[:,0]<=-self.dist_b))]

    # Remove z-component, scale with resolution and round to integer
    pc = np.around(pc[:,:2] * self.grid_res, decimals=0).astype(int)

    # Remove points which are out of scope for the grid shape
    pc = pc[pc[:,1]<self.grid_height//2]
    pc = pc[pc[:,0]<self.grid_width//2]
    pc = pc[pc[:,1]>= -self.grid_height//2]
    pc = pc[pc[:,0]>= -self.grid_width//2]

    # Transform points to grid coordinate system
    pc[:,0] += self.grid_width//2
    pc[:,1] += self.grid_height//2

    # Build grid
    grid_data = np.zeros((self.grid_width, self.grid_height), dtype=int)
    grid_data[tuple(pc.T)] = 100
    
    grid_msg = OccupancyGrid()
    grid_msg.data = grid_data.flatten(order='F')
    grid_msg.info.resolution = 1/self.grid_res
    grid_msg.info.width = self.grid_width
    grid_msg.info.height = self.grid_height
    grid_msg.info.origin.position.x = -self.grid_width/(self.grid_res*2)
    grid_msg.info.origin.position.y = -self.grid_height/(self.grid_res*2)
    grid_msg.info.origin.position.z = 0
        
    return grid_msg


  def load_parameters(self):
    """load parameters from config file"""
    if not rospy.has_param('grid'):
      rospy.logerr("Configuration parameters could not be found. Make sure the contents of config/params.yaml have been loaded to the parameter server")
      rospy.signal_shutdown("error")
    else:
      self.grid_width = rospy.get_param("grid/width")
      self.grid_height = rospy.get_param("grid/height")
      self.grid_res = rospy.get_param("grid/resolution")
      self.max_height = rospy.get_param("grid/max_height")
      self.min_height = rospy.get_param("grid/min_height")
      self.dist_f = rospy.get_param("grid/dist_front")
      self.dist_b = rospy.get_param("grid/dist_back")
      self.dist_r = rospy.get_param("grid/dist_right")
      self.dist_l = rospy.get_param("grid/dist_left")
      self.lidar_topic = rospy.get_param("lidar/topic")


def main(args):
  name = "obstacle_detector"
  rospy.init_node(name, anonymous=False)
  estimator = obstacle_detector(args)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print(name + ": Shutting down")

if __name__ == '__main__':
    main(sys.argv)