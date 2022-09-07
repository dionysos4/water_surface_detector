#!/usr/bin/env python
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2, PointField

''' 
    File contains all point cloud related functions
    - generation from stereo images
    - transformation between rosmsg and numpy array
'''

DUMMY_FIELD_PREFIX = '__'

# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}


def fields_to_dtype(fields, point_step):
    '''
    Convert a list of PointFields to a numpy record datatype.
    '''
    offset = 0
    np_dtype_list = []
    for f in fields:
        while offset < f.offset:
            # might be extra padding between fields
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        dtype = pftype_to_nptype[f.datatype]
        if f.count != 1:
            dtype = np.dtype((dtype, f.count))

        np_dtype_list.append((f.name, dtype))
        offset += pftype_sizes[f.datatype] * f.count

    # might be extra padding between points
    while offset < point_step:
        np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
        offset += 1
        
    return np_dtype_list


def dtype_to_fields(dtype):
    '''
    Convert a numpy record datatype into a list of PointFields.
    '''
    fields = []
    for field_name in dtype.names:
        np_field_type, field_offset = dtype.fields[field_name]
        pf = PointField()
        pf.name = field_name
        if np_field_type.subdtype:
            item_dtype, shape = np_field_type.subdtype
            pf.count = np.prod(shape)
            np_field_type = item_dtype
        else:
            pf.count = 1

        pf.datatype = nptype_to_pftype[np_field_type]
        pf.offset = field_offset
        fields.append(pf)
    return fields


def pointcloud2_to_array(cloud_msg, squeeze=True):
    ''' 
    Converts a rospy PointCloud2 message to a numpy recordarra
    '''
    dtype_list = fields_to_dtype(cloud_msg.fields, cloud_msg.point_step)
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)

    cloud_arr = cloud_arr[[fname for fname, _type in dtype_list if not (fname[:len(DUMMY_FIELD_PREFIX)] == DUMMY_FIELD_PREFIX)]]
    if squeeze and cloud_msg.height == 1:
        return np.reshape(cloud_arr, (cloud_msg.width,))
    else:
        return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))


def array_to_pointcloud2(pointcloud, stamp=None, frame_id=None):
    '''
    Converts a numpy record array to a sensor_msgs.msg.PointCloud2
    '''
    cloud_arr = pointcloud.astype(np.float32)
    data = pointcloud.flatten().tostring()

    #cloud_arr = np.atleast_2d(cloud_arr)
    cloud_msg = PointCloud2()
    data = cloud_arr.flatten().tostring()

    if stamp is not None:
               cloud_msg.header.stamp = stamp
    if frame_id is not None:
         cloud_msg.header.frame_id = frame_id
    

    cloud_msg.height = 1
    cloud_msg.width = cloud_arr.shape[0]
    fields =  [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),    
    ]

    if cloud_arr.shape[1] == 6:
        fields += [ 
        PointField('r', 12, PointField.FLOAT32, 1),
        PointField('g', 16, PointField.FLOAT32, 1),
        PointField('b', 20, PointField.FLOAT32, 1)
    ]

    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False 
    cloud_msg.point_step = cloud_arr.shape[1] * 4
    cloud_msg.row_step = len(data)
    cloud_msg.is_dense = True 
    cloud_msg.data = data
    return cloud_msg


def crop_masked(img_1, img_2, mask, min_height=0):
    """
    Removes all rows which are black in 'mask' from the top of 'img1', 'img2', and 'img3' 
    
    Parameters
    ----------
    img1 : np.array
        image
    img2 : np.array
        image
    mask : np.array
        image
    min_height : int
        minimum height
    return: tuple
        tuple[0] : cropped img1
        tuple[1] : cropped img2
        tuple[2] : cropped mask
        tuple[3] : number of removed rows
    """

    mask1 = mask.any(1)
    row_start = mask1.argmax()
    row_start = min(row_start, mask.shape[0] - min_height)
    return img_1[row_start:,:,:], img_2[row_start:,:,:], mask[row_start:,:], row_start


def compute_pointcloud(left_img, disp, Q):
    """
    Compute pointcloud with given disparity map

    Parameters
    ----------
    left_img : np.array
        image
    disp : np.array
        disparity map
    Q : np.array
        reconstruction matrix
    return: np.array
        reconstructed point cloud
    """
    pc_img = cv2.reprojectImageTo3D(disp, Q)

    inf_mask_x = np.isfinite(pc_img[:,:,0])
    inf_mask_y = np.isfinite(pc_img[:,:,1])
    inf_mask_z = np.isfinite(pc_img[:,:,2])
    z_mask = pc_img[:,:,2] > 0

    inf_mask = inf_mask_x & inf_mask_y & inf_mask_z & z_mask
    pc = pc_img[inf_mask]
    rgb = left_img[inf_mask]/255
    pc = np.concatenate((pc, rgb), axis=1)
    return pc

     
def stereo_to_pointcloud(img_l, img_r, Q_orig, max_disp=256, block_size=5, min_disp=0, mask=None):
    """ Generates pointcloud from stereo images by usage of SGBM

    Parameters
    ----------
    img_l : np.array
        left rectified image
    img_r : np.array
        right rectified image
    Q_orig : np.array
        reconstruction matrix
    max_disp : int
        maximum disparity
    block_size : int
        block size
    min_disp : int
        minimum disparity
    mask : np.array
        segmentation mask
    return: tuple
        point cloud and disparity map
    """ 
    Q = np.copy(Q_orig)
    # Remove ros which are completly black in the right image in both images
    if mask is not None:
        img_l, img_r, mask, delta_cy = crop_masked(img_l, img_r, mask)
        Q[1,3] += delta_cy

    # Transform to gray scale and calculate Disparity
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_RGB2GRAY)
    gray_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
    gray_l = gray_l.astype("uint8")
    gray_r = gray_r.astype("uint8")

    stereo = cv2.StereoSGBM_create(numDisparities=max_disp, blockSize=block_size, preFilterCap=31, uniquenessRatio=15, P1=200, P2=400, speckleRange=4, speckleWindowSize=100)
    #stereo.setMinDisparity(min_disp)
    disp = stereo.compute(gray_l, gray_r)
    disp = disp / 16.0
    disp = disp.astype("float32")

    disp = cv2.bitwise_and(disp, disp, mask = mask)

    pc = compute_pointcloud(img_l, disp, Q)
    
    return pc, disp