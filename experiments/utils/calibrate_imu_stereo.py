import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from sympy import degree
import torch
from models.vgg_encoder_decoder import VGGNet, FCN8s
from torchvision import transforms
import utils.utils as utils
import math
import matplotlib.pyplot as plt
import math
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm


class ImuStereoCalibration():
    """
    This class performs a hand eye calibration between camera and imu
    
    Parameters
    ----------
    segmentation_model : torch.nn.Module
        pytorch segmentation CNN
    normalization : torch.transform.Normalize
        Normalizing parameters
    dataloader : torch dataset
        dataset which contains the data
    seq_from : int
        idx where first frame is read
    seq_to : int
        idx where last frame is read
    degrees : bool
        degrees or radians
    debug : bool
        prints more debug infos
    resize_factor : float
        resize images (can accelerate block matching)
    """
    def __init__(self, segmentation_model, normalization, dataloader, seq_from = 8200, seq_to = 8220, degrees=True, debug=False, resize_factor=1):
        self.frames = []
        self.imu_rot_mats = []
        self.stereo_rot_mats = []
        self.model = segmentation_model
        self.normalization = normalization
        self.dataloader = dataloader
        self.seq_from = seq_from
        self.seq_to = seq_to
        self.degrees = degrees
        self.roll_median = 0
        self.pitch_median = 0
        self.debug = debug
        self.resize_factor = resize_factor
        self.calibration_mat = np.eye(3)
        self.translations = []


    def calibrate(self, minDisparity = 0, numDisparities=512, blocksize=17):
        """
        segments an image, 3d reconstruction, estimates roll and pitch in stereo images and perform hand eye calibration
        
        Parameters
        ----------
        minDisparity : int 
            minimum disparity
        numDisparities : int
            number of disparities
        blocksize : int
            blocksize for block matching

        return : np.array
            rotation matrix
        """
        diff_mats = []
        for j in tqdm(range(self.seq_to - self.seq_from)):
            j += self.seq_from
            if j == self.seq_to:
                break

            sample = self.dataloader[j]

            img_l = sample["img_l"] / 255
            K_l = sample["K_l"]
            D_l = sample["D_l"]
            R_l = sample["R_l"]
            P_l = sample["P_l"]
            img_r = sample["img_r"] / 255
            K_r = sample["K_r"]
            D_r = sample["D_r"]
            R_r = sample["R_r"]
            P_r = sample["P_r"]
            R = sample["R"]
            T = sample["T"]
            imu = sample["imu"]

            imu = Rotation.from_quat(imu).as_euler("ZYX", degrees=False)

            map1_left, map2_left = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l, (img_l.shape[1], img_l.shape[0]), cv2.CV_32FC1)
            map1_right, map2_right = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r, (img_r.shape[1], img_r.shape[0]), cv2.CV_32FC1)

            img_l = cv2.remap(img_l, map1_left, map2_left, cv2.INTER_LINEAR)
            img_r = cv2.remap(img_r, map1_right, map2_right, cv2.INTER_LINEAR)

            res_img = cv2.resize(img_l, dsize=(960, 640), interpolation=cv2.INTER_CUBIC)
            # Perform forward pass with resized image as input tensor
            trans1 = transforms.ToTensor()
            input_tensor = self.normalization(trans1(res_img)).to("cuda:0")
            with torch.no_grad():
                output = self.model(input_tensor.unsqueeze(0).float())

            # Binarize output with a high threshold (less false positives (but also less true positives))
            output = utils.binarize(torch.squeeze(output.data.cpu()), activation="sigmoid", threshold=0.5).numpy()
            res_out = cv2.resize(output, dsize=(img_l.shape[1], img_l.shape[0]), interpolation=cv2.INTER_CUBIC)
            res_out *= 255
            res_out = res_out.astype(np.uint8)
            # # Mask original image
            img_l_masked = cv2.bitwise_and(img_l, img_l, mask = res_out)

            if self.debug:
                plt.imshow(img_l_masked)
                plt.show()

            ## resize image for faster stereo block matching
            img_l_masked_resized, img_r_resized, P_l_resize, P_r_resize = utils.resize_img_and_projection(img_l_masked, img_r, P_l, P_r, self.resize_factor)

            pc_water, disparity = utils.compute_pointcloud((img_l_masked_resized * 255).astype("uint8"), (img_r_resized * 255).astype("uint8"), P_l_resize, P_r_resize,
                                                            minDisparity=minDisparity, numDisparities=numDisparities, blocksize=blocksize)
            pc_water = pc_water[pc_water[:,2]<=40]

            normal, support = utils.calc_plane(pc_water, var_index=[0,2,1])
            normal = np.asarray(normal)
            support = np.asarray(support)
            normal = 1 / np.sqrt(np.sum(normal**2)) * normal

            roll, pitch = utils.normal_to_euler(normal, degrees=False)
            imu_roll = imu[2]
            imu_pitch = imu[1]
            imu_yaw = imu[0]

            imu_rot_mat = Rotation.from_euler("ZYX", [0., imu_pitch, imu_roll], degrees=False).as_matrix()
            stereo_rot_mat = Rotation.from_euler("ZYX", [0., pitch, roll], degrees=False).as_matrix()

            self.imu_rot_mats.append(imu_rot_mat)
            self.stereo_rot_mats.append(stereo_rot_mat) 

            diff_mats.append(np.dot(stereo_rot_mat.T, imu_rot_mat))

            self.translations.append(np.array([0, 0, 0]))


        self.calibration_mat, error = self.hand_eye_calibration(copy.deepcopy(self.imu_rot_mats), copy.deepcopy(self.stereo_rot_mats))
        self.calibration_mat = self.calibration_mat.T
        return self.calibration_mat


    def hand_eye_calibration(self, imu_mat, stereo_mat):
        """
        hand eye calibration
        
        Parameters
        ----------
        imu_mat : list
            list of imu measurements as dcm
        stereo_mat : list
            list of stereo pitch and roll measurements as dcm
        """
        imu_mats = []
        stereo_mats = []
 
        for i in range(len(imu_mat)):
            if i == 0:
                continue
            imu_mats.append(imu_mat[i-1].T @ imu_mat[i])
            stereo_mats.append(stereo_mat[i-1].T @ stereo_mat[i])

        M = np.zeros((3,3))
        for imu, stereo in zip(imu_mats, stereo_mats):
            # compute log of the matrix
            phi = np.arccos((np.trace(imu)-1) / 2)
            A1 = (phi/(2*np.sin(phi))) * (imu - imu.T)

            phi = np.arccos((np.trace(stereo)-1) / 2)
            B1 = (phi/(2*np.sin(phi))) * (stereo - stereo.T)

            a1 = np.array([A1[2,1], A1[0,2], A1[1,0]])
            b1 = np.array([B1[2,1], B1[0,2], B1[1,0]])

            M = M + np.outer(b1,a1)
        v, u = np.linalg.eig(M.T @ M)
        S = np.diag(v**(-1/2))
        S[S == np.inf] = 0
        R  = u @ S @ np.linalg.inv(u) @ M.T

        # compute error
        sum = 0
        for imu, stereo in zip(imu_mats, stereo_mats):
            # compute log of the matrix
            phi = np.arccos((np.trace(imu)-1) / 2)
            A1 = (phi/(2*np.sin(phi))) * (imu - imu.T)

            phi = np.arccos((np.trace(stereo)-1) / 2)
            B1 = (phi/(2*np.sin(phi))) * (stereo - stereo.T)

            a1 = np.array([A1[2,1], A1[0,2], A1[1,0]])
            b1 = np.array([B1[2,1], B1[0,2], B1[1,0]])

            sum += np.linalg.norm(R @ b1 - a1)**2

        return R, sum

    
    def plot_calibration_data(self, degrees=True):
        """plot roll and pitch for comparison"""
        imu_euler = [Rotation.from_matrix(rot).as_euler("ZYX", degrees=True) for rot in self.imu_rot_mats]
        stereo_euler = [Rotation.from_matrix(rot @ self.calibration_mat).as_euler("ZYX", degrees=True) for i,rot in enumerate(self.stereo_rot_mats)]

        imu_euler = np.array(imu_euler)
        stereo_euler = np.array(stereo_euler)
        
        x = np.arange(imu_euler.shape[0])
        fig, ax = plt.subplots()
        ax.set_title("Roll")
        ax.plot(x, imu_euler[:,2], 'o', label="imu")
        ax.plot(x, stereo_euler[:,2], 'o', label="stereo")
        ax.set_ylabel("degrees")
        ax.legend(loc = 'upper left')

        fig, ax = plt.subplots()
        ax.set_title("Pitch")
        ax.plot(x, imu_euler[:,1], 'o', label="imu")
        ax.plot(x, stereo_euler[:,1], 'o', label="stereo")
        ax.set_ylabel("degrees")
        ax.legend(loc = 'upper left')

        fig, ax = plt.subplots()
        ax.set_title("Yaw")
        ax.plot(x, imu_euler[:,0], 'o', label="imu")
        ax.plot(x, stereo_euler[:,0], 'o', label="stereo")
        ax.set_ylabel("degrees")
        ax.legend(loc = 'upper left')

        plt.show()