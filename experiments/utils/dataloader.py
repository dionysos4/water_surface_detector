import h5py
import os
from torch.utils.data import Dataset


class CONSTANCEDATASET(Dataset):
    """
    Pytorch constance imu orientation dataset
    
    Parameters
    ----------
    dataset_dir : str
        Directory where the dataset is stored
    day : int
        Sequence of day 1 or 2
    sequence : string
        Which sequence should be loaded
    """

    def __init__(self, dataset_dir, day=None, sequence=None):
        self.dataset_dir = dataset_dir
        self.img_list = []
        self.K_l_list = []
        self.K_r_list = []
        self.D_l_list = []
        self.D_r_list = []
        self.P_l_list = []
        self.P_r_list = []
        self.R_l_list = []
        self.R_r_list = []
        self.T_list = []
        self.R_list = []
        self.imu_list = []
        self.gyro_list = []
        self.day = day
        self.seq = sequence
        self.__load_data_to_ram()        


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        """return stereo images, calibration data and imu data"""
        img_path = self.img_list[idx]
        f = h5py.File(img_path)
        left_img = f['left_image/image'][:]
        right_img = f['right_image/image'][:]
        K_l = self.K_l_list[idx]
        K_r = self.K_r_list[idx]
        D_l = self.D_l_list[idx]
        D_r = self.D_r_list[idx]
        R_l = self.R_l_list[idx]
        R_r = self.R_r_list[idx]
        P_l = self.P_l_list[idx]
        P_r = self.P_r_list[idx]
        R = self.R_list[idx]
        T = self.T_list[idx]
        imu = self.imu_list[idx]
        gyro = self.gyro_list[idx]

        sample = {"img_l" : left_img,
                    "K_l" : K_l,
                    "D_l" : D_l,
                    "R_l" : R_l,
                    "P_l" : P_l,
                    "img_r" : right_img,
                    "K_r" : K_r,
                    "D_r" : D_r,
                    "R_r" : R_r,
                    "P_r" : P_r,
                    "R" : R,
                    "T" : T,
                    "imu" : imu,
                    "dq" : gyro}
        return sample

        
    def __load_data_to_ram(self):
        """loads data to ram"""
        if self.day == 1:
            self.dataset_dir = os.path.join(self.dataset_dir, "day_1")
        else:
            self.dataset_dir = os.path.join(self.dataset_dir, "day_2")
        
        for seq in sorted(os.listdir(self.dataset_dir)):
            if seq != None:
                if seq != "seq_" + self.seq:
                    continue
            sequence_path = os.path.join(self.dataset_dir, seq)
            for hdf5_file in sorted(os.listdir(sequence_path)):
                f = h5py.File(os.path.join(sequence_path, hdf5_file))
                self.imu_list.append(f['imu/orientation'][:])
                self.gyro_list.append(f['imu/dq'][:])
                self.img_list.append(os.path.join(sequence_path, hdf5_file))
                self.K_l_list.append(f['left_image/cam_info/K'][:].reshape(3,3))
                self.K_r_list.append(f['right_image/cam_info/K'][:].reshape(3,3))
                self.D_l_list.append(f['left_image/cam_info/D'][:])
                self.D_r_list.append(f['right_image/cam_info/D'][:])
                self.R_l_list.append(f['left_image/cam_info/R'][:].reshape(3,3))
                self.R_r_list.append(f['right_image/cam_info/R'][:].reshape(3,3))
                self.P_l_list.append(f['left_image/cam_info/P'][:].reshape(3,4))
                self.P_r_list.append(f['right_image/cam_info/P'][:].reshape(3,4))
                self.R_list.append(f['right_image/cam_info/R_r_l'][:].reshape(3,3))
                self.T_list.append(f['right_image/cam_info/t_r_l'][:])