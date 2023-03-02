import os
import numpy as np
import torch
import shutil
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from zipfile import ZipFile

__all__ = ["Water"]

class Water(Dataset):
    """
    Pytorch water segmentation dataset. Contains images and corresponding masks taken from a vessel (600 pictures),
    as well as a very broad variety of water surface images taken from land (2400 pictures).
    Also contains 30 images and masks taken by Solgenia Sensorsystem at Lake of Constance.
    Dataset is downloaded from ios share.
    
    Originally taken from:
        https://etsin.fairdata.fi/dataset/e0c6ef65-6e1e-4739-abe3-0455697df5ab
    and:
       https://www.kaggle.com/gvclsu/water-segmentation-dataset 

    Parameters
    ----------
    dataset_dir : str
        Directory where the dataset is stored
    data_list_tamp
        defines the subset of the tampere images to be contained by the dataset
        Available are 'channel', 'dock' and 'open'
    data_list_misc
        defines the subset of the kaggle images to be contained by the dataset
        Available are either: 'training' and/or 'validation'
        OR: 'ADE20K', 'river_segs', 'aberlour', 'auldgirth', 'bewdley', 'cockermouth', 'dublin','evesham-lock','galway-city','holmrook','keswick_greta', 'worcester'
        'stream0', 'stream1', 'stream3_small', 'stream2', 'boston_harbor2_small_rois', 'buffalo0_small', 'canal0', 'mexico_beach_clip0', 'holiday_inn_clip0', 'gulf_crest'
    data_constance: boolean
        defines if images of the Lake of Constance should be contained in the dataset
    transform : transform
        transform operations to be applied
    img_size : (int,int)
        resize every image to the size specified by the tuple. If not set original size is kept    
    """

    def __init__(self, dataset_dir, data_list_tamp=['channel', 'dock', 'open'], data_list_misc=['training', 'validation'], data_constance=True, transforms=None, img_size=(None, None)):
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.img_size = img_size
        self.sub_dir_tamp = "tampere"
        self.sub_dir_misc = "water_v2"
        self.sub_dir_constance = "constance"
        self.masks = []
        self.imgs = []
        self.data_list_tamp = data_list_tamp
        self.data_list_misc = data_list_misc
        self.data_constance = data_constance

        #### Finding and adding Water Misc Images to self.imgs and self.masks ###  
        if data_list_misc:
            # List of directories containing the desire data
            dirs = []

            if 'training' in data_list_misc:    
                stream = open(os.path.join(dataset_dir, self.sub_dir_misc, "train.txt"), mode='r')
                dir_names = stream.readlines()
                dirs += ([i.replace('\n','') for i in dir_names])
        
            if 'validation' in data_list_misc:
                stream = open(os.path.join(dataset_dir, self.sub_dir_misc, "val.txt"), mode='r')
                dir_names = stream.readlines()
                dirs += ([i.replace('\n','') for i in dir_names])   
          
            if not 'validation' in data_list_misc and not 'training' in data_list_misc:
                dirs = data_list_misc

            # save all subfolder names containing the images/masks
            img_dirs = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_misc, "JPEGImages"))))
            mask_dirs = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_misc, "Annotations"))))   

            # search every subdirectory in the "Annotations" directory for pngs or jps
            # save the path to every mask
            self.masks = []
            for directory in mask_dirs:
                if directory in dirs:
                    mask_list = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_misc, "Annotations", directory))))
                    for mask in mask_list:
                        if mask.endswith("png") or mask.endswith("jpg"):
                            self.masks.append(os.path.join(self.sub_dir_misc, "Annotations", directory, mask))

            # search every subdirectory in the "JPEGImages" directory for pngs or jps
            # check if an corresponding image with the same name already exists in the masks list
            # save the path of those images
            self.imgs = []
            idx = 0
            for directory in img_dirs:
                if directory in dirs:
                    img_list = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_misc, "JPEGImages", directory))))
                    for img in img_list:
                        if idx >= len (self.masks):
                            break
                        # crop the path of the corresponding mask to check if there is a fitting mask for the current image
                        mask_crop = self.masks[idx].split('/')[-2:]
                        mask_crop = '/'.join(mask_crop).split(".",1)[0]
                        if mask_crop == os.path.join(directory, img).split(".",1)[0]:
                            self.imgs.append(os.path.join(self.sub_dir_misc, "JPEGImages", directory, img))
                            idx += 1

        #### Finding and adding Tampere Imgs to self.imgs and self.masks ###
        if data_list_tamp:
            mask_dirs = []
            img_dirs = []

            # save all subfolder names containing the images/masks
            dirs = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_tamp))))
            for directory in dirs:
                if directory.endswith("mask") and directory.rsplit("_")[0] in data_list_tamp:
                    mask_dirs.append(directory)
                    img_dirs.append(directory.rsplit("_")[0])
      
            # save the paths to all images/masks in the subfolders
            idx = 0
            for directory in img_dirs:
                img_list = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_tamp, directory))))
                for img in img_list:
                    self.imgs.append(os.path.join(self.sub_dir_tamp,directory, img))
        
            for directory in mask_dirs:
                mask_list = list(sorted(os.listdir(os.path.join(dataset_dir, self.sub_dir_tamp, directory))))
                for mask in mask_list:
                    self.masks.append(os.path.join(self.sub_dir_tamp,directory, mask))  

        #### Finding and adding Constance Imgs to self.imgs and self.masks ###
        if data_constance:
            mask_dir = os.path.join(dataset_dir, self.sub_dir_constance, "masks")
            mask_list = list(sorted(os.listdir(mask_dir)))
            for img_name in mask_list:
                self.imgs.append(os.path.join(self.sub_dir_constance, "imgs", img_name))
                self.masks.append(os.path.join(self.sub_dir_constance, "masks", img_name)) 

         
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.dataset_dir, self.imgs[idx])
        mask_path = os.path.join(self.dataset_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.img_size[0] is not None:
            mask = mask.resize(self.img_size, resample=Image.BICUBIC)
            img = img.resize(self.img_size, resample=Image.BICUBIC)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # convert the numpy array to a binary mask
        if mask.ndim == 3:
            mask = (mask[:, :,0])
        mask[mask>0]=1
        mask[mask<=0]=0
        # convert mask and image to tensor
        mask = torch.as_tensor(mask, dtype=torch.float32)
        mask.unsqueeze_(dim=0)
        trans = transforms.ToTensor()
        img = trans(img)
        
        if self.transforms is not None:
            img= self.transforms(img)
        
        return img, mask

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = Water('/home/konstantin/projects/WaterDataset/', data_list_tamp=[], data_list_misc=[], transforms=None)
    print(len(dataset))
     

        