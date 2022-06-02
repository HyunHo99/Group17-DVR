from torch.utils.data import Dataset
import glob
import numpy as np
import imageio
from PIL import Image
from pathlib import Path
from torchvision import transforms

class DTUdataset(Dataset):
    def __init__(self, dir_data, ignore_idx=[], depth_type = 'mvs'):
        self.depths = []
        self.images = []
        self.masks = []
        self.W_mats = []
        self.S_mats = []
        self.C_mats = []
        self.gt_sparses = []
        self.pixel_sparses = []
        self.transform_img = transforms.ToTensor()
        self.dir_data = dir_data
        self.ignore_idx = ignore_idx
        self.idx = []
        self.depth_type = depth_type
        self.add_images()
        self.add_cam()
        
        if(depth_type=='mvs'):
            self.add_depths()
            assert(len(self.images) == len(self.depths))
        elif(depth_type=='sfm'):
            self.add_pcl_depths()
            assert(len(self.images) == len(self.gt_sparses))
    
        self.add_masks()
        
        assert(len(self.images) == len(self.masks))
        assert(len(self.C_mats) == len(self.S_mats) == len(self.W_mats) == len(self.images))
        
    def __getitem__(self, index):
        data_dict = {
            'image' : self.images[index],
            'mask' : self.masks[index],
            'W_mat' : self.W_mats[index],
            'S_mat' : self.S_mats[index],
            'C_mat' : self.C_mats[index],
        }
        if(self.depth_type=='mvs'):
            data_dict['depth'] = self.depths[index]
        elif(self.depth_type=='sfm'):
            data_dict['gt_sparse'] = self.gt_sparses[index]
            data_dict['pixel_sparse'] = self.pixel_sparses[index]
        return data_dict
        
    def __len__(self):
        return len(self.images)
    
    def add_images(self):
        for file in sorted(glob.glob(self.dir_data + "/image/*.png")):
            if int(Path(file).stem) not in self.ignore_idx:
                self.idx.append(int(Path(file).stem))
                image = Image.open(file).convert("RGB")
                image = self.transform_img(image)
                self.images.append(image)
                
    def add_depths(self):
        for file in sorted(glob.glob(self.dir_data + "/depth/*.exr")):
            if int(Path(file).stem) not in self.ignore_idx:
                depth = np.array(imageio.imread(file)).astype(np.float32)
                depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
                self.depths.append(depth)
                
                
    def add_masks(self):
        for file in sorted(glob.glob(self.dir_data + "/mask/*.png")):
            if int(Path(file).stem) not in self.ignore_idx:
                mask = np.array(imageio.imread(file)).astype(np.bool)
                mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
                self.masks.append(mask.astype(np.float32))   
                
    def add_cam(self):
        cam_dict = np.load(self.dir_data + '/cameras.npz')
        for idx in self.idx:
            self.W_mats.append(cam_dict['world_mat_%d' % idx].astype(np.float32))
            self.S_mats.append(cam_dict['scale_mat_%d' % idx].astype(np.float32))
            self.C_mats.append(cam_dict['camera_mat_%d' % idx].astype(np.float32))
                
    def add_pcl_depths(self):
        pcl = np.load(self.dir_data + '/pcl.npz')
        is_in_visual_hull = pcl['is_in_visual_hull']
        points = pcl['points']
        for i, idx in enumerate(self.idx):
            visible = pcl['visibility_%04d' % idx]
            points_i = points[visible][is_in_visual_hull[visible]]

            S_mat = self.S_mats[i]
            W_mat = self.W_mats[i]
            C_mat = self.C_mats[i]
            p_homo = np.concatenate([points_i, np.ones((points_i.shape[0], 1))], 
                        axis=-1).transpose(1, 0)
            gt_sparse = np.linalg.inv(S_mat) @ p_homo
            gt_sparse = gt_sparse[:3].transpose(1, 0).astype(np.float32)
            
            pixel_sparse = C_mat @ W_mat @ p_homo
            pixel_sparse = (pixel_sparse[:2] / pixel_sparse[-2].reshape(1, -1)).transpose(1, 0).astype(np.float32)
            self.gt_sparses.append(gt_sparse)
            self.pixel_sparses.append(pixel_sparse)