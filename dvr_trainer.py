import numpy as np
import os
import pytorch_lightning as pl
import torch

from util_func import (
    image_to_world, get_random_pixels, get_value_at_pixel
)

from loss_func import (
    loss_rgb, loss_depth, loss_freespace, loss_normal,
    loss_occupancy
)

loss_str = ['loss', 'loss_rgb', 'loss_depth', 'normal_loss', 'loss_freespace', 'loss_occupied',
            'loss_sparse_depth', 'mask_intersection']

class DVR_trainer(pl.LightningModule):
    def __init__(self, model, lr, lr_schedule, gamma, generator, out_dir=None,
                 num_pixel_train=2048, num_pixel_eval=16000,
                 lambda_occupied=1., lambda_freespace=1., lambda_rgb=1.,
                 lambda_normal=0.05, lambda_depth=0.,
                 lambda_sparse_depth=0.,
                 depth_range=[0, 1400]):
        super().__init__()
        self.model = model
        self.generator = generator
        
        self.vis_dir = os.path.join(out_dir, "visualize")
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        
        self.lambda_occupied = lambda_occupied
        self.lambda_freespace = lambda_freespace
        self.lambda_rgb = lambda_rgb
        self.lambda_normal = lambda_normal
        self.lambda_depth = lambda_depth
        self.lambda_sparse_depth = lambda_sparse_depth
        self.depth_range = depth_range
        
        self.num_pixel_train = num_pixel_train
        self.num_pixel_eval = num_pixel_eval
    
    def validation_step(self, data, batch_idx):
        loss_dict = self.compute_loss(data, eval=True)
        
        return loss_dict
    
    def validation_epoch_end(self, outputs):
        loss_dict = {}
        for str in loss_str:
            loss_dict[str] = 0
        for str in loss_str:
            for output in outputs:
                loss_dict[str] += output.get(str, 0)
                
        for k, v in loss_dict.items():
            self.log("val/"+k, v/len(outputs))
            
        self.visualize()
        return loss_dict

    def training_step(self, batch, batch_index):
        loss_dict = self.compute_loss(batch, iter = self.global_step)
        loss = loss_dict['loss']
        for k, v in loss_dict.items():
            self.log("train/"+k, v)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), lr = self.lr)
        return {
            "optimizer" : optim,
            "lr_scheduler" : torch.optim.lr_scheduler.MultiStepLR(optim,
                                                                  self.lr_schedule, gamma=self.gamma)
        }
    
    def compute_loss(self, data, iter=None, eval=False):
        num_pixel = self.num_pixel_eval if eval else self.num_pixel_train
        
        device = self.device
        
        img = data['image'].to(device)
        mask_img = data['mask'].unsqueeze(1).to(device)
        W_mat = data["W_mat"].to(device)
        C_mat = data["C_mat"].to(device)
        S_mat = data["S_mat"].to(device)
        pixel_sparse=None
        gt_sparse=None
        depth_img=None
        depth_gt=None
        mask_depth=None
                
        loss = {}
        batch_size, _, _, _ = img.shape
        if(self.lambda_sparse_depth != 0.):
            pixel_sparse = data["pixel_sparse"]
            gt_sparse = data["gt_sparse"]
            num_sparse = int(1/4 * num_pixel)
            rand = np.random.choice(pixel_sparse.shape[1], num_sparse)
            pixel_sparse = pixel_sparse[:,rand,:].to(device)
            gt_sparse = gt_sparse[:,rand,:].to(device)
            
            pixels = get_random_pixels(batch_size, num_pixel - num_sparse).to(device)
            pixels = torch.concat([pixels, pixel_sparse], dim=1)
        else:
            pixels = get_random_pixels(batch_size, num_pixel).to(device)
            
        if(self.lambda_depth!=0.):
            depth_img = data['depth'].unsqueeze(1).to(device)
            depth_gt = get_value_at_pixel(depth_img, pixels).squeeze(-1)
            mask_depth = (depth_gt != np.inf)
            
        rgb_gt = get_value_at_pixel(img, pixels)
        mask_gt = get_value_at_pixel(mask_img, pixels).squeeze(-1).bool()
        
        p_hat, rgb_pred, o_ocu, o_free, mask_pred, normals = self.model(pixels, C_mat, W_mat, S_mat, mask_gt, iter,
                                                                        depth_gt, mask_depth)
            
        mask = mask_pred & mask_gt
        loss['loss_rgb'] = self.lambda_rgb * loss_rgb(rgb_pred[mask], rgb_gt[mask], batch_size)
    
        loss['normal_loss'] = self.lambda_normal * loss_normal(normals, batch_size) if normals!= None else 0
        
        loss['loss_freespace'] = self.lambda_freespace * \
            loss_freespace(o_free, batch_size)
            
        loss['loss_occupied'] = self.lambda_occupied * \
            loss_occupancy(o_ocu, batch_size)
            
        if(self.lambda_depth!=0.):
            mask_depth = mask & mask_depth
            p_gt = image_to_world(pixels, depth_gt.unsqueeze(-1), C_mat, W_mat, S_mat)
            loss['loss_depth'] = self.lambda_depth * \
                loss_depth(p_hat[mask_depth], p_gt[mask_depth], batch_size) \
                if mask_depth.sum()>0 else 0

        if(self.lambda_sparse_depth!=0. and gt_sparse!=None):
            num_sparse = int(1/4 * num_pixel)
            sparse_hat = p_hat[:,num_pixel-num_sparse:,:]
            mask_sparse = mask_pred[:,num_pixel-num_sparse:]
            loss['loss_sparse_depth'] = self.lambda_sparse_depth * \
                loss_depth(sparse_hat[mask_sparse], gt_sparse[mask_sparse], batch_size)
            
        loss_total = 0
        for _, v in loss.items():
            loss_total += v 
        loss['loss'] = loss_total
        
        loss['mask_intersection'] = (mask_gt == mask_pred).float().mean()
        return loss

    def visualize(self):
        if self.vis_dir != None and not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        mesh = self.generator.generate()
        filename = os.path.join(
            self.vis_dir, 'vis.ply')
        mesh.export(filename)