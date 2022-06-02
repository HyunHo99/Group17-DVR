import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch import distributions as dist

from model.rayCasting import Module_RayCasting
from util_func import image_to_world, get_random_points

class DVR(nn.Module):
    def __init__(self, decoder,
                 device=None, depth_range = [0, 1400]):
        super().__init__()
        self.decoder = decoder.to(device)
        self.depth_range = depth_range

        self.device = device
        self.rayCaster = Module_RayCasting(depth_range=depth_range)

    def forward(self, pixels, C_mat, W_mat, S_mat, mask_gt, iter=None, depth_gt=None, mask_depth=None):
        device = self.device
        batch_size, _, _ = pixels.shape
        
        p_hat, mask_object = self.pixels_marching(pixels, C_mat, W_mat, S_mat, iter)
        rgb_pred = self.decoder(p_hat, texture=True)
        
        p_ocu = get_random_points(pixels, C_mat, W_mat, S_mat, self.depth_range, depth_gt, mask_depth)
        p_free = get_random_points(pixels, C_mat, W_mat, S_mat, self.depth_range)
        
        p_ocu = p_ocu[(mask_object == 0) & mask_gt].view(batch_size, -1, 3)
        o_ocu = self.decoder(p_ocu)
        
        p_free[mask_object] = p_hat[mask_object].detach()
        p_free = p_free[mask_gt == 0].view(batch_size, -1, 3)
        o_free = self.decoder(p_free)
        
        normals = None
        if (mask_object.sum()>0):
            normals = self.get_normals(p_hat[mask_object].detach())

        return p_hat, rgb_pred, o_ocu, o_free, mask_object, normals
        

    def get_normals(self, points, neighbor_dist=1e-3, gradiant_distance=1e-3):
        neighbors = points + (torch.rand_like(points) * neighbor_dist - (neighbor_dist / 2))
        grad = self.calculate_gradiant(torch.stack([points, neighbors]), d=gradiant_distance)
        normals_p = F.normalize(grad[0])
        normals_neighbor = F.normalize(grad[1])

        return [normals_p, normals_neighbor]

    def calculate_gradiant(self, points, d=1e-3):
        batch, n_points, _ = points.shape
        dxyz = d/2 * torch.cat([torch.eye(3), -torch.eye(3)]).repeat(batch, n_points, 1, 1).to(self.device)
        points = (points.unsqueeze(2).repeat(1,1,6,1) + dxyz).view(batch, -1, 3)
        fxyz = self.decoder(points).view(batch, n_points, 6)
        gradient = fxyz[:,:,:3] - fxyz[:,:,3:6]
        
        return gradient
    
    def get_prob(self, p): 
        logits=self.decoder(p)
        if torch.isnan(logits).sum() !=0:
            breakpoint()
        return dist.Bernoulli(logits=logits)

    def pixels_marching(self, pixels, C_mat, W_mat, S_mat,
                        iter=None, steps=None):
        device = self.device
        batch_size, num_pixels, _ = pixels.shape

        p_world = image_to_world(pixels, torch.ones(batch_size, num_pixels, 1).to(device), 
                                 C_mat, W_mat, S_mat)
        cam_o = image_to_world(pixels, torch.zeros(batch_size, num_pixels, 1).to(device), 
                               C_mat, W_mat, S_mat)
        dir = p_world - cam_o
        
        d_tmp = self.rayCaster(cam_o, dir, self.decoder, iter=iter, steps=steps)
        
        mask_object = (d_tmp != np.inf).detach()
        d_hat = torch.zeros_like(d_tmp).to(device)
        d_hat[mask_object] = d_tmp[mask_object]
        
        p_hat = cam_o + dir * d_hat.unsqueeze(-1)
        
        return p_hat, mask_object