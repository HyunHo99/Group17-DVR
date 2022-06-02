import torch
from torch.nn import functional as F
import torch.nn as nn

def loss_rgb(rgb_pred, rgb_gt, batch_size):
    loss_fn = nn.L1Loss(reduction='sum') 
    return loss_fn(rgb_pred, rgb_gt) / batch_size

def loss_depth(depth_pred, depth_gt, batch_size):
    return torch.norm((depth_gt - depth_pred), dim=-1).sum() / batch_size

def loss_normal(normals, batch_size):
    norm = torch.norm(normals[0] - normals[1], dim=-1)
    return norm.sum() / batch_size
    
def loss_freespace(logit_pred, batch_size):
    loss = F.binary_cross_entropy_with_logits(logit_pred, torch.zeros_like(logit_pred), reduction='sum')
    return loss / batch_size

def loss_occupancy(logit_pred, batch_size):
    loss = F.binary_cross_entropy_with_logits(logit_pred, torch.ones_like(logit_pred), reduction='sum')
    return loss / batch_size

