import torch
import torch.nn.functional as F

def get_random_pixels(batch_size, n_points):
    return (torch.rand(batch_size, n_points, 2) * 2) - 1

def get_random_points(pixels, C_mat, W_mat, S_mat, depth_range=[0,1400], depth_gt=None, mask_depth=None):
    device = pixels.device
    batch_size, num_pixels, _ = pixels.shape

    depth_rand = depth_range[0] + (depth_range[1]-depth_range[0]) * torch.rand(batch_size, num_pixels).to(device) 
    
    p_world = image_to_world(pixels, torch.ones(batch_size, num_pixels, 1).to(device), 
                                C_mat, W_mat, S_mat)
    cam_o = image_to_world(pixels, torch.zeros(batch_size, num_pixels, 1).to(device), 
                            C_mat, W_mat, S_mat)
    dir = p_world - cam_o
    
    _, d_intersect, mask = intersect_with_cube(cam_o, dir)
    cube_rand = d_intersect[:,:, 0] + (d_intersect[:,:,1] - d_intersect[:,:,0]) * torch.rand(num_pixels).to(device)
    depth_rand[mask] = cube_rand[mask]
    
    if(depth_gt!=None and mask_depth!=None):
        depth_rand[mask_depth] = depth_gt[mask_depth]
    
    p_rand = image_to_world(pixels, depth_rand.unsqueeze(-1), C_mat, W_mat, S_mat)
    
    return p_rand

def intersect_with_cube(cam_o, dir):
    batch_size, num_pixels, _ = cam_o.shape
    device = cam_o.device
    
    # d = - <n, cam_o - plain> / <n, dir>
    cube_normals = torch.cat([torch.eye(3), -torch.eye(3)])\
        .repeat(batch_size, num_pixels, 1, 1).to(device)
    cube_points = cube_normals * 0.5
    delt = cube_points - cam_o.unsqueeze(2).repeat(1,1,6,1)
    top = (cube_normals * delt).sum(-1)
    bottom = (cube_normals * dir.unsqueeze(2).repeat(1,1,6,1)).sum(-1)
    d = top / bottom
    
    p_plain = cam_o.unsqueeze(2) + d.unsqueeze(-1) * dir.unsqueeze(2) # B X N X 6 X 3
    mask1 = p_plain>=-0.5
    mask2 = p_plain<=0.5
    mask_p_intersect = ((mask1 & mask2).sum(-1) == 3)
    mask_p_valid = mask_p_intersect.sum(-1) == 2
    
    p_intersects = torch.zeros(batch_size, num_pixels, 2, 3).to(device)
    p_intersects[mask_p_valid] = p_plain[mask_p_valid][mask_p_intersect[mask_p_valid]].view(-1, 2, 3)
    
    # d = norm(cam_o - p_intersects)
    d_intersects = torch.zeros(batch_size, num_pixels, 2).to(device)
    dir_norm = torch.norm(dir[mask_p_valid], dim=-1).view(-1,1).repeat(1,2)
    d_intersects[mask_p_valid] = torch.norm((p_intersects[mask_p_valid] - \
        cam_o[mask_p_valid].unsqueeze(1).repeat(1,2,1)), dim=-1)/ dir_norm
    
    d_intersects, index_sorted = d_intersects.sort()
    p_intersects = p_intersects[torch.arange(batch_size).view(-1, 1, 1),
        torch.arange(num_pixels).view(1, -1, 1), index_sorted]
    
    return p_intersects, d_intersects, mask_p_valid


def image_to_world(pixels, depths, C_mat, W_mat, S_mat):
    device = pixels.device
    
    pixels = torch.cat([pixels, torch.ones_like(pixels).to(device)], dim=2)
    pixels[:, :,:3] = pixels[:, :,:3] * depths
    
    C_inv = torch.inverse(C_mat)
    W_inv = torch.inverse(W_mat)
    S_inv = torch.inverse(S_mat)
    
    pixels = S_inv @ W_inv @ C_inv @ pixels.permute(0, 2, 1)
    pixels = pixels[:,:3].permute(0, 2, 1)
    return pixels


def world_to_cam(points, C_mat, W_mat, S_mat):
    device = points.device
    points = torch.cat([points, torch.ones_like(points).to(device)], dim=2)
    points = C_mat @ W_mat @ S_mat @ points.permute(0, 2, 1)
    points = points[:,:3].permute(0, 2, 1)
    return points

def get_value_at_pixel(image , p):
    value = F.grid_sample(image, p.unsqueeze(1), mode='nearest')
    value = value.squeeze(2)
    return value.permute(0, 2, 1)