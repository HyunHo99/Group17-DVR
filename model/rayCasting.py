import random
import torch
import numpy as np
import torch.nn as nn
from util_func import intersect_with_cube


class Module_RayCasting(nn.Module):
    def __init__(self, num_secant_steps=8,
                 depth_range=[0., 1400.], decode_in_once=5700000,
                 steps_schedule=[50000, 100000, 250000],
                 init_steps=16, default_steps=128):
        super().__init__()
        self.num_secant_steps = num_secant_steps
        self.depth_range = depth_range
        self.decode_in_once = decode_in_once
        self.steps_schedule = steps_schedule
        self.init_steps = init_steps
        self.default_steps = default_steps
        self.fn_casting = Diff_RayCasting.apply

    def forward(self, cam_o, dir, decoder, iter=None, steps=None):
        if steps==None or steps<=0:
            if iter==None:
                steps = self.default_steps
            else:
                schedule_step = len([i for i in self.steps_schedule if i<iter])
                steps = self.init_steps * (2 ** schedule_step)
        input = [cam_o, dir, steps, self.num_secant_steps, self.depth_range, 
                 decoder, self.decode_in_once] + list(decoder.parameters())
        d_hat = self.fn_casting(*input)
            
        return d_hat


class Diff_RayCasting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *input):
        (cam_o, dir, steps, num_secant_steps, depth_range, decoder, decode_in_once) = input[:7]
        with torch.no_grad():
            d_hat, p_hat, mask = Diff_RayCasting.ray_casting(cam_o, dir, steps, num_secant_steps, depth_range, decoder, decode_in_once)
    
        ctx.decoder = decoder
        ctx.save_for_backward(p_hat, dir, mask)
        
        return d_hat

    @staticmethod
    def backward(ctx, grad_output):
        p_hat, w, mask = ctx.saved_tensors
        decoder = ctx.decoder
        
        if mask.sum() == 0 : 
            out = [None] * (7 + len(list(decoder.parameters())))
            return tuple(out)
        
        with torch.enable_grad():
            p_hat.requires_grad = True
            f_p = decoder(p_hat)
            df_dp = torch.autograd.grad(f_p.sum(), p_hat, retain_graph=True)[0]
            df_dp_dot_w = (df_dp * w).sum(-1)
            
            df_dp_dot_w[torch.sign(df_dp_dot_w) == 0] = 1e-3
            df_dp_dot_w = torch.sign(df_dp_dot_w) * torch.clamp(torch.abs(df_dp_dot_w), min = 1e-3)
            mu = -(grad_output.squeeze(-1)/ df_dp_dot_w)
            mu[mask == 0] = 0
            
            mu_df_dtheta =  torch.autograd.grad(f_p, decoder.parameters(), grad_outputs=mu, retain_graph=True)
            
        return tuple([None]*7 + list(mu_df_dtheta))

    @staticmethod
    def secant(x1, f1, x2, f2, steps,
                   cam_o, dir, decoder):
        x_n = (x1*f2 - x2*f1) / (f2-f1)
        for _ in range(steps):
            with torch.no_grad():
                p_n = cam_o + x_n.unsqueeze(-1) * dir
                p_n = p_n.unsqueeze(0)
                f_n = decoder(p_n).squeeze(0)
                
            #find the index that outside of occupancy
            lower = f_n < 0
            higher = (lower==0)
            
            x1[lower] = x_n[lower]
            x2[higher] = x_n[higher]
            f1[lower] = f_n[lower]
            f2[higher] = f_n[higher]
            
            #update x_n
            x_n = (x1*f2 - x2*f1) / (f2-f1)
            
        return x_n

    @staticmethod
    def ray_casting(cam_o, dir, steps,
                    num_secant_steps, depth_range, decoder, decode_in_once):
        batch_size, num_pixels, _ = cam_o.shape
        device = cam_o.device
        steps += random.randint(0,1)
        
        d_hat = torch.zeros(batch_size, num_pixels).to(device)
        p_hat = torch.zeros(batch_size, num_pixels, 3).to(device)
        
        _, d_intersects, mask_p_valid = intersect_with_cube(cam_o, dir)
                    
        s0 = d_intersects[:, :, 0].unsqueeze(-1)
        j = torch.linspace(0, 1, steps=steps).to(device).view(1, 1, -1)
        ds = (d_intersects[:, :, 1] - d_intersects[:, :, 0]).unsqueeze(-1)
        # d = r(j * ds + s0)
        d_cube = (j * ds + s0).unsqueeze(-1)
            
        d_steps = torch.linspace(depth_range[0], depth_range[1], steps=steps)\
            .view(1, 1, steps, 1).repeat(batch_size,num_pixels,1,1).to(device)
        d_steps[mask_p_valid] = d_cube[mask_p_valid]
        
        # p = origin + x*dir
        points = cam_o.unsqueeze(2).repeat(1, 1, steps, 1) + \
            d_steps * dir.unsqueeze(2).repeat(1, 1, steps, 1) 
        points = points.view(batch_size, -1, 3)
        
        with torch.no_grad():
            out = []
            for i in torch.split(points, int(decode_in_once/batch_size), dim=1):
                out.append(decoder(i))
            out = torch.cat(out, dim=1).view(batch_size, num_pixels, steps)
            
        # the point is valid when
        # 1. first point not in object.
        # 2. sign is change in step i, i+1
        # 3. the change should be - to +.
        diff = torch.cat([torch.diff(torch.sign(out), dim=-1), torch.full((batch_size, num_pixels, 1),2).to(device)], dim=-1)
        _, index = diff.max(-1)
        mask_first = out[:, :, 0] < 0
        mask = (index != (steps-1)) & mask_first
        
        if(mask.sum()==0):
            d_hat[mask==0] = np.inf
            return d_hat, p_hat, mask
        
        index = index.view(-1)
        k = index.shape[0]
        x1 = d_steps.view(k,steps)[torch.arange(k), index].view(batch_size, num_pixels)[mask]
        f1 = out.view(k,steps)[torch.arange(k), index].view(batch_size, num_pixels)[mask]
        index += 1
        index = torch.clamp(index, max=steps-1)
        x2 = d_steps.view(k,steps)[torch.arange(k), index].view(batch_size, num_pixels)[mask]
        f2 = out.view(k,steps)[torch.arange(k), index].view(batch_size, num_pixels)[mask]
        
        d_hat_mask = Diff_RayCasting.secant(x1, f1, x2, f2, num_secant_steps, cam_o[mask], dir[mask], decoder)
        
        p_hat[mask] = cam_o[mask] + d_hat_mask.unsqueeze(-1) * dir[mask]
        
        d_hat[mask] = d_hat_mask
        d_hat[mask==0] = np.inf
        
        
        return d_hat, p_hat, mask
