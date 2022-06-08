import torch
import torch.optim as optim
from torch import autograd
from utils.libmise import MISE
from utils import libmcubes
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import trimesh


class Mesh_Generator(object):
    def __init__(self, model, device=None, 
                 decode_in_once=100000, refine_batch_size=10000,
                 grid_resolution=16, 
                 num_upsample=4, num_refine=0):
        self.model = model.to(device)
        self.decode_in_once = decode_in_once
        self.device = device
        self.grid_resolution = grid_resolution
        self.num_refine = num_refine
        self.num_upsample = num_upsample
        self.refine_batch_size = refine_batch_size
        
    def generate(self):
        # we borrowed this part from occupancy network
        self.model.eval()
        
        mesh_extractor = MISE(self.grid_resolution, self.num_upsample, 0)
        points = mesh_extractor.query()

        while points.shape[0] != 0:
            pointsf = torch.FloatTensor(points).to(self.device)
            pointsf = pointsf / mesh_extractor.resolution
            pointsf = pointsf - 0.5
            
            values = self.point_to_occupancy(pointsf).cpu().numpy()
            values = values.astype(np.float64)
            mesh_extractor.update(points, values)
            points = mesh_extractor.query()

        value_grid = mesh_extractor.to_dense()
            
        mesh = self.get_mesh(value_grid)
                
        return mesh

    def point_to_occupancy(self, p):
        p_occs = []
        for pi in torch.split(p, self.decode_in_once):
            pi = pi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                p_occ = self.model.get_prob(pi).logits
            p_occs.append(p_occ.squeeze(0).detach().cpu())

        p_occ = torch.cat(p_occs, dim=0)

        return p_occ
    
    def get_mesh(self, p_occ):
        # we borrowed this part from occupancy network
        w, h, z = p_occ.shape
        vert, tri = libmcubes.marching_cubes(np.pad(p_occ, 1, 'constant', constant_values = -1e6), 0)
        vert -= 0.5
        vert -= 1
        
        vert /= np.array([w-1, h-1, z-1])
        vert = vert - 0.5
        
        mesh = trimesh.Trimesh(vert, tri, process=None)
        
        if vert.shape[0] == 0:
            return mesh
                
        if self.num_refine > 0:
            self.refinement(mesh)
            
        vert_colors = self.get_vert_color(np.array(mesh.vertices))
        mesh = trimesh.Trimesh(
            vertices = mesh.vertices, faces = mesh.faces,
            vertex_colors=vert_colors, process = False)
        
        return mesh
        
    def get_vert_color(self, vert):
        vert = torch.FloatTensor(vert).to(self.device)
        colors = []
        for vi in torch.split(vert, self.decode_in_once):
            with torch.no_grad():
                color = self.model.decoder(vi.unsqueeze(0), texture=True).squeeze(0).cpu()
            colors.append(color)
        colors = np.concatenate(colors, axis=0)
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
        colors = np.concatenate([colors, np.full((colors.shape[0], 1), 255, dtype=np.uint8)], axis=1)
        
        return colors
    
    
    def refinement(self, mesh):               
        vertices = torch.from_numpy(mesh.vertices).float().to(self.device)
        v_param = torch.nn.Parameter(vertices.clone())

        faces = torch.from_numpy(mesh.faces).long().to(self.device)
        optimizer = optim.RMSprop([v_param], lr=1e-5)

        dataset = TensorDataset(faces)
        dataloader = DataLoader(dataset, batch_size=self.refine_batch_size,
                                shuffle=True)
        
        self.model.eval()
        iter = 0
        for face in dataloader:
            if(iter>=self.num_refine):
                break
            face = face[0]
            optimizer.zero_grad()

            p_face = v_param[face]
            
            perterb = np.random.dirichlet((0.5, 0.5, 0.5), size=face.shape[0])
            perterb = torch.from_numpy(perterb).float().to(self.device)
            p_face = (p_face * perterb[:, :, None]).sum(dim=1)
            
            p_ocu = []
            for split in torch.split(p_face.unsqueeze(0), 20000, dim=1):
                p_ocu.append(torch.sigmoid(self.model.get_prob(split).logits))
            p_ocu = torch.cat(p_ocu, dim=1)
            
            loss = (p_ocu - 0.5).pow(2).mean()
            
            loss.backward()
            optimizer.step()
            iter+=1

        mesh.vertices = v_param.data.cpu().numpy()
