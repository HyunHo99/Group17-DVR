import torch
import os
import argparse
import numpy as np
import yaml
from pathlib import Path

from model.dvr import DVR
from model.decoder import Decoder
from generator import Mesh_Generator
from dvr_trainer import DVR_trainer
import glob


parser = argparse.ArgumentParser()
parser.add_argument('cfg', type=str)

args = parser.parse_args()
with open(args.cfg, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

out_dir = cfg['trainer']['out_dir']

device = torch.device("cuda")
decoder = Decoder()
dvr = DVR(decoder, device = device, depth_range=cfg['trainer']['depth_range'])

generator = Mesh_Generator(dvr, device=device, **cfg['generator'])
model = DVR_trainer(model=dvr, generator=generator, **cfg['trainer'])
best_path = glob.glob(out_dir+"/best/*.ckpt")
model = model.load_from_checkpoint(checkpoint_path=best_path[0], 
                                   model = dvr, generator=generator, **cfg['trainer'])

generator = Mesh_Generator(model.model, device=device, **cfg['generator'])

torch.manual_seed(0)

generate_dir = out_dir + "/generate"
scan_name = Path(cfg["dataset"]["dir_data"]).stem
if not os.path.exists(generate_dir):
    os.makedirs(generate_dir)

cam_dict = np.load(cfg["dataset"]["dir_data"]+ '/cameras.npz')
scale_mat = cam_dict['scale_mat_0'].astype(np.float32)

out = generator.generate()
vertices = np.asarray(out.vertices).astype(np.float32)
vert_homo = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=-1).transpose(1,0)
vert_homo = scale_mat @ vert_homo
out.vertices = vert_homo[:3].transpose(1, 0)

file_name = os.path.join(generate_dir, '%s_%s_result.ply' % (scan_name, cfg['dataset']['depth_type']))
out.export(file_name)

print("done")