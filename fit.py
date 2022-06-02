from dvr_trainer import DVR_trainer
import yaml
import os 
import argparse
from dataset import DTUdataset
from model.dvr import DVR
from model.decoder import Decoder
from generator import Mesh_Generator
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

parser = argparse.ArgumentParser()
parser.add_argument('cfg', type=str)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

out_dir = cfg['trainer']['out_dir']    

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

batch_size = cfg['train']['batch_size']
batch_size_val = cfg['train']['batch_size_val']

train_dataset = DTUdataset(**cfg['dataset'])
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

val_dataset = DTUdataset(**cfg['dataset'])
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size_val, num_workers=0)

decoder = Decoder()

cfg['generator']['num_upsample'] = 2
cfg['generator']['num_refine'] = 0

device = torch.device("cuda")
dvr = DVR(decoder, device = device, depth_range=cfg['trainer']['depth_range'])
generator = Mesh_Generator(dvr, device=device, **cfg['generator'])
model = DVR_trainer(model=dvr, generator=generator, **cfg['trainer'])
    
checkpoint_callback = ModelCheckpoint(dirpath=out_dir+"/best", save_top_k=1, mode='max', monitor="val/mask_intersection")
logger = TensorBoardLogger(out_dir+"/log", name="my_model")

trainer = Trainer(gpus=1, max_epochs=cfg['train']['max_epochs'], check_val_every_n_epoch=cfg['train']['check_val_every_n_epoch'],
                  default_root_dir=out_dir, callbacks=checkpoint_callback,
                  flush_logs_every_n_steps=20, log_every_n_steps=10, logger=logger)

trainer.fit(model, train_loader, val_loader)