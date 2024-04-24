import torch, os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transform import make_transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from module import *

parser = argparse.ArgumentParser(description='Resume training from a checkpoint. WARNING: PLEASE CREATE CHECKPOINT BACKUPS FIRST') 
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('-d', '--dataset_path', type=str, required=True)
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-e', '--new_epochs', type=int, default=100)
args = parser.parse_args()

chkpt = torch.load(args.checkpoint)
last_lr = chkpt['lr_schedulers'][0]['_last_lr'][0]

print('loading dataset')
dataset = ImageFolder(args.dataset_path, make_transform())

dataloader_workers = max((multiprocessing.cpu_count() // 2) - 1, 0)
print('num_workers:', dataloader_workers)

train_set_size = int(len(dataset) * 0.98)
valid_set_size = len(dataset) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=dataloader_workers)
valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=dataloader_workers)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_last=True,
    every_n_epochs=1,
    save_top_k=1,
    filename='arcface-{epoch:02d}-{val_loss:.2f}',
)

lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = L.Trainer(max_epochs=(args.new_epochs + int(chkpt['epoch']) + 1), callbacks=[checkpoint_callback, lr_monitor])

trainer.fit(model, train_loader, valid_loader, ckpt_path=args.checkpoint)

