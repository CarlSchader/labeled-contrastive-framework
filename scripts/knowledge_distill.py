import time, argparse, sys, os, torch, multiprocessing, torchvision
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from labeled_contrastive_framework.transform import make_transform 
from labeled_contrastive_framework.module import * 
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as L
from transformers import  MobileViTV2Model, MobileViTV2Config

class StudentEncoder(nn.Module):
        def __init__(self, out_dim):
            super(StudentEncoder, self).__init__()
            # configuration = MobileViTV2Config()
            # configuration.return_dict = False
            # self.backbone = MobileViTV2Model(configuration)
            # backbone_out_dim = 512

            self.backbone = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)
            backbone_out_dim = 1000


            # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            # backbone_out_dim = 384

            self.head = nn.Sequential( # add an embedding head
                nn.Linear(backbone_out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, out_dim),
            )

        def forward(self, x):
            # x = self.backbone(x)[1]
            x = self.backbone(x)
            x = self.head(x)
            return x

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True, type=str, help='Path to the checkpoint')
    parser.add_argument('-d', '--dataset_path', required=True, type=str, help='Path to the image dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--task_loss_weight', type=float, default=1.0)
    parser.add_argument('--kd_loss_weight', type=float, default=1.0)
    parser.add_argument('--nd_loss_weight', type=float, default=1.0)
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()

    start = time.time()
    
    print('loading dataset')
    training_transform = make_transform()
    dataset = datasets.ImageFolder(args.dataset_path, transform=training_transform)

    # get the number of classes
    num_classes = len(dataset.classes)
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size
    epochs = args.epochs
    dataloader_workers = max((multiprocessing.cpu_count() // 2) - 1, 0)
    print('num_workers:', dataloader_workers)

    train_set_size = int(len(dataset) * 0.98)
    valid_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    print('loading teacher')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    teacher_module = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)
    teacher_encoder = teacher_module.encoder
    centers = teacher_module.centers

    print('initializing student')
    student_encoder = StudentEncoder(embedding_dim)

    lightning_module = LabeledContrastiveDistillationModule(
        teacher_encoder, 
        student_encoder, 
        centers,
        task_loss_weight=args.task_loss_weight,
        kd_loss_weight=args.kd_loss_weight,
        nd_loss_weight=args.nd_loss_weight,
        learning_rate=args.learning_rate,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        filename='distilled-{epoch:02d}-{val_loss:.2f}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    print('initializing trainer')
    trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback, lr_monitor])

    print('training')
    trainer.fit(lightning_module, train_loader, valid_loader)

    print('done')
    print(time.time() - start)
    
    

    

