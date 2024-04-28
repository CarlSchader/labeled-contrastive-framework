# Attempting to implement ArcFace without needing to compare 
# the embeddings against the class centers across all classes,
# but instead just against the class centers within the batch.

import torch, multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torch import optim
from fiblat import sphere_lattice


# cross_entropy = nn.CrossEntropyLoss()

# arcface loss function (centers are pre-normalized on a sphere of radius 1)
def arcface_loss(embeddings, targets, centers, m=0.5, s=64.0):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    cos_sims = torch.mm(normalized_embeddings, centers.t())
    angles = torch.acos(cos_sims)
    angles = angles + m # add margin
    margin_distances = s*torch.cos(angles)
    return F.cross_entropy(margin_distances, targets)
    # return cross_entropy(margin_distances, targets)

class SphereNormalization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return torch.nn.functional.normalize(X, p=2, dim=1)

class LabeledContrastiveEncoder(L.LightningModule):
    def __init__(
            self, 
            backbone, 
            backbone_out_dim,
            num_classes, 
            loss_fn=arcface_loss, 
            embedding_dim=128, 
            margin=0.5, 
            scale=64.0,
            learning_rate=1e-4,
            weight_decay=0.01,
        ):
        super().__init__()
        self.encoder = nn.Sequential( # add an embedding head
            backbone,
            nn.Linear(backbone_out_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.loss_fn = loss_fn
        self.margin = margin
        self.scale = scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # initialize class centers on a sphere
        self.centers = torch.tensor(sphere_lattice(embedding_dim, num_classes), dtype=torch.float32)
        
        self.save_hyperparameters(ignore=['backbone'])

    def on_fit_start(self):
        self.centers = self.centers.to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        norms = F.normalize(z, p=2, dim=1)
        loss = self.loss_fn(norms, y, self.centers, self.margin, self.scale)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        norms = F.normalize(z, p=2, dim=1)
        loss = self.loss_fn(norms, y, self.centers, self.margin, self.scale)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        #     optim.lr_scheduler.LinearLR(optimizer, 0.33, 1.0, total_iters=5),
        #     optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5),
        # ],
        # milestones=[1])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                "monitor": "val_loss",
            }
        }

import sys
sys.path.append('/home/carl/programming/knowledge-distillation-framework/')
from knowledge_distillation_framework import KnowledgeDistillationModule

class LabeledContrastiveDistillationModule(KnowledgeDistillationModule):
    '''
    A PyTorch Lightning module for knowledge distillation.
    Based on the paper "Improving Knowledge Distillation via Regularizing Feature 
    Norm and Direction."

    The student encoder and teacher encoder should output the same feature dimension.
    '''
    def __init__(
            self, 
            teacher_encoder,
            student_encoder,
            teacher_centers,
            class_centers, # class_centers could be the fibonnaci centers of the teacher or the class averages. Either way, they should be L2 normalized.
            learning_rate=1e-4,
            weight_decay=0.01,
        ):
        super().__init__(
            teacher_encoder=teacher_encoder,
            student_encoder=student_encoder,
            task_loss_fn=arcface_loss,
            teacher_head=teacher_head,
            student_head=student_head,
            kd_loss_fn=kd_loss_fn,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        self.centers = class_centers

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            zt = self.teacher_encoder(x)
            qt = self.teacher_head(zt)
        zs = self.student_encoder(x)
        qs = self.student_head(zs)
        loss = self.task_loss_fn(zs, zt, qs, qt, y) + self.kd_loss_fn(zs, zt, qs, qt, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            zt = self.teacher_encoder(x)
            qt = self.teacher_head(zt)
        zs = self.student_encoder(x)
        qs = self.student_head(zs)
        loss = self.task_loss_fn(zs, zt, qs, qt, y) + self.kd_loss_fn(zs, zt, qs, qt, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
        #     optim.lr_scheduler.LinearLR(optimizer, 0.33, 1.0, total_iters=5),
        #     optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5),
        # ],
        # milestones=[1])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                "monitor": "val_loss",
            }
        }


if __name__ == '__main__':
    import time, argparse
    from torchvision import datasets
    from transform import make_transform
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

    # from transformers import Dinov2Model

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Path to the image dataset')
    args = parser.parse_args()

    start = time.time()
    
    print('loading dataset')
    dataset = datasets.ImageFolder(args.dataset_path, make_transform())

    # get the number of classes
    num_classes = len(dataset.classes)
    embedding_dim = 128
    backbone_out_dim = 384 
    batch_size = 128
    epochs = 100
    dataloader_workers = max((multiprocessing.cpu_count() // 2) - 1, 0)
    print('num_workers:', dataloader_workers)

    train_set_size = int(len(dataset) * 0.98)
    valid_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    print('loading backbone')
    # backbone = Dinov2Model.from_pretrained("facebook/dinov2-base").base_model
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

    print('initializing lightining module')
    lightning_module = LabeledContrastiveEncoder(backbone, backbone_out_dim, num_classes, embedding_dim=embedding_dim)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs=1,
        save_top_k=1,
        filename='arcface-{epoch:02d}-{val_loss:.2f}',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    print('initializing trainer')
    trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback])

    print('training')
    trainer.fit(lightning_module, train_loader, valid_loader)

    print('done')
    print(time.time() - start)
    
    

    

