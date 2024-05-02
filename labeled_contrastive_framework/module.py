# Attempting to implement ArcFace without needing to compare 
# the embeddings against the class centers across all classes,
# but instead just against the class centers within the batch.

import torch, multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as L
from torch import optim
from fiblat import sphere_lattice
from knowledge_distillation_framework import KnowledgeDistillationModule

def arcface_loss(embeddings, targets, centers, m=0.5, s=64.0):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    cos_sims = torch.mm(normalized_embeddings, centers.t())
    angles = torch.acos(cos_sims)
    angles = angles + m # add margin
    margin_distances = s*torch.cos(angles)
    return F.cross_entropy(margin_distances, targets)

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
        self.loss_fn = arcface_loss
        self.margin = margin
        self.scale = scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # initialize class centers on a sphere
        self.centers = torch.tensor(sphere_lattice(embedding_dim, num_classes), dtype=torch.float32, requires_grad=False)
        
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
            centers, # fibonnaci centers
            task_loss_weight=1.0,
            kd_loss_weight=2.0,
            nd_loss_weight=4.0,
            margin=0.5,
            scale=64.0,
            learning_rate=1e-4,
            weight_decay=0.01,
        ):
        
        self.centers = centers
        self.task_loss_weight = task_loss_weight
        self.kd_loss_weight = kd_loss_weight
        self.nd_loss_weight = nd_loss_weight
        self.margin = margin
        self.scale = scale
    
        def combined_kd_loss(zs, zt, qs, qt, y):
            kd_loss = (qs - qt).norm(p=2, dim=1).mean()

            class_counts = torch.bincount(y)
            qs_norm = qs.norm(p=2, dim=1)
            qt_norm = qt.norm(p=2, dim=1)
            max_norms = torch.stack([qs_norm, qt_norm]).max(dim=0)[0]
            nd_loss = (qs * self.centers[y]).sum(dim=1)
            nd_loss = nd_loss / max_norms
            nd_loss = nd_loss / class_counts[y]
            nd_loss = nd_loss.sum()
            nd_loss = -nd_loss / torch.count_nonzero(class_counts)
            return self.kd_loss_weight * kd_loss + self.nd_loss_weight * nd_loss

            # return self.kd_loss_weight * kd_loss

        super().__init__(
            teacher_encoder=teacher_encoder,
            student_encoder=student_encoder,
            task_loss_fn=lambda zs, zt, qs, qt, y: self.task_loss_weight*arcface_loss(qs, y, self.centers, self.margin, self.scale),
            teacher_head=nn.Identity(),
            student_head=nn.Identity(),
            kd_loss_fn=combined_kd_loss,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        self.save_hyperparameters(ignore=['teacher_encoder', 'student_encoder'])

    def on_fit_start(self):
        self.centers = self.centers.to(self.device)


if __name__ == '__main__':
    import time, argparse
    from torchvision import datasets
    from transform import make_transform
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

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
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

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
    
    

    

