# Attempting to implement ArcFace without needing to compare 
# the embeddings against the class centers across all classes,
# but instead just against the class centers within the batch.

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch import optim
from fiblat import sphere_lattice

cross_entropy = nn.CrossEntropyLoss()

# arcface loss function (centers are pre-normalized on a sphere of radius 1)
def arcface_loss(embeddings, targets, centers, m=0.5, s=64.0):
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
    cos_sims = torch.mm(normalized_embeddings, centers.t())
    angles = torch.acos(cos_sims)
    angles = angles + m # add margin
    margin_distances = s*torch.cos(angles)
    return cross_entropy(margin_distances, targets)

class LabeledContrastiveModule(L.LightningModule):
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
        self.lodd_fn = loss_fn
        self.loss_fn = arcface_loss
        self.margin = margin
        self.scale = scale
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # initialize class centers on a sphere
        self.centers = torch.tensor(sphere_lattice(embedding_dim, num_classes), dtype=torch.float32)

    def on_fit_start(self):
        self.centers = self.centers.to(self.device)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        norms = F.normalize(z, p=2, dim=1)
        return self.loss_fn(norms, y, self.centers, self.margin, self.scale)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        norms = F.normalize(z, p=2, dim=1)
        return self.loss_fn(norms, y, self.centers, self.margin, self.scale)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, 0.33, 1.0, total_iters=5),
            optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5),
        ],
        milestones=[5])
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

if __name__ == '__main__':
    import time, argparse
    from torchvision import datasets
    from transform import make_transform

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

    train_set_size = int(len(dataset) * 0.98)
    valid_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = torch.utils.data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = torch.utils.data.DataLoader(train_set)
    valid_loader = torch.utils.data.DataLoader(valid_set)
    
    

    print('loading backbone')
    # backbone = Dinov2Model.from_pretrained("facebook/dinov2-base").base_model
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

    print('initializing lightining module')
    lightning_module = LabeledContrastiveModule(backbone, backbone_out_dim, num_classes, embedding_dim=embedding_dim)

    print('initializing trainer')
    trainer = L.Trainer(max_epochs=100)

    print('training')
    trainer.fit(lightning_module, train_loader, valid_loader)

    print('done')
    print(time.time() - start)
    
    

    

