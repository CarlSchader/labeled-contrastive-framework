import time, argparse, sys, os, torch, multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from labeled_contrastive_framework.transform import make_eval_transform
from labeled_contrastive_framework.module import LabeledContrastiveEncoder, LabeledContrastiveDistillationModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as L
from transformers import AutoImageProcessor, MobileViTV2Model, MobileViTV2Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True, type=str, help='Path to the checkpoint')
    parser.add_argument('-d', '--dataset_path', required=True, type=str, help='Path to the image dataset')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--embedding_dim', type=int, default=128)
    args = parser.parse_args()

    start = time.time()
    
    print('loading dataset')
    dataset = datasets.ImageFolder(args.dataset_path, make_eval_transform())

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

    print('loading backbone')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    transform = make_eval_transform()

    teacher_model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

    teacher_encoder = teacher_model.encoder

    student_configuration = MobileViTV2Config()
    student_configuration.return_dict = False
    student_encoder = MobileViTV2Model(config=student_configuration)

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
    
    

    

