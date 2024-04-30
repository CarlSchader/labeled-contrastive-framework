import time, argparse, sys, os, torch, multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from labeled_contrastive_framework.transform import make_eval_transform
from labeled_contrastive_framework.module import * 
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import pytorch_lightning as L
from transformers import AutoImageProcessor, MobileViTV2Model, MobileViTV2Config

class StudentEncoder(nn.Module):
        def __init__(self, out_dim):
            super(StudentEncoder, self).__init__()
            configuration = MobileViTV2Config()
            backbone_out_dim = 512
            configuration.return_dict = False
            self.backbone = MobileViTV2Model(configuration)
            self.head = nn.Sequential( # add an embedding head
                nn.Linear(backbone_out_dim, out_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, out_dim),
            )

        def forward(self, x):
            x = self.backbone(x)[1]
            x = self.head(x)
            return x

def number_of_parameters(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True, type=str, help='Path to the checkpoint')
    parser.add_argument('-d', '--dataset_path', required=True, type=str, help='Path to the image dataset')
    parser.add_argument('-a', '--class_averages', type=str, required=True, help='Path to the class averages')
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
    class_averages, class_average_labels = torch.load(args.class_averages)
    print('class_averages:', class_averages.shape)
    print('class_average_labels:', len(class_average_labels))
    print('num_workers:', dataloader_workers)

    train_set_size = int(len(dataset) * 0.98)
    valid_set_size = len(dataset) - train_set_size

    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers)

    # correct = 0
    # total = 0
    # batches = 0
    # print('dataset classes', len(dataset.classes))
    # for batch, targets in train_loader:
    #     for target in targets:
    #         # print(dataset.classes[target], class_average_labels[target])
    #         try:
    #             if dataset.classes[target] == class_average_labels[target]:
    #                 correct += 1
    #             else:
    #                 print(f'Error: {dataset.classes[target]} != {class_average_labels[target]} target is {target}')
    #                 exit()
    #         except:
    #             print(f'Error: {dataset.classes[target]} target is {target}')
    #             exit()
    #         total += 1
    #     batches += 1
    #     print(f'Processed {batches} batches of {len(train_loader)} batches. correct {correct} total {total}', end='\r')
    # print(f'accuracy: {correct / total} correct: {correct} total: {total}')
    # exit()

    print('loading teacher')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    transform = make_eval_transform()

    teacher_model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)
    
    teacher_encoder = teacher_model.encoder
    print(f'teacher encoder has {number_of_parameters(teacher_encoder)} parameters')

    print('initializing student')
    student_encoder = StudentEncoder(embedding_dim)
    print(f'student encoder has {number_of_parameters(student_encoder)} parameters')

    # # testing the student student_encoder
    # from PIL import Image
    # test_input = Image.open('/home/carl/Desktop/monster_reborn.jpg')
    # test_input = make_eval_transform()(test_input).unsqueeze(0)
    # 
    # total_params = sum(p.numel() for p in student_encoder.parameters())
    # print(f'encoder has {total_params} parameters')
    # 
    # test_output = student_encoder(test_input)
    # print(test_output.shape)
    #
    # exit()

    lightning_module = LabeledContrastiveDistillationModule(teacher_encoder, student_encoder, class_averages, embedding_dim=embedding_dim)

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
    
    

    

