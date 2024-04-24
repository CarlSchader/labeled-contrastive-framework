import torch, argparse, os, sys, multiprocessing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from PIL import Image
from module import *
from transform import make_eval_transform
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Classify image with respect to a folder of embeddings')
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('-d', '--dataset', type=str, required=True, help='Path to the dataset')
args = parser.parse_args()

image_path = args.image_path
embedding_folder = args.embedding_folder

dataset = ImageFolder(embedding_folder, transform=make_eval_transform())
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=multiprocessing.cpu_count()-1)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
transform = make_eval_transform()

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

encoder = model.encoder 
encoder = torch.nn.Sequential(encoder, SphereNormalization())
encoder = encoder.to(device)
encoder.eval()

centers = model.centers.to(device)
print(centers.shape)

with torch.no_grad():
    for i, (batch, targets) in enumerate(dataloader):
        labels = dataset.classes[targets]
        batch = batch.to(device)
        features = encoder(batch)
        print(features.shape)
        break
