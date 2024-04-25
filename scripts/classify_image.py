import torch, argparse, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'labeled_contrastive_framework'))
from PIL import Image
from module import *
from transform import make_eval_transform
from torchvision.datasets import ImageFolder, DatasetFolder
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Classify image with respect to a folder of embeddings')
parser.add_argument('-i', '--image_path', required=True, type=str, help='Path to the image to classify')
parser.add_argument('-d', '--dataset', required=True, type=str, help='Path to the dataset image foler')
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to the model checkpoint')
parser.add_argument('-s', '--sphere', action='store_true', help='Normalize the embeddings onto the unit sphere')
args = parser.parse_args()

image_path = args.image_path
dataset_path = args.dataset

dataset = DatasetFolder(dataset_path , extensions=['.pth'], loader=lambda x: torch.load(x))
dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

centers = None
labels = []
for embeddings, targets in dataloader:
    if centers is None:
        centers = embeddings
    else:
        centers = torch.cat([centers, embeddings], dim=0)

    for target in targets:
        labels.append(dataset.classes[target])

# dataset = ImageFolder(dataset_path , transform=make_eval_transform())
# centers = model.centers

centers = centers.to(device)
print(centers.shape)

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
transform = make_eval_transform()

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

encoder = model.encoder 
# normalize the embeddings onto the unit sphere
if args.sphere:
    print('sphere normalization')
    encoder = torch.nn.Sequential(encoder, SphereNormalization())
encoder = encoder.to(device)
encoder.eval()

with torch.no_grad():
    image = Image.open(image_path)
    image = transform(image).to(device)
    print(image.shape)
    image_embedding = encoder(image.unsqueeze(0))
    i = 0
    if args.sphere:
        distances = 1 - torch.mm(image_embedding, centers.t())
    else:
        distances = torch.cdist(image_embedding, centers)
    print(distances.shape)
    min_distance, min_index = torch.min(distances, dim=-1)
    print(min_distance.item(), min_index.item())
    print(labels[min_index.item()])
