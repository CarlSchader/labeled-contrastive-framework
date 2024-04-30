import argparse, torch, multiprocessing
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', required=True, type=str, help='Path to the image dataset')
parser.add_argument('-b', '--batch_size', type=int, default=128)
parser.add_argument('-o', '--output_path', type=str, default='embeddings.pt')
args = parser.parse_args()

dataset = DatasetFolder(args.dataset_path, loader=torch.load, extensions=('.pt', '.pth'))
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=max(multiprocessing.cpu_count() - 1, 0), shuffle=False)

print(len(dataset.classes))

embeddings = None
labels = []
count = 0
for batch, targets in dataloader:
    if embeddings is None:
        embeddings = batch.clone()
    else:
        embeddings = torch.cat((embeddings, batch))

    for target in targets:
        labels.append(dataset.classes[target])
    count += 1
    print(f'Processed {count} batches of {len(dataloader)} batches', end='\r')


with open(args.output_path, 'wb') as f:
    torch.save((embeddings, labels), f)

