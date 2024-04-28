from torchvision.datasets import ImageFolder
import argparse, json

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_path', type=str, required=True)
parser.add_argument('-o', '--output_path', type=str, default='classes.json')
args = parser.parse_args()

dataset = ImageFolder(args.dataset_path)

classes = dataset.classes

with open(args.output_path, 'w') as f:
    f.write(json.dumps(classes))

