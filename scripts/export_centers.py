import argparse, json, sys, os, torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from labeled_contrastive_framework.module import * 

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('-o', '--output_path', type=str, default='centers.pth')
parser.add_argument('-j', '--json_output_path', type=str, default=None)
args = parser.parse_args()

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

centers = model.centers

torch.save(centers, args.output_path)

if args.json_output_path is not None:
    with open(args.json_output_path, 'w') as f:
        f.write(json.dumps(centers.tolist()))

