import torch, os, sys, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from module import *

parser = argparse.ArgumentParser(description='Export a PyTorch model to TorchScript') 
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('-o', '--output', type=str, default='model.pt')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

encoder = model.encoder
encoder = encoder.to(device)
encoder.eval()


traced_encoder = torch.jit.trace(encoder, torch.rand(1, 3, 224, 224).to(device))

torch.jit.save(traced_encoder, args.output)

