# import lightning as L
import torch, os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from transform import make_eval_transform
from carlschader_ml_utils.image_utils import embed_image_folder
from module import *

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('-o', '--out_dir', type=str, default='embeddings/last/')
    parser.add_argument('-a', '--average', action='store_true')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-s', '--sphere', action='store_true', help='Normalize the embeddings onto the unit sphere')
    args = parser.parse_args()

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    transform = make_eval_transform()

    model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

    encoder = model.encoder 
    # normalize the embeddings onto the unit sphere
    if args.sphere:
        print('sphere normalization')
        encoder = torch.nn.Sequential(encoder, SphereNormalization())
    encoder.eval()

    embed_image_folder(encoder, args.dataset, args.out_dir, averages_only=args.average, verbose=True, transform=transform, batch_size=args.batch_size)

