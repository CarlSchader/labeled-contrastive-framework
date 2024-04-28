import torch, argparse, sys, os
from torchvision.datasets import DatasetFolder, ImageFolder
from torch.utils.data import DataLoader
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'labeled_contrastive_framework'))

from module import *

def margin_distances(a, b, m=0.5, s=64.0):
    cos_sims = torch.mm(a, b.t())
    angles = torch.acos(cos_sims)
    angles = angles + m # add margin
    return s*torch.cos(angles)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Given an evaluation dataset folder of embeddings, classify them using euclidean distance against another dataset folder of embeddings.')
parser.add_argument('-c', '--checkpoint', type=str, required=True)
parser.add_argument('-e' ,'--embedding_eval_set_path', required=True, type=str, help='Path to the folder of embeddings')
parser.add_argument('-d', '--original_dataset_path', required=True, type=str, help='Path to the original dataset folder')
parser.add_argument('-t', '--test-batch-size', type=int, default=32, help='Batch size for the test set')
parser.add_argument('-m', '--metric', type=str, default='euclidean', help='Distance metric to use')
parser.add_argument('--angular_margin', type=float, default=0., help='Angular margin for the margin distance function')
args = parser.parse_args()

test_set_path = args.embedding_eval_set_path
test_batch_size = args.test_batch_size
metric = args.metric

backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

model = LabeledContrastiveEncoder.load_from_checkpoint(args.checkpoint, backbone=backbone)

centers = model.centers.to(device)

def cos_sim(a, b):
    anorm = a / a.norm(dim=1).unsqueeze(1)
    bnorm = b / b.norm(dim=1).unsqueeze(1)
    return 1 - torch.mm(anorm, bnorm.T)

distance_function = None
if metric == 'euclidean':
    distance_function = torch.cdist
elif metric == 'cosine':
    distance_function = cos_sim
elif metric == 'softmax':
    print('Using softmax distance function')
    print('Angular margin: ', args.angular_margin)
    distance_function = lambda a, b: -torch.nn.functional.softmax(margin_distances(a, b, m=args.angular_margin), dim=-1)
else:
    raise ValueError('Invalid distance metric, valid options are euclidean and cosine')

dataset = ImageFolder(args.original_dataset_path)
test_set = DatasetFolder(test_set_path, loader=lambda x: torch.load(x), extensions=('.pt', '.pth'))

testloader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

correct = 0
total = 0
test_batch_count = 0
test_batch_total = len(testloader)

for _, (test_batch, test_targets) in enumerate(testloader):
    test_batch = test_batch.to(device)

    distances = distance_function(test_batch, centers)
    min_distances, min_idxs = torch.min(distances, dim=1)

    min_classes = [dataset.classes[min_idxs[i]] for i in range(len(distances))]
    test_classes = [test_set.classes[test_targets[i]] for i in range(len(distances))]
    correct_vec = torch.tensor([1 if min_classes[i] == test_classes[i] else 0 for i in range(len(min_classes))])
    batch_correct = torch.sum(correct_vec).item()
    batch_total = len(test_targets)
    
    correct += batch_correct
    total += batch_total

    test_batch_count += 1

    print("Test Batches: ", test_batch_count, "/", test_batch_total)
    print("Batch accuracy: ", batch_correct / batch_total)
    print("Total accuracy: ", correct / total)
