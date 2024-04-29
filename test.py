import torch
from torch.nn import functional as F

a = torch.tensor([[1, 2, 3], [9, 5, 6], [7, 8, 9]], dtype=torch.float32)
b = torch.tensor([[0, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
y = torch.tensor([0, 2, 2], dtype=torch.long)
class_unit_averages = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1], [7.1, 8.1, 9.1]], dtype=torch.float32)
class_unit_averages = F.normalize(class_unit_averages, p=2, dim=1)

print('class_unit_averages:')
print(class_unit_averages)
print()

class_counts = torch.bincount(y)
print('class_counts:')
print(class_counts)
print()

an = a.norm(p=2, dim=1)
bn = b.norm(p=2, dim=1)
print('an:')
print(an)
print('bn:')
print(bn)
print()

max_norms = torch.stack([an, bn]).max(dim=0)[0]
print('max_norms:')
print(max_norms)
print()

nd_loss = (a * class_unit_averages[y]).sum(dim=1)
print('elementwise:')
print(a * class_unit_averages[y])
print()

print('sum:')
print(nd_loss)
print()

nd_loss = nd_loss / max_norms

print('div max')
print(nd_loss)
print()

nd_loss = nd_loss / class_counts[y]

print('class_counts:')
print(class_counts[y])
print()

print('div class_counts:')
print(nd_loss)
print()

nd_loss = nd_loss.sum()
nd_loss = -nd_loss / torch.count_nonzero(class_counts)

print('count_nonzero:')
print(torch.count_nonzero(class_counts))
print()

print('nd_loss:')
print(nd_loss)
print()


