import torch
from torch.nn import functional as F

qs = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
qt = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
y = torch.tensor([0, 2, 2], dtype=torch.long)
centers = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

kd_loss = (qs - qt).norm(p=2, dim=1).mean()
print('norm diff')
print((qs - qt).norm(p=2, dim=1))

mse_loss = F.mse_loss(qs, qt)
print('kd_loss:')
print(kd_loss)
print('mse_loss:')
print(mse_loss)
print()


class_counts = torch.bincount(y)
qs_norm = qs.norm(p=2, dim=1)
qt_norm = qt.norm(p=2, dim=1)
max_norms = torch.stack([qs_norm, qt_norm]).max(dim=0)[0]
nd_loss = (qs * centers[y]).sum(dim=1)
nd_loss = nd_loss / max_norms
nd_loss = nd_loss / class_counts[y]
nd_loss = nd_loss.sum()
nd_loss = nd_loss / torch.count_nonzero(class_counts)
print('nd_loss:')
print(nd_loss)
print()


