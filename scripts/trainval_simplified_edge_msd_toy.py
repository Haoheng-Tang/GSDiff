import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle
from scipy.stats import truncnorm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.msd_room_semantics import MSDRoomSemantics
from gsdiff.house_nn3 import EdgeModel

'''Edge prediction trainer — MSD wall-graph toy run.

Mirrors trainval_simplified_edge_unconstrained.py (the RPLAN edge trainer)
but uses the MSD wall-graph toy dataset (MAX_NODES=80, 9 room types).

The edge model is NOT a diffusion model — single forward pass, CE loss on
per-cell adjacency. Trained against GT corners + small Gaussian noise (so it
later tolerates the noisy stage-1 (node-diffusion) outputs at val time).

Memory note: at MAX_NODES=80 the attention operates over N²=6400 tokens.
Each attention layer materializes a (bs, heads, 6400, 6400) score tensor.
Default batch_size=1 keeps this around ~650 MB per layer in fp32; with 12
layers and backprop, expect peaks ~10+ GB. Drop batch further or enable
torch.amp.autocast if you hit OOM.
'''

NUM_ROOM_TYPES = 9
MAX_NODES = 80
TOY_DATA_ROOT = 'msd-v1-wallgraph-toy'

lr = 1e-4
weight_decay = 1e-5

total_steps = 20   # toy budget
batch_size = 1       # keep low: edge attn cost is O(N^4)
device = 'cuda:0'

output_dir = 'outputs/msd-edge-toy/'
os.makedirs(output_dir, exist_ok=True)

# Data — same toy split as the node trainer
dataset_train = MSDRoomSemantics('train', permute=True, data_root=TOY_DATA_ROOT,
                                 max_nodes=MAX_NODES, return_edges=True)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
dataloader_train_iter = iter(cycle(dataloader_train))

dataset_val = MSDRoomSemantics('test', permute=True, data_root=TOY_DATA_ROOT,
                               max_nodes=MAX_NODES, return_edges=True)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                            num_workers=0, drop_last=False, pin_memory=True)
dataloader_val_iter = iter(cycle(dataloader_val))

# Model
model = EdgeModel(num_room_types=NUM_ROOM_TYPES).to(device)
print('Edge model param count:', sum(p.numel() for p in model.parameters()))

optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)


def truncated_normal(tensor, mu, sigma, lower, upper, dtype, device):
    with torch.no_grad():
        size = tensor.shape
        tmp = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma,
                            loc=mu, scale=sigma, size=size)
        tmp = torch.as_tensor(tmp, dtype=dtype, device=device)
        tensor.copy_(tmp)
    return tensor


step = 0
loss_curve = []
val_metrics = []
interval = 20

while step < total_steps:
    corners_withsemantics, global_attn_matrix, corners_padding_mask, edges = next(dataloader_train_iter)
    corners_withsemantics = corners_withsemantics.to(device).clamp(-1, 1)
    global_attn_matrix = global_attn_matrix.to(device).bool()
    corners_padding_mask = corners_padding_mask.to(device)
    edges = edges.to(device)

    corners = corners_withsemantics[:, :, :2]
    semantics = corners_withsemantics[:, :, 2:]

    # Add small truncated Gaussian noise to corners (stage-1 output simulation)
    sigma = 1 / 128
    corners_noise = truncated_normal(
        torch.empty((batch_size, MAX_NODES, 2), dtype=corners.dtype, device=device),
        0, sigma, -3 * sigma, 3 * sigma, dtype=corners.dtype, device=device)
    corners = corners + corners_noise

    # edges target: (bs, N^2, 2) one-hot
    edges_onehot = torch.cat((1 - edges, edges), dim=2).to(torch.uint8)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    output_edges = model(corners, global_attn_matrix, corners_padding_mask, semantics)  # (bs, N^2, 2)

    edges_target = edges_onehot.reshape(*output_edges.shape)
    output_edges_flat = output_edges.reshape(-1, 2)
    edges_target_flat = edges_target[:, :, 1].reshape(-1).long()

    edges_CELoss = torch.nn.CrossEntropyLoss(reduction='none')(output_edges_flat, edges_target_flat)
    edges_pad_mask_flat = global_attn_matrix.reshape(-1).to(torch.uint8)
    loss = (edges_CELoss * edges_pad_mask_flat).sum() / edges_pad_mask_flat.sum().clamp(min=1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    step += 1

    print(f'step {step}  edge_CE*1e5={loss.item()*1e5:.2f}')
    loss_curve.append([loss.item() * 1e5])

    if step % interval == 0:
        torch.save(model.state_dict(), output_dir + f'model{step:07d}.pt')
        torch.save(optimizer.state_dict(), output_dir + f'optim{step:07d}.pt')
        np.save(output_dir + 'loss_curve.npy', np.array(loss_curve))

    if step % interval == 0:
        model.eval()
        TP = TN = FP = FN = 0
        for _ in tqdm(range(len(dataset_val)), desc='val'):
            cw, attn_v, pm_v, e_v = next(dataloader_val_iter)
            cw = cw.to(device).clamp(-1, 1)
            attn_v = attn_v.to(device).bool()
            pm_v = pm_v.to(device)
            e_v = e_v.to(device).to(torch.uint8)
            corners_v = cw[:, :, :2] + (torch.randn_like(cw[:, :, :2]) * 1 / 128)
            sem_v = cw[:, :, 2:]
            with torch.no_grad():
                out_v = model(corners_v, attn_v, pm_v, sem_v)
                out_v = torch.argmax(F.softmax(out_v, dim=2), dim=2)
            tgt = e_v.reshape(*out_v.shape)
            valid = attn_v.reshape(*out_v.shape)
            TP += int(((out_v == 1) & (tgt == 1) & valid).sum())
            TN += int(((out_v == 0) & (tgt == 0) & valid).sum())
            FP += int(((out_v == 1) & (tgt == 0) & valid).sum())
            FN += int(((out_v == 0) & (tgt == 1) & valid).sum())
        acc = (TP + TN) / max(TP + TN + FP + FN, 1)
        prec = TP / max(TP + FP, 1)
        rec = TP / max(TP + FN, 1)
        print(f'step {step}  acc={acc:.4f}  precision={prec:.4f}  recall={rec:.4f}')
        val_metrics.append([step, acc, prec, rec])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))
        model.train()
