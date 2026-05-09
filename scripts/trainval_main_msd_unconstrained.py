import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from itertools import cycle
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.msd_room_semantics import MSDRoomSemantics
from gsdiff.house_nn1 import HeterHouseModel
from gsdiff.house_nn3 import EdgeModel
from gsdiff.utils import edges_remove_padding, inverse_normalize_and_remove_padding_100
from scripts.metrics.fid import fid
from scripts.metrics.kid import kid

'''Stage 1 node generation training — MSD unconstrained'''

NUM_ROOM_TYPES = 9   # MSD has 9 room type codes 0-8
# After appending (1 - padding_mask): feature width = 2 coords + 9 types + 1 pad = 12

diffusion_steps = 1000
lr = 1e-4
weight_decay = 0

total_steps = 1000000
# total_steps = 100  # quick smoke-test

batch_size = 4
batch_size_val = 4

device = 'cuda:0'
merge_points = False
clamp_trick_training = True

# MSD room-type color palette (BGR for OpenCV), types 0-8
MSD_COLORS = {
    0: (200, 220, 240),  # generic room / bedroom
    1: (180, 230, 180),  # living room
    2: (200, 180, 230),  # room / corridor
    3: (180, 230, 230),  # dining
    4: (200, 200, 160),  # bathroom / corridor
    5: (160, 200, 230),  # balcony / storeroom
    6: (180, 180, 180),  # stairs / shaft
    7: (230, 210, 180),  # corridor / kitchen
    8: (210, 180, 210),  # kitchen / balcony
    9: (0,   0,   0),    # background / wall
}


# ---------------------------------------------------------------------------
# helpers identical to trainval_main_unconstrained.py
# ---------------------------------------------------------------------------

def map_to_binary(tensor):
    batch_size, n_values = tensor.shape
    binary_tensor = torch.zeros((batch_size, n_values, 12), dtype=torch.float32, device=tensor.device)
    mask = tensor != 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part
    for i in range(8):
        binary_tensor[:, :, 7 - i] = integer_part % 2
        integer_part //= 2
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(4):
        binary_tensor[:, :, 11 - i] = fractional_part % 2
        fractional_part //= 2
    binary_tensor = torch.where(mask.unsqueeze(-1), binary_tensor, torch.zeros_like(binary_tensor))
    return binary_tensor

def map_to_fournary(tensor):
    batch_size, n_values = tensor.shape
    fournary_tensor = torch.zeros((batch_size, n_values, 6), dtype=torch.float32, device=tensor.device)
    mask = tensor != 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part
    for i in range(4):
        fournary_tensor[:, :, 3 - i] = integer_part % 4
        integer_part //= 4
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        fournary_tensor[:, :, 5 - i] = fractional_part % 4
        fractional_part //= 4
    fournary_tensor = torch.where(mask.unsqueeze(-1), fournary_tensor, torch.zeros_like(fournary_tensor))
    return fournary_tensor

def map_to_eightnary(tensor):
    batch_size, n_values = tensor.shape
    eightnary_tensor = torch.zeros((batch_size, n_values, 5), dtype=torch.float32, device=tensor.device)
    mask = tensor != 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part
    for i in range(3):
        eightnary_tensor[:, :, 2 - i] = integer_part % 8
        integer_part //= 8
    fractional_part *= 64
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(2):
        eightnary_tensor[:, :, 4 - i] = fractional_part % 8
        fractional_part //= 8
    eightnary_tensor = torch.where(mask.unsqueeze(-1), eightnary_tensor, torch.zeros_like(eightnary_tensor))
    return eightnary_tensor

def map_to_sxtnary(tensor):
    batch_size, n_values = tensor.shape
    sxtnary_tensor = torch.zeros((batch_size, n_values, 3), dtype=torch.float32, device=tensor.device)
    mask = tensor != 99999
    valid_values = torch.where(mask, tensor, torch.zeros_like(tensor))
    integer_part = valid_values.floor().to(torch.int32)
    fractional_part = valid_values - integer_part
    for i in range(2):
        sxtnary_tensor[:, :, 1 - i] = integer_part % 16
        integer_part //= 16
    fractional_part *= 16
    fractional_part = fractional_part.floor().to(torch.int32)
    for i in range(1):
        sxtnary_tensor[:, :, 2 - i] = fractional_part % 16
        fractional_part //= 16
    sxtnary_tensor = torch.where(mask.unsqueeze(-1), sxtnary_tensor, torch.zeros_like(sxtnary_tensor))
    return sxtnary_tensor


def render_centroid_graph(corners_px, semantics_int, edges_flat, n_nodes, img_size=256):
    """Render a bubble-diagram: colored circles at centroids, gray lines for edges.

    corners_px : (n_nodes, 2) int  pixel coords in [0, img_size]
    semantics_int : (n_nodes, NUM_ROOM_TYPES) int  one-hot or soft predictions
    edges_flat    : (n_nodes*n_nodes,) binary
    """
    img = np.ones((img_size, img_size, 3), np.uint8) * 255
    # draw edges
    adj = edges_flat.reshape(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if adj[i, j]:
                p1 = (int(corners_px[i, 0]), int(corners_px[i, 1]))
                p2 = (int(corners_px[j, 0]), int(corners_px[j, 1]))
                cv2.line(img, p1, p2, color=(180, 180, 180), thickness=2)
    # draw room circles
    for i in range(n_nodes):
        rt = int(np.argmax(semantics_int[i])) if semantics_int[i].sum() > 0 else NUM_ROOM_TYPES
        color = MSD_COLORS.get(rt, (128, 128, 128))
        cx, cy = int(corners_px[i, 0]), int(corners_px[i, 1])
        cv2.circle(img, (cx, cy), radius=8, color=color, thickness=-1)
        cv2.circle(img, (cx, cy), radius=8, color=(100, 100, 100), thickness=1)
    return img


# ---------------------------------------------------------------------------
# output directory
# ---------------------------------------------------------------------------
output_dir = 'outputs/msd-stage1-unconstrained/'
os.makedirs(output_dir, exist_ok=True)
open(output_dir + 'file_description.txt', 'w').close()

# ---------------------------------------------------------------------------
# diffusion schedule (cosine beta)
# ---------------------------------------------------------------------------
alpha_bar = lambda t: math.cos(t / 1.000 * math.pi / 2) ** 2
betas = []
max_beta = 0.999
for i in range(diffusion_steps):
    t1 = i / diffusion_steps
    t2 = (i + 1) / diffusion_steps
    betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
betas = np.array(betas, dtype=np.float64)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)
sqrt_recip_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod)
sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / alphas_cumprod - 1)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)

# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
dataset_train = MSDRoomSemantics('train', permute=True)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=True)
dataloader_train_iter = iter(cycle(dataloader_train))

dataset_val = MSDRoomSemantics('val', permute=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size_val, shuffle=False,
                            num_workers=0, drop_last=False, pin_memory=True)
dataloader_val_iter = iter(cycle(dataloader_val))

# GT rendering: no permutation so corners and edges stay consistent
dataset_val_gt = MSDRoomSemantics('val', permute=False)
dataloader_val_gt = DataLoader(dataset_val_gt, batch_size=batch_size_val, shuffle=False,
                               num_workers=0, drop_last=False, pin_memory=True)
dataloader_val_gt_iter = iter(cycle(dataloader_val_gt))

# ---------------------------------------------------------------------------
# render GT val images
# ---------------------------------------------------------------------------
gt_dir_val = output_dir + 'val_gt/'
if os.path.exists(gt_dir_val):
    shutil.rmtree(gt_dir_val)
os.makedirs(gt_dir_val)

n_val_batches = (len(dataset_val_gt) + batch_size_val - 1) // batch_size_val
for batch_count in tqdm(range(n_val_batches), desc='rendering GT'):
    cs_batch, attn_batch, mask_batch, edges_batch = next(dataloader_val_gt_iter)
    for i in range(cs_batch.shape[0]):
        val_idx = batch_count * batch_size_val + i
        cs = cs_batch[i].cpu().numpy()       # (53, 11)
        mask = mask_batch[i].cpu().numpy()   # (53, 1)
        edges = edges_batch[i].cpu().numpy() # (2809, 1)
        n = int(mask.sum())
        if n == 0:
            cv2.imwrite(os.path.join(gt_dir_val, f'val_gt_{val_idx}.png'),
                        np.ones((256, 256, 3), np.uint8) * 255)
            continue
        coords_norm = cs[:n, :2]  # (n, 2) in [-1, 1]
        coords_px = (coords_norm * 118 + 128).astype(int).clip(0, 255)  # [10, 246]
        semantics = cs[:n, 2:].astype(int)  # (n, 9)
        adj_flat = edges.reshape(53, 53)[:n, :n].reshape(-1)
        img = render_centroid_graph(coords_px, semantics, adj_flat, n)
        cv2.imwrite(os.path.join(gt_dir_val, f'val_gt_{val_idx}.png'), img)

# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
model = HeterHouseModel(num_room_types=NUM_ROOM_TYPES).to(device)
print('Stage 1 param count:', sum(p.numel() for p in model.parameters()))
optimizer = AdamW(list(model.parameters()), lr=lr, weight_decay=weight_decay)

# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
step = 0
loss_curve = []
val_metrics = []

interval = 500
# interval = 250000  # full training

while step < total_steps:
    cs0, global_attn, mask = next(dataloader_train_iter)
    # cs0: (bs, 53, 11) — 2 coords + 9 room types
    cs0 = cs0.to(device).clamp(-1, 1)
    global_attn = global_attn.to(device)
    mask = mask.to(device)

    # append padding indicator → (bs, 53, 12)
    cs0 = torch.cat((cs0, (1 - mask).type(cs0.dtype)), dim=2)

    for param in model.parameters():
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()

    t_dist = np.ones([diffusion_steps]) / diffusion_steps
    t = torch.tensor(np.random.choice(len(t_dist), size=(batch_size,), p=t_dist),
                     dtype=torch.long, device=device)

    noise = torch.randn_like(cs0)
    cs_t = (torch.tensor(sqrt_one_minus_alphas_cumprod, device=device)[t][:, None, None].expand_as(noise) * noise
            + torch.tensor(sqrt_alphas_cumprod, device=device)[t][:, None, None].expand_as(cs0) * cs0)

    out1, out2 = model(cs_t, global_attn, t)
    out = torch.cat((out1, out2), dim=2)   # (bs, 53, 12)

    # noise prediction loss (L2)
    corners_loss = ((noise - out) ** 2).sum(dim=[1, 2]) / 53
    corners_loss_batch = corners_loss.mean()

    # geometric alignment loss (same as RPLAN)
    model_variance = torch.tensor(posterior_variance, device=device)[t][:, None, None].expand_as(cs_t)
    pred_xstart = (torch.tensor(sqrt_recip_alphas_cumprod, device=device)[t][:, None, None].expand_as(cs_t) * cs_t
                   - torch.tensor(sqrt_recipm1_alphas_cumprod, device=device)[t][:, None, None].expand_as(cs_t) * out)

    if clamp_trick_training:
        pred_coord = torch.clamp(pred_xstart[:, :, 0:2], min=-1, max=1)
        pred_seman = pred_xstart[:, :, 2:] >= 0.5
        pred_cs = torch.cat((pred_coord, pred_seman), dim=2)
    else:
        pred_cs = pred_xstart

    time_weight = torch.tensor(betas.tolist()[::-1], dtype=torch.float64, device=device)
    pad_flag = pred_cs[:, :, -1]  # 1 = padding node
    mask_valid = pad_flag == 0

    x_coords = pred_cs[:, :, 0] * mask_valid
    y_coords = pred_cs[:, :, 1] * mask_valid
    inf_mask = torch.where(mask_valid, 0, float('inf'))
    x_coords = x_coords + inf_mask
    y_coords = y_coords + inf_mask

    x_uns = x_coords.unsqueeze(2)
    y_uns = y_coords.unsqueeze(2)
    dist_x = torch.abs(x_uns - x_uns.transpose(1, 2))
    dist_y = torch.abs(y_uns - y_uns.transpose(1, 2))
    for dm in (dist_x, dist_y):
        dm[torch.isinf(dm) | torch.isnan(dm)] = 99999
    eye_mask = (torch.eye(53).unsqueeze(0) == 1).to(device).expand(batch_size, 53, 53)
    dist_x[eye_mask] = 99999
    dist_y[eye_mask] = 99999

    min_x, _ = dist_x.min(dim=2)
    min_y, _ = dist_y.min(dim=2)
    min_d = torch.stack((min_x, min_y), dim=2).min(dim=2)[0]
    min_d_unnorm = torch.where(min_d != 99999, min_d * 128,
                               torch.tensor(0.0, dtype=min_d.dtype, device=device))

    bin_m   = map_to_binary(min_d_unnorm)
    four_m  = map_to_fournary(min_d_unnorm)
    eight_m = map_to_eightnary(min_d_unnorm)
    sxt_m   = map_to_sxtnary(min_d_unnorm)

    t_w = time_weight[t]
    masked_bin   = ((-12 * torch.log(1 - (1/12  - 1e-8) * bin_m.sum(2)).sum(1)   * t_w).sum() / 128) / batch_size
    masked_four  = ((-18 * torch.log(1 - (1/18  - 1e-8) * four_m.sum(2)).sum(1)  * t_w).sum() / 128) / batch_size
    masked_eight = ((-35 * torch.log(1 - (1/35  - 1e-8) * eight_m.sum(2)).sum(1) * t_w).sum() / 128) / batch_size
    masked_sxt   = ((-45 * torch.log(1 - (1/45  - 1e-8) * sxt_m.sum(2)).sum(1)   * t_w).sum() / 128) / batch_size
    masked_inf   = (torch.where(min_d != 99999,
                                -2 * torch.log(1 - (0.5 - 1e-8) * min_d),
                                torch.tensor(0.0, dtype=min_d.dtype, device=device)
                                ).sum(1) * t_w).sum() / batch_size
    align_loss = masked_bin + masked_four + masked_eight + masked_sxt + masked_inf

    loss = corners_loss_batch + align_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
    optimizer.step()
    step += 1

    print(f'step {step}  loss*1e5={loss.item()*1e5:.2f}  '
          f'noise*1e5={corners_loss_batch.item()*1e5:.2f}  '
          f'align*1e5={align_loss.item()*1e5:.2f}')
    loss_curve.append([loss.item()*1e5, corners_loss_batch.item()*1e5, align_loss.item()*1e5])

    if step % interval == 0:
        torch.save(model.state_dict(), output_dir + f'model{step:07d}.pt')
        torch.save(optimizer.state_dict(), output_dir + f'optim{step:07d}.pt')
        np.save(output_dir + 'loss_curve.npy', np.array(loss_curve))

    if step % interval == 0:
        model.eval()

        # ---- stage 1 denoising on val set ----
        res_corners = []
        res_semantics = []
        res_numbers = []

        n_val_batches_eval = (len(dataset_val) + batch_size_val - 1) // batch_size_val
        for _ in tqdm(range(n_val_batches_eval), desc='val denoising'):
            cs0_val, attn_val, mask_val, _ = next(dataloader_val_iter)
            cs0_val = cs0_val.to(device).clamp(-1, 1)
            attn_val = attn_val.to(device)
            mask_val = mask_val.to(device)
            cs0_val = torch.cat((cs0_val, (1 - mask_val).type(cs0_val.dtype)), dim=2)

            cs_t_val = torch.randn(*cs0_val.shape, device=device, dtype=cs0_val.dtype)

            for cur_step in range(diffusion_steps - 1, -1, -1):
                t_val = torch.tensor([cur_step], device=device)
                with torch.no_grad():
                    o1, o2 = model(cs_t_val, attn_val, t_val)
                    o_val = torch.cat((o1, o2), dim=2)

                    pv = torch.tensor(posterior_variance, device=device)[t_val][:, None, None].expand_as(cs_t_val)
                    px = (torch.tensor(sqrt_recip_alphas_cumprod, device=device)[t_val][:, None, None].expand_as(cs_t_val) * cs_t_val
                          - torch.tensor(sqrt_recipm1_alphas_cumprod, device=device)[t_val][:, None, None].expand_as(cs_t_val) * o_val)
                    px[:, :, 0:2] = torch.clamp(px[:, :, 0:2], min=-1, max=1)
                    px[:, :, 2:] = px[:, :, 2:] >= 0.5

                    mean = (torch.tensor(posterior_mean_coef1, device=device)[t_val][:, None, None].expand_as(cs_t_val) * px
                            + torch.tensor(posterior_mean_coef2, device=device)[t_val][:, None, None].expand_as(cs_t_val) * cs_t_val)
                    cs_t_val = mean + torch.sqrt(pv) * torch.randn_like(cs_t_val)

            for i in range(cs0_val.shape[0]):
                res_corners.append(cs_t_val[i, :, :2][None])
                res_semantics.append(cs_t_val[i, :, 2:2+NUM_ROOM_TYPES][None])
                res_numbers.append(cs_t_val[i, :, 2+NUM_ROOM_TYPES:2+NUM_ROOM_TYPES+1].view(-1))

        corners_denorm, semantics_denorm = inverse_normalize_and_remove_padding_100(
            res_corners, res_semantics, res_numbers)

        # ---- render predicted val images ----
        pred_dir_val = output_dir + f'val_{step:07d}/'
        if os.path.exists(pred_dir_val):
            shutil.rmtree(pred_dir_val)
        os.makedirs(pred_dir_val)

        for vi in tqdm(range(len(dataset_val)), desc='rendering pred'):
            c_i = corners_denorm[vi]  # (1, n, 2) uint8 pixels in [0, 255]
            s_i = semantics_denorm[vi]  # (1, n, 9) int8
            n_i = c_i.shape[1]
            if n_i == 0:
                cv2.imwrite(os.path.join(pred_dir_val, f'val_pred_{vi}.png'),
                            np.ones((256, 256, 3), np.uint8) * 255)
                continue
            img = np.ones((256, 256, 3), np.uint8) * 255
            for j in range(n_i):
                rt = int(np.argmax(s_i[0, j])) if s_i[0, j].sum() > 0 else NUM_ROOM_TYPES
                color = MSD_COLORS.get(rt, (128, 128, 128))
                cx, cy = int(c_i[0, j, 0]), int(c_i[0, j, 1])
                cv2.circle(img, (cx, cy), radius=8, color=color, thickness=-1)
                cv2.circle(img, (cx, cy), radius=8, color=(100, 100, 100), thickness=1)
            cv2.imwrite(os.path.join(pred_dir_val, f'val_pred_{vi}.png'), img)

        current_fid = fid(gt_dir_val, pred_dir_val, fid_batch_size=128, fid_device=device)
        current_kid = kid(gt_dir_val, pred_dir_val, kid_batch_size=128, kid_device=device)
        print(f'step {step}  FID={current_fid:.2f}  KID={current_kid:.6f}')
        val_metrics.append([step, current_fid, current_kid])
        np.save(output_dir + 'val_metrics.npy', np.array(val_metrics))

        model.train()

        if step in [500000]:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.1
