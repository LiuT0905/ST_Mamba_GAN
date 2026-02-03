# generate_error_maps_only_diff_random.py
import os
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import traceback
import random  # <--- 新增：引入随机库

# 引入你的模型组件
from sam_unet.unet_parts import OutConv
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from sam_unet.layers import CBAM
from mamba_ssm import Mamba

# ---------------- 配置参数 ----------------
FOLDER_PATH = r"D:\dataset\output_2019_huabei"
MASK_PATH = r"D:\mamba_GAN\code\mask\mask_huabei.txt"
CHECKPOINT_PATH = r'./experiment/ST_Mamba_GAN_12/nets/netG_epoch_300.pth'
OUT_DIR = './experiment/ST_Mamba_GAN_12/error_maps_diff'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMAGE_SIZE = 64
SEQUENCE_LENGTH = 12
NUM_RANDOM_BATCHES = 5  # <--- 新增：设置你想处理的随机 Batch 数量

# 反归一化参数
L_ELE = 0.0
H_ELE = 990.0

NUM_WORKERS = 0


# ----------------- 绘图核心函数 -----------------
def save_batch_mosaic_with_colorbar(arr, out_path, nrow=8, cmap='seismic',
                                    vmin=None, vmax=None, cb_label='', dpi=300,
                                    tile_size=1.0):
    """
    保存带有颜色条的马赛克拼图，专用于绘制 Diff Error Map
    """
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    else:
        a = np.array(arr)

    # 确保维度正确 (B, H, W)
    if a.ndim == 4 and a.shape[1] == 1:
        a = a[:, 0, :, :]

    B, H, W = a.shape
    ncol = nrow
    nrow_calc = math.ceil(B / ncol)

    # 自动确定范围，如果未指定
    if vmin is None: vmin = float(np.nanmin(a))
    if vmax is None: vmax = float(np.nanmax(a))
    if vmin == vmax: vmax = vmin + 1e-6

    # 计算画布尺寸
    fig_w = ncol * tile_size + 1.2  # 留出Colorbar空间
    fig_h = nrow_calc * tile_size

    fig = plt.figure(figsize=(fig_w, fig_h))
    # 最后一列留给 Colorbar
    gs = fig.add_gridspec(nrow_calc, ncol + 1, width_ratios=[1] * ncol + [0.08],
                          wspace=0.01, hspace=0.01)

    im = None
    idx = 0
    for r in range(nrow_calc):
        for c in range(ncol):
            ax = fig.add_subplot(gs[r, c])
            ax.set_xticks([])
            ax.set_yticks([])

            if idx < B:
                img = a[idx]
                im = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
            else:
                # 空白位置填充
                ax.imshow(np.full((H, W), 0), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
            idx += 1

    # 绘制 Colorbar
    cax = fig.add_subplot(gs[:, -1])
    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        if cb_label:
            cbar.set_label(cb_label, rotation=270, labelpad=12)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------- Dataset 定义 (保持不变) -----------------
class ResizeTransform:
    def __init__(self, size): self.size = int(size)

    def __call__(self, img): return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img): return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                                   f.lower().endswith(('.jpg', '.png', '.tif'))])
        if not isinstance(mask, torch.Tensor): mask = torch.tensor(mask, dtype=torch.float32)
        self.mask = mask.unsqueeze(0) if mask.dim() == 2 else mask
        self.transform = transform

    def __len__(self):
        return len(self.image_files) - self.num_images_per_sequence + 1

    def __getitem__(self, idx):
        seq_files = self.image_files[idx:idx + self.num_images_per_sequence]
        imgs = []
        for p in seq_files:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im.ndim == 3: im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32) / float(H_ELE)
            if self.transform: im = self.transform(im)
            imgs.append(im)
        imgs = torch.stack(imgs, dim=0)
        return imgs, (imgs[-1].clone() * self.mask), seq_files


transform = transforms.Compose([ResizeTransform(IMAGE_SIZE), ToTensor16Bit(), transforms.Normalize((0.5,), (0.5,))])
mask = torch.tensor(np.loadtxt(MASK_PATH, dtype=np.float32), dtype=torch.float32)
dataset = CombinedDataset(FOLDER_PATH, SEQUENCE_LENGTH, mask, transform, IMAGE_SIZE)


def my_collate(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masked = torch.stack([item[1] for item in batch], dim=0)
    seqs = [item[2] for item in batch]
    return imgs, masked, seqs


dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=NUM_WORKERS,
                        collate_fn=my_collate)


# ----------------- Model 定义 (保持不变) -----------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SmaAt_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16, sequence_length=4,
                 image_size=64, mamba_d_model=1024, mamba_d_state=16, mamba_d_conv=4, mamba_expand=2):
        super().__init__()
        self.sequence_length = sequence_length
        self.inc = DoubleConvDS(n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)

        self.up1 = UpDS(256 + 512, 128, bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(256, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(128, 64, bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, n_classes)

        self.pre_mamba_conv = nn.Conv2d(512, 16 * sequence_length, 1)
        self.post_mamba_conv = nn.Conv2d(16 * sequence_length, 512, 1)
        self._bottleneck_D = (16 * sequence_length * (image_size // 8) ** 2) // sequence_length

        self.mamba_proj_in = nn.Linear(self._bottleneck_D, mamba_d_model)
        self.mamba_proj_out = nn.Linear(mamba_d_model, self._bottleneck_D)
        self.mamba_layernorm_in = RMSNorm(mamba_d_model)
        self.mamba = Mamba(d_model=mamba_d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x, cp):
        x1 = self.inc(torch.cat((x, cp), dim=1))
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.pre_mamba_conv(F.dropout2d(self.down3(x3), 0.3, self.training))

        B, C, H, W = x4.shape
        x4_seq = x4.view(B, self.sequence_length, -1)
        x_proj = self.mamba_layernorm_in(self.mamba_proj_in(x4_seq))
        mamba_out = x_proj + self.mamba(x_proj)
        x4_enhanced = self.post_mamba_conv(self.mamba_proj_out(mamba_out).view(B, C, H, W))

        x = self.up1(x4_enhanced, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)
        return self.outc(x)


class Generator(nn.Module):
    def __init__(self, input_channels=12, output_channels=1):
        super().__init__()
        self.sma_at_unet = SmaAt_UNet(n_channels=input_channels, n_classes=output_channels,
                                      sequence_length=input_channels)

    def forward(self, x, cp): return self.sma_at_unet(x, cp)


# ----------------- 加载模型 -----------------
netG = Generator()
netG.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
netG.to(DEVICE).eval()


def denormalize_array(x_norm):
    return L_ELE + (np.array(x_norm) / 2.0 + 0.5) * (H_ELE - L_ELE)


# ----------------- 主循环：只保存随机的几个 Batch 的 DIFF MOSAIC -----------------
print(f"Starting processing. Output directory: {OUT_DIR}")

# 1. 计算总 Batch 数并生成随机索引
total_batches = len(dataloader)
num_targets = min(NUM_RANDOM_BATCHES, total_batches)
target_indices = set(random.sample(range(total_batches), num_targets))
print(f"Total batches: {total_batches}. Selected {num_targets} random batches: {sorted(list(target_indices))}")

for batch_idx, batch in enumerate(dataloader):
    # 2. 检查当前 batch 是否在目标列表中
    if batch_idx not in target_indices:
        continue  # 跳过不处理

    try:
        images, masked_last_image, _ = batch

        # 3. 模型推理
        dems = images[:, -1, :, :, :].to(DEVICE)
        pm = images[:, 0:11, :, :, :].squeeze(2).to(DEVICE)
        cpLayer = masked_last_image.squeeze(2).to(DEVICE)

        with torch.no_grad():
            fake = netG(pm, cpLayer)

        # 4. 数据处理与反归一化
        fake_den = denormalize_array(fake.detach().cpu().numpy()).squeeze(1)  # (B, H, W)
        gt_den = denormalize_array(dems.detach().cpu().numpy()).squeeze(1)  # (B, H, W)

        # 5. 计算误差 (Diff)
        diff_arr = fake_den - gt_den

        # 6. 设置保存路径
        diff_mosaic_path = os.path.join(OUT_DIR, f'batch_{batch_idx + 1}_diff_mosaic.png')

        # 7. 计算对称的 Colorbar 范围 (使0为白色/中间色)
        max_abs_val = float(np.max(np.abs(diff_arr)))
        if max_abs_val < 1e-6: max_abs_val = 1e-6

        # 8. 保存图片
        save_batch_mosaic_with_colorbar(
            diff_arr,
            diff_mosaic_path,
            nrow=8,
            cmap='seismic',  # 红蓝分明，中间白
            vmin=-max_abs_val,  # 对称范围
            vmax=max_abs_val,
            cb_label='Error (generated-real)',
            tile_size=0.9
        )

    except Exception:
        print(f"Error processing batch {batch_idx + 1}")
        traceback.print_exc()

print("Selected random batches processed. Only diff mosaics were saved.")