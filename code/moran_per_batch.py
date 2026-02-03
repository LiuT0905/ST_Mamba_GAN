# generate_morans_i_for_gt_and_fake.py

# 这个脚本计算随机选择的几个batch的真实值图和生成值图的莫兰指数（Moran's I），并绘制折线图。

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
import pandas as pd
import traceback
import random  # 新增导入

# ---------------- CONFIG (modify these) ----------------
FOLDER_PATH = r"D:\dataset\output_2019_huabei"  # folder with image sequence files
MASK_PATH = r"D:\mamba_GAN\code\mask\mask_huabei.txt"
CHECKPOINT_PATH = r'./experiment/ST_Mamba_GAN_12/nets/netG_epoch_300.pth'  # your netG
OUT_DIR = r'./experiment/ST_Mamba_GAN_12/morans_i_full'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMAGE_SIZE = 64
SEQUENCE_LENGTH = 12

# 新增：随机选择的批次数量
RANDOM_BATCH_COUNT = 5  # 随机选择5个批次

# denorm constants (must match training)
L_ELE = 0.0
H_ELE = 990.0

# keep drop_last False so we evaluate all samples
NUM_WORKERS = 0


# -------------------------------------------------------

# ----------------- Moran's I 计算函数 -----------------
def compute_morans_i(image, neighborhood=8):
    """
    计算图像的Moran's I（空间自相关指数）。
    image: 2D numpy array (H, W)
    neighborhood: 4 (rook) 或 8 (queen)，默认8。
    返回: Moran's I 值 (float)
    """
    if image.ndim != 2:
        raise ValueError("Image must be 2D array.")
    H, W = image.shape
    mean_val = np.mean(image)
    dev = image - mean_val
    denom = np.sum(dev ** 2)
    if denom == 0:
        return 0.0  # 避免除零

    num = 0.0
    S0 = 0.0  # 权重总和
    n = H * W

    # 定义邻域偏移
    if neighborhood == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    elif neighborhood == 4:
        offsets = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    else:
        raise ValueError("neighborhood must be 4 or 8.")

    for i in range(H):
        for j in range(W):
            for di, dj in offsets:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W:
                    w_ij = 1.0  # 二元权重
                    num += w_ij * dev[i, j] * dev[ni, nj]
                    S0 += w_ij

    if S0 == 0:
        return 0.0

    I = (n / S0) * (num / denom)
    return float(I)


# ----------------- Dataset -----------------
class ResizeTransform:
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, img):
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1,H,W)


class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path)
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {folder_path}")
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
        if mask.dim() == 2:
            self.mask = mask.unsqueeze(0)  # (1,H,W)
        else:
            self.mask = mask
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return len(self.image_files) - self.num_images_per_sequence + 1

    def __getitem__(self, idx):
        seq_files = self.image_files[idx:idx + self.num_images_per_sequence]
        imgs = []
        for p in seq_files:
            im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise RuntimeError(f"Failed to read {p}")
            if im.ndim == 3:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = im.astype(np.float32) / float(H_ELE)
            if self.transform:
                im = self.transform(im)
            imgs.append(im)
        imgs = torch.stack(imgs, dim=0)  # (T,1,H,W)
        last = imgs[-1].clone()  # (1,H,W)
        masked_last = last * self.mask  # broadcast
        return imgs, masked_last, seq_files


# transforms same as training
transform = transforms.Compose([
    ResizeTransform(IMAGE_SIZE),
    ToTensor16Bit(),
    transforms.Normalize((0.5,), (0.5,))
])

# load mask
mask_arr = np.loadtxt(MASK_PATH, dtype=np.float32)
mask = torch.tensor(mask_arr, dtype=torch.float32)

dataset = CombinedDataset(folder_path=FOLDER_PATH,
                          num_images_per_sequence=SEQUENCE_LENGTH,
                          mask=mask,
                          transform=transform,
                          image_size=IMAGE_SIZE)


def my_collate(batch):
    imgs = [item[0] for item in batch]
    masked = [item[1] for item in batch]
    seqs = [item[2] for item in batch]
    imgs = torch.stack(imgs, dim=0)
    masked = torch.stack(masked, dim=0)
    return imgs, masked, seqs


dataloader = DataLoader(dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,  # 保持shuffle以随机选择批次
                        drop_last=False,
                        num_workers=NUM_WORKERS,
                        collate_fn=my_collate)

# 计算总批次数量
total_batches = len(dataloader)
print(f"总共有 {total_batches} 个批次")

# 随机选择批次索引
if RANDOM_BATCH_COUNT > total_batches:
    print(f"警告：要求随机选择 {RANDOM_BATCH_COUNT} 个批次，但只有 {total_batches} 个批次。将选择所有批次。")
    selected_batch_indices = list(range(total_batches))
else:
    selected_batch_indices = random.sample(range(total_batches), RANDOM_BATCH_COUNT)

print(f"随机选择的批次索引: {sorted(selected_batch_indices)}")

# ----------------- Model definition (SmaAt_UNet / Generator) -----------------
try:
    from sam_unet.unet_parts import OutConv
    from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
    from sam_unet.layers import CBAM
    from mamba_ssm import Mamba
except Exception as e:
    print("Error importing model modules (sam_unet / mamba_ssm). Ensure PYTHONPATH includes them.")
    traceback.print_exc()
    raise


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class SmaAt_UNet(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernels_per_layer=2,
            bilinear=True,
            reduction_ratio=16,
            sequence_length=4,
            image_size=64,
            mamba_d_model=1024,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio
        self.sequence_length = sequence_length
        self.image_size = image_size

        # Encoder
        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)  # bottleneck channels

        # Decoder
        self.up1 = UpDS(256 + 512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.n_classes)

        self._mamba_d_model = mamba_d_model
        self._mamba_d_state = mamba_d_state
        self._mamba_d_conv = mamba_d_conv
        self._mamba_expand = mamba_expand

        self.pre_mamba_conv = nn.Conv2d(512, 16 * sequence_length, kernel_size=1, bias=True)
        self.post_mamba_conv = nn.Conv2d(16 * sequence_length, 512, kernel_size=1, bias=True)

        downsample_factor = 2 ** 3
        H_out = self.image_size // downsample_factor
        W_out = H_out
        x4_channels = 16 * sequence_length
        total_feat = x4_channels * H_out * W_out
        assert total_feat % self.sequence_length == 0, (
            f"channels*H*W ({total_feat}) must be divisible by sequence_length ({self.sequence_length})"
        )
        D = total_feat // self.sequence_length
        self._bottleneck_D = D

        self.mamba_proj_in = nn.Linear(D, self._mamba_d_model)
        self.mamba_proj_out = nn.Linear(self._mamba_d_model, D)
        self.mamba_dropout = nn.Dropout(p=0.1)

        self.mamba_layernorm_in = RMSNorm(self._mamba_d_model)

        self.mamba = Mamba(
            d_model=self._mamba_d_model, d_state=16, d_conv=4, expand=2
        )

    def forward(self, x, cp):
        # x: (B, in_ch, H, W); cp: (B, cp_ch, H, W)
        x1 = torch.cat((x, cp), dim=1)
        x1 = self.inc(x1)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = F.dropout2d(x4, p=0.3, training=self.training)

        x4 = self.pre_mamba_conv(x4)
        B, C, H, W = x4.shape
        T = self.sequence_length
        expected_D = self._bottleneck_D
        if (C * H * W) // T != expected_D:
            raise RuntimeError(f"Runtime bottleneck D mismatch: got {(C * H * W) // T} expected {expected_D}. "
                               "Check image_size / sequence_length consistency with model init.")

        D = expected_D
        x4_seq = x4.view(B, T, D)  # (B, T, D)

        x_proj = self.mamba_proj_in(x4_seq)  # (B, T, d_model)
        x_proj = self.mamba_layernorm_in(x_proj)

        mamba_out = self.mamba(x_proj)
        mamba_out = self.mamba_dropout(mamba_out)

        mamba_out = x_proj + mamba_out

        x_back = self.mamba_proj_out(mamba_out)  # (B, T, D)
        x_back = x_back.to(x4.device)
        x4_enhanced = x_back.view(B, C, H, W)
        x4_enhanced = self.post_mamba_conv(x4_enhanced)

        # decoder
        x = self.up1(x4_enhanced, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)
        logits = self.outc(x)
        return logits


class Generator(nn.Module):
    def __init__(self, input_channels=4, output_channels=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(Generator, self).__init__()
        self.sma_at_unet = SmaAt_UNet(
            n_channels=input_channels,
            n_classes=output_channels,
            kernels_per_layer=kernels_per_layer,
            bilinear=bilinear,
            reduction_ratio=reduction_ratio,
            sequence_length=input_channels,
            image_size=64,
            mamba_d_model=1024,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2
        )

    def forward(self, x, cp):
        return self.sma_at_unet(x, cp)


# ----------------- load model -----------------
netG = Generator(input_channels=12, output_channels=1)
if not os.path.isfile(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
netG.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cpu'))
netG.to(DEVICE)
netG.eval()


# ----------------- denormalize util -----------------
def denormalize_array(x_norm):
    x = np.array(x_norm)
    return L_ELE + (x / 2.0 + 0.5) * (H_ELE - L_ELE)


# ----------------- iterate and compute per-batch Moran's I -----------------
processed_count = 0

for batch_idx, batch in enumerate(dataloader):
    # 只处理随机选择的批次
    if batch_idx not in selected_batch_indices:
        continue

    processed_count += 1
    print(f"处理批次 {batch_idx + 1}/{total_batches}...")

    try:
        images, masked_last_image, seq_files = batch
    except Exception:
        print(f"Failed to unpack batch {batch_idx}, skipping.")
        continue

    # move to device and prepare inputs
    dems = images[:, -1, :, :, :].to(DEVICE)  # (B,1,H,W)
    pm = images[:, 0:11, :, :, :].squeeze(2).to(DEVICE)  # (B,11,H,W)
    cpLayer = masked_last_image.squeeze(2).to(DEVICE)  # (B,1,H,W)

    with torch.no_grad():
        fake = netG(pm, cpLayer)  # (B,1,H,W)

    fake_np = fake.detach().cpu().numpy()  # (B,1,H,W) normalized
    gt_np = dems.detach().cpu().numpy()  # (B,1,H,W) normalized

    # denormalize
    fake_den = denormalize_array(fake_np)  # (B,1,H,W) in original units
    gt_den = denormalize_array(gt_np)

    fake_den = np.squeeze(fake_den, axis=1)  # (B,H,W)
    gt_den = np.squeeze(gt_den, axis=1)  # (B,H,W)

    B = fake_den.shape[0]
    gt_morans = []
    fake_morans = []

    for b in range(B):
        gt_img = gt_den[b]
        fake_img = fake_den[b]

        gt_i = compute_morans_i(gt_img, neighborhood=8)
        fake_i = compute_morans_i(fake_img, neighborhood=8)

        gt_morans.append(gt_i)
        fake_morans.append(fake_i)

    # 绘制折线图：x轴为样本索引，y轴为Moran's I，两个线（gt和fake）
    plt.figure(figsize=(10, 6))
    plt.plot(range(B), gt_morans, label='Real Moran‘s I', marker='o', markersize=4)
    plt.plot(range(B), fake_morans, label='Generated Moran’s I', marker='x', markersize=4)
    plt.xlabel('Sample Index in Batch', fontsize=14)
    plt.ylabel('Moran‘s I', fontsize=14)
    # plt.title(f'Batch {batch_idx + 1} Moran‘s I Comparison (N={B})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    # gt_mean = np.mean(gt_morans)
    # fake_mean = np.mean(fake_morans)
    # plt.text(0.02, 0.98, f'Real Mean: {gt_mean:.4f}\nGenerated Mean: {fake_mean:.4f}',
    #          transform=plt.gca().transAxes, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    line_plot_path = os.path.join(OUT_DIR, f'morans_i_line_plot_batch_{batch_idx + 1}.png')
    plt.savefig(line_plot_path, dpi=500, bbox_inches='tight', pad_inches=0.05)
    plt.close()
    print(f"Saved Morans I line plot: {line_plot_path}")

    # 如果已经处理完所有选择的批次，退出循环
    if processed_count >= RANDOM_BATCH_COUNT:
        break

print(f"done!")