from pathlib import Path
import matplotlib.pyplot as plt
import csv
import scipy.stats as st
import argparse
import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import LambdaLR
from PIL import Image
from sam_unet.unet_parts import OutConv
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from sam_unet.layers import CBAM
# Mamba SSM
from mamba_ssm import Mamba


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


# ----------------- Dataset utils -----------------
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64, device='cuda'):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'tif'))])
        self.mask = mask
        self.image_size = image_size
        self.device = device
        self.transform = transform

    def __len__(self):
        return len(self.image_files) - self.num_images_per_sequence + 1

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx:idx + self.num_images_per_sequence]
        images = []
        for img_path in sequence_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            image = image.astype(np.float32) / 990
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        last_image_original = images[-1].clone()
        masked_last_image = last_image_original * self.mask
        return images, masked_last_image, sequence_files


# ----------------- Helper utilities -----------------
def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_mask(mask_path):
    m = np.loadtxt(mask_path, dtype=np.float32)
    m = (m != 0).astype(np.uint8)
    return m


def inv_transform(t):
    return (t / 2 + 0.5) * 990


# ----------------- Temporal consistency at station locations -----------------
def extract_timeseries_at_stations(dataloader, netG, station_coords, device, L_ele, H_ele, max_batches=0):
    netG.eval()
    ts = {c: {'real': [], 'pred': []} for c in station_coords}
    processed = 0
    with torch.no_grad():
        for step, (images, masked_last_image, seq_files) in enumerate(dataloader):
            if max_batches > 0 and processed >= max_batches:
                break
            images = images.to(device)  # (B,T,1,H,W)
            dems = images[:, -1, :, :, :].to(device)
            pm = images[:, 0:11, :, :, :].squeeze(2).to(device)
            cpLayer = masked_last_image.to(device)

            fake = netG(pm, cpLayer)

            real_phys = inv_transform(dems)  # (B,1,H,W)
            fake_phys = inv_transform(fake)

            B = real_phys.shape[0]
            for b in range(B):
                for (yy, xx) in station_coords:
                    H_idx = real_phys.shape[-2]
                    W_idx = real_phys.shape[-1]
                    if yy < 0 or yy >= H_idx or xx < 0 or xx >= W_idx:
                        continue
                    rv = float(real_phys[b, 0, yy, xx].cpu().numpy())
                    fv = float(fake_phys[b, 0, yy, xx].cpu().numpy())
                    ts[(yy, xx)]['real'].append(rv)
                    ts[(yy, xx)]['pred'].append(fv)
            processed += 1
    return ts


def compute_time_metrics_and_plots(ts_dict, outdir):
    ensure_dir(outdir)
    for coord, d in ts_dict.items():
        r = np.array(d['real'], dtype=float)
        p = np.array(d['pred'], dtype=float)

        if r.size == 0:
            continue

        plt.figure(figsize=(8, 3))
        plt.plot(r, label='real')
        plt.plot(p, label='generated', alpha=0.6)
        # plt.title(f'Station {coord}') # 如果需要标题可以取消注释
        plt.ylabel('Values', fontsize=11)
        plt.xlabel('Time (hour)', fontsize=11)
        plt.legend()
        plt.tight_layout()

        # 保存图片
        save_path = Path(outdir) / f'timeseries_{coord[0]}_{coord[1]}.png'
        plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.05)
        plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default=r"D:\\dataset\\output_2019_huabei")
    parser.add_argument('--mask_path', type=str, default=r"D:\mamba_GAN\code\mask\mask_huabei.txt")
    parser.add_argument('--model_weights', type=str,
                        default=r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_12\nets\netG_epoch_300.pth")
    parser.add_argument('--outdir', type=str, default=r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_12\temporal_analysis")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--max_batches', type=int, default=0)  # 0 表示处理全部
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--select_n_stations', type=int, default=3)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # 加载 mask 并选取站点
    mask_np = load_mask(args.mask_path)
    station_coords = list(zip(*np.nonzero(mask_np)))
    if len(station_coords) == 0:
        raise RuntimeError('No station locations found in mask')
    random.seed(0)
    if args.select_n_stations >= len(station_coords):
        chosen = station_coords
    else:
        chosen = random.sample(station_coords, args.select_n_stations)
    print('Chosen stations (y,x):', chosen)

    # 数据变换与数据集
    transform = transforms.Compose([
        ResizeTransform(args.image_size),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CombinedDataset(folder_path=args.folder_path,
                              num_images_per_sequence=12,
                              mask=mask_np,
                              transform=transform,
                              image_size=args.image_size,
                              device=args.device)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,   # 确保处理所有样本
                            num_workers=0,
                            pin_memory=True)

    # 加载模型
    netG = Generator(input_channels=12, output_channels=1)
    state = torch.load(args.model_weights, map_location='cpu')
    try:
        netG.load_state_dict(state)
    except Exception:
        if isinstance(state, dict) and 'state_dict' in state:
            netG.load_state_dict(state['state_dict'])
        else:
            netG.load_state_dict(state, strict=False)
    netG.to(args.device)
    netG.eval()

    # 提取并分析时间序列
    ts_outdir = outdir / 'temporal'
    ts = extract_timeseries_at_stations(dataloader, netG, chosen, args.device, 0.0, 990.0, max_batches=args.max_batches)
    compute_time_metrics_and_plots(ts, ts_outdir)

    print('Temporal analysis completed. Results saved to:', ts_outdir)


if __name__ == '__main__':
    main()