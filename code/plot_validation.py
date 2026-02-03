import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import math
from mamba_ssm import Mamba

# 假设这些模块在您的工程目录下仍然可用
from sam_unet.unet_parts import OutConv
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from sam_unet.layers import CBAM

# 解决Matplotlib中文显示问题
plt.switch_backend('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False


# ----------------- 1. 模型定义 (保持不变) -----------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class SmaAt_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, kernels_per_layer=2, bilinear=True, reduction_ratio=16,
                 sequence_length=8, image_size=64, mamba_d_model=1024, mamba_d_state=16, mamba_d_conv=4,
                 mamba_expand=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.kernels_per_layer = kernels_per_layer
        self.bilinear = bilinear
        self.reduction_ratio = reduction_ratio
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.inc = DoubleConvDS(self.n_channels, 64, kernels_per_layer=kernels_per_layer)
        self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)
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
        D = total_feat // self.sequence_length
        self._bottleneck_D = D
        self.mamba_proj_in = nn.Linear(D, self._mamba_d_model)
        self.mamba_proj_out = nn.Linear(self._mamba_d_model, D)
        self.mamba_dropout = nn.Dropout(p=0.1)
        self.mamba_layernorm_in = RMSNorm(self._mamba_d_model)
        self.mamba = Mamba(d_model=self._mamba_d_model, d_state=16, d_conv=4, expand=2)

    def forward(self, x, cp):
        x1 = torch.cat((x, cp), dim=1)
        x1 = self.inc(x1)
        x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        x3Att = self.cbam3(x3)
        x4 = self.down3(x3)
        x4 = self.pre_mamba_conv(x4)
        B, C, H, W = x4.shape
        T = self.sequence_length
        D = self._bottleneck_D
        x4_seq = x4.view(B, T, D)
        x_proj = self.mamba_proj_in(x4_seq)
        x_proj = self.mamba_layernorm_in(x_proj)
        mamba_out = self.mamba(x_proj)
        mamba_out = self.mamba_dropout(mamba_out)
        mamba_out = x_proj + mamba_out
        x_back = self.mamba_proj_out(mamba_out)
        x_back = x_back.to(x4.device)
        x4_enhanced = x_back.view(B, C, H, W)
        x4_enhanced = self.post_mamba_conv(x4_enhanced)
        x = self.up1(x4_enhanced, x3Att)
        x = self.up2(x, x2Att)
        x = self.up3(x, x1Att)
        logits = self.outc(x)
        return logits


class Generator(nn.Module):
    def __init__(self, input_channels=8, output_channels=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
        super(Generator, self).__init__()
        self.sma_at_unet = SmaAt_UNet(
            n_channels=input_channels,
            n_classes=output_channels,
            kernels_per_layer=kernels_per_layer,
            bilinear=bilinear,
            reduction_ratio=reduction_ratio,
            sequence_length=input_channels,
            image_size=64
        )

    def forward(self, x, cp):
        return self.sma_at_unet(x, cp)


# ----------------- 数据集与辅助函数 -----------------
class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64, h_ele=990):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'tif'))])
        # Mask 只需要加载一次，unsqueeze逻辑可能需要根据实际 mask 维度调整
        # 假设 mask 是 (H, W)，我们需要 (1, H, W)
        if mask.ndim == 2:
            self.mask = mask.unsqueeze(0)
        else:
            self.mask = mask

        self.image_size = image_size
        self.transform = transform
        self.h_ele = h_ele

    def __len__(self):
        return len(self.image_files) - self.num_images_per_sequence + 1

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx:idx + self.num_images_per_sequence]
        images = []
        for img_path in sequence_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            image = image.astype(np.float32) / self.h_ele
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        last_image_original = images[-1].clone()
        masked_last_image = last_image_original * self.mask
        return images, masked_last_image


def update_csv_rmse(csv_path, nc, epoch, rmse_value):
    column_name = f"epoch_{epoch}"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'nc' in df.columns:
                df.set_index('nc', inplace=True)
        except Exception:
            df = pd.DataFrame()
            df.index.name = 'nc'
    else:
        df = pd.DataFrame()
        df.index.name = 'nc'

    if column_name not in df.columns:
        df[column_name] = np.nan

    df.loc[nc, column_name] = rmse_value

    try:
        cols = [c for c in df.columns if c.startswith('epoch_')]
        cols_sorted = sorted(cols, key=lambda x: int(x.split('_')[1]))
        df = df[cols_sorted]
    except:
        pass

    df.to_csv(csv_path, index=True)


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 核心参数
    parser.add_argument('--start_epoch', type=int, default=20)
    parser.add_argument('--end_epoch', type=int, default=300)
    parser.add_argument('--interval', type=int, default=20, help='Epoch interval')
    parser.add_argument('--target_batch', type=int, default=116, help='指定计算batch (0开始)')

    # 路径参数
    parser.add_argument('--experiment_root', default='experiment', help='实验根目录')
    parser.add_argument('--dataset_path', default=r"D:\dataset\output_2019_huabei")
    parser.add_argument('--mask_path', default=r"D:\mamba_GAN\code\mask\mask_huabei.txt")
    parser.add_argument('--output_dir', default='experiment/all_nc_results', help='结果保存目录')

    # 配置
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--H_ele', type=float, default=990)
    # 下面这个参数主要作为占位，代码中会自动检测 CUDA
    parser.add_argument('--cuda', action='store_true', default=True)

    opt = parser.parse_args()

    # ---------------- GPU 设置与诊断 ----------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # 加速卷积运算
        print(f"==================================================")
        print(f" [INFO] CUDA is available. Running on GPU: {torch.cuda.get_device_name(0)}")
        print(f"==================================================")
    else:
        device = torch.device("cpu")
        print(f" [WARNING] CUDA not available. Running on CPU (Will be slow).")

    # 配置列表
    nc_configs = [4, 8, 12, 16, 20, 24]

    os.makedirs(opt.output_dir, exist_ok=True)
    image_save_dir = os.path.join(opt.output_dir, 'images')
    os.makedirs(image_save_dir, exist_ok=True)
    rmse_csv_path = os.path.join(opt.output_dir, 'all_rmse_results.csv')

    mask_np = np.loadtxt(opt.mask_path, dtype=np.float32)
    mask = torch.tensor(mask_np)

    transform = transforms.Compose([
        ResizeTransform(64),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    criterionMSE = nn.MSELoss()

    epochs_to_test = list(range(opt.start_epoch, opt.end_epoch + 1, opt.interval))
    if 1 not in epochs_to_test:
        epochs_to_test.insert(0, 1)

    print(f"Start Multi-NC Processing")
    print(f"Target Batch Index: {opt.target_batch}")
    print(f"Configs (NC): {nc_configs}")
    print(f"Epochs: {epochs_to_test}\n")

    all_nc_results = {}

    for nc in nc_configs:
        print(f"--- Processing NC={nc} ---")

        model_dir = os.path.join(opt.experiment_root, f"ST_Mamba_GAN_{nc}", "nets")
        if not os.path.exists(model_dir):
            print(f"Directory not found: {model_dir}. Skipping NC={nc}.")
            continue

        # ---------------- 极速优化：只加载目标 Batch 的数据 ----------------
        # 1. 初始化完整数据集
        full_dataset = CombinedDataset(opt.dataset_path, num_images_per_sequence=nc, mask=mask, transform=transform,
                                       h_ele=opt.H_ele)

        # 2. 计算目标 batch 对应的样本索引范围
        start_idx = opt.target_batch * opt.batchSize
        end_idx = start_idx + opt.batchSize

        # 3. 检查越界
        if end_idx > len(full_dataset):
            print(
                f"Error: Target batch {opt.target_batch} out of range for NC={nc}. Max batches: {len(full_dataset) // opt.batchSize}")
            continue

        # 4. 创建 Subset (只包含这64张图，不读取其他的)
        target_indices = list(range(start_idx, end_idx))
        subset_dataset = Subset(full_dataset, target_indices)

        # 5. 创建 DataLoader (此时 len(dataloader) 只有 1)
        # pin_memory=True 加速 CPU 到 GPU 传输
        dataloader = DataLoader(subset_dataset, batch_size=opt.batchSize, shuffle=False, drop_last=False, num_workers=0,
                                pin_memory=True)

        current_run_epochs = []
        current_run_rmse = []

        for epoch in epochs_to_test:
            model_name = f"netG_epoch_{epoch:03d}.pth"
            model_path = os.path.join(model_dir, model_name)

            if not os.path.exists(model_path):
                continue

            try:
                # 初始化对应 nc 的模型
                netG = Generator(input_channels=nc, output_channels=1)
                # 这里的 map_location 确保加载时就放入 GPU
                netG.load_state_dict(torch.load(model_path, map_location=device))
                netG.to(device)
                netG.eval()
            except Exception as e:
                print(f"Error loading model nc={nc}, epoch={epoch}: {e}")
                continue

            with torch.no_grad():
                # 这个循环只会执行一次，因为 dataloader 里只有 1 个 batch
                for i, (images, masked_last_image) in enumerate(dataloader):

                    # 数据移入 GPU
                    dems = images[:, -1, :, :, :].to(device)
                    # 确保维度匹配
                    if images.shape[1] < nc:
                        print(f"Data shape mismatch. Expected {nc} frames.")
                        break

                    pm = images[:, 0:nc - 1, :, :, :].squeeze(2).to(device)
                    cpLayer = masked_last_image.squeeze(2).to(device)

                    fake = netG(pm, cpLayer)

                    #RMSE
                    dems_denorm = (dems * 0.5 + 0.5) * opt.H_ele
                    fake_denorm = (fake * 0.5 + 0.5) * opt.H_ele
                    mse_loss = criterionMSE(fake_denorm, dems_denorm)
                    rmse_val = math.sqrt(mse_loss.item())

                    print(f"[nc={nc}] Epoch {epoch} | RMSE: {rmse_val:.4f}")

                    # SAVE
                    real_filename = f"nc{nc}_epoch{epoch}_batch{opt.target_batch}_real.png"
                    fake_filename = f"nc{nc}_epoch{epoch}_batch{opt.target_batch}_fake.png"
                    vutils.save_image(dems.cpu(), os.path.join(image_save_dir, real_filename), normalize=True)
                    vutils.save_image(fake.cpu(), os.path.join(image_save_dir, fake_filename), normalize=True)

                    update_csv_rmse(rmse_csv_path, nc, epoch, rmse_val)
                    current_run_epochs.append(epoch)
                    current_run_rmse.append(rmse_val)

        if len(current_run_epochs) > 0:
            all_nc_results[nc] = {'epochs': current_run_epochs, 'rmse': current_run_rmse}

            plt.figure(figsize=(10, 6))
            plt.plot(current_run_epochs, current_run_rmse, marker='o', color='blue')
            plt.title(f'RMSE Trend for NC={nc} (Batch {opt.target_batch})')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.ylim(0, 20)
            plt.grid(True)
            # plt.savefig(os.path.join(opt.output_dir, f'rmse_trend_nc_{nc}.png'))
            plt.close()

    # ----------------- 汇总绘图 -----------------
    if len(all_nc_results) > 0:
        plt.figure(figsize=(12, 8))
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown']

        markers = ['o', '^', 's', 'D', 'v', '*']

        for idx, nc in enumerate(nc_configs):
            if nc in all_nc_results:
                data = all_nc_results[nc]
                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]

                plt.plot(data['epochs'], data['rmse'], marker=marker, label=f'Time series length = {nc}', color=color)

        # plt.title(f'RMSE Comparison Across Different NCs (Batch {opt.target_batch})')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('RMSE', fontsize=14)
        plt.ylim(0, 20)
        plt.legend(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        final_plot_path = os.path.join(opt.output_dir, 'rmse_comparison_all_epoch.png')
        plt.savefig(final_plot_path ,dpi=500, bbox_inches='tight', pad_inches=0.05)
        print(f"\n[Done] Final consolidated plot saved to: {final_plot_path}")
        print(f"[Done] Consolidated CSV saved to: {rmse_csv_path}")
    else:
        print("No results were generated.")