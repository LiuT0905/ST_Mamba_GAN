import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import argparse
import time
import math
from sam_unet.unet_parts import OutConv
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from sam_unet.layers import CBAM
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from mamba_ssm import Mamba

# --- 新增：导入评估指标库 (仅保留 FID) ---
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    print("错误: 未找到 torchmetrics 库。请运行 `pip install torchmetrics[image]` 进行安装。")
    exit()

# 设置Matplotlib后端
plt.switch_backend('TkAgg')

# 设置支持中文的字体
plt.rcParams['font.family'] = 'Times New Roman ,simsun'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


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
            # fixed Mamba config (as you requested)
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
        # self.cbam1 = CBAM(64, reduction_ratio=reduction_ratio)
        self.down1 = DownDS(64, 128, kernels_per_layer=kernels_per_layer)
        # self.cbam2 = CBAM(128, reduction_ratio=reduction_ratio)
        self.down2 = DownDS(128, 256, kernels_per_layer=kernels_per_layer)
        # self.cbam3 = CBAM(256, reduction_ratio=reduction_ratio)
        factor = 2 if self.bilinear else 1
        self.down3 = DownDS(256, 512, kernels_per_layer=kernels_per_layer)  # bottleneck channels

        # Decoder
        self.up1 = UpDS(256 + 512, 256 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up2 = UpDS(256, 128 // factor, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.up3 = UpDS(128, 64, self.bilinear, kernels_per_layer=kernels_per_layer)
        self.outc = OutConv(64, self.n_classes)

        # ---------------- Mamba related ----------------
        self._mamba_d_model = mamba_d_model  # 模型维度
        self._mamba_d_state = mamba_d_state  # SSM 状态维度 / 状态展开因子
        self._mamba_d_conv = mamba_d_conv  # 局部卷积宽度
        self._mamba_expand = mamba_expand  # 模块内部扩展因子

        # projection convs between 512 <-> mamba_d_model / 64  * self.sequence_length
        self.pre_mamba_conv = nn.Conv2d(512, 16 * sequence_length, kernel_size=1, bias=True)
        self.post_mamba_conv = nn.Conv2d(16 * sequence_length, 512, kernel_size=1, bias=True)

        # precompute bottleneck shape given image_size and 3 downsamples
        downsample_factor = 2 ** 3
        H_out = self.image_size // downsample_factor
        W_out = H_out
        x4_channels = 16 * sequence_length
        total_feat = x4_channels * H_out * W_out
        assert total_feat % self.sequence_length == 0, (
            f"channels*H*W ({total_feat}) must be divisible by sequence_length ({self.sequence_length})"
        )
        D = total_feat // self.sequence_length
        self._bottleneck_D = D  # original per-timestep dim

        self.mamba_proj_in = nn.Linear(D, self._mamba_d_model)
        self.mamba_proj_out = nn.Linear(self._mamba_d_model, D)
        self.mamba_dropout = nn.Dropout(p=0.1)
        # self.mamba_layernorm = nn.LayerNorm(self._mamba_d_model)
        # self.mamba_layernorm_in = nn.LayerNorm(self._mamba_d_model)
        # self.mamba_layernorm_out = nn.LayerNorm(self._mamba_d_model)
        self.mamba_layernorm_in = RMSNorm(self._mamba_d_model)
        # self.mamba_layernorm_out = RMSNorm(self._mamba_d_model)

        # self.mamba = Mamba(d_model=1024, d_state=16, d_conv=4, expand=2).to("cuda")
        self.mamba = Mamba(
            d_model=self._mamba_d_model, d_state=16, d_conv=4, expand=2
        )

    def forward(self, x, cp):
        # x: (B, in_ch, H, W); cp: (B, cp_ch, H, W)
        x1 = torch.cat((x, cp), dim=1)
        x1 = self.inc(x1)
        # x1Att = self.cbam1(x1)
        x2 = self.down1(x1)
        # x2Att = self.cbam2(x2)
        x3 = self.down2(x2)
        # x3Att = self.cbam3(x3)
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

        # move sequence to mamba device and project in
        x_proj = self.mamba_proj_in(x4_seq)  # (B, T, d_model)
        x_proj = self.mamba_layernorm_in(x_proj)


        mamba_out = self.mamba(x_proj)
        mamba_out = self.mamba_dropout(mamba_out)

        # residual + dropout + norm
        mamba_out = x_proj + mamba_out
        # mamba_out = self.mamba_layernorm_out(mamba_out)

        # project back and return to encoder device
        x_back = self.mamba_proj_out(mamba_out)  # (B, T, D)
        x_back = x_back.to(x4.device)
        x4_enhanced = x_back.view(B, C, H, W)
        x4_enhanced = self.post_mamba_conv(x4_enhanced)

        # decoder
        x = self.up1(x4_enhanced, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits


# ----------------- Generator & Discriminator (unchanged logic) -----------------
class Generator(nn.Module):
    def __init__(self, input_channels=20, output_channels=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
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


class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64, device='cuda'):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'tif'))])
        self.mask = mask.unsqueeze(0).unsqueeze(0)
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
            image = image.astype(np.float32) / H_ele
            if self.transform:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images)
        last_image_original = images[-1].clone()
        masked_last_image = last_image_original * self.mask
        return images, masked_last_image, sequence_files


class ResizeTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)


class ToTensor16Bit:
    def __call__(self, img):
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


# 参数解析
parser = argparse.ArgumentParser(description='Test script for SmaAt-UNet GAN')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imagewideSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--imagehighSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--outf', default='.', help='folder to output images and logs')
parser.add_argument('--dataset', default='PM25', help='which dataset to test on, DEM')
parser.add_argument('--netG', default='', help="path to netG (to load trained model)")
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--logfile', default='experiment/withoutCBAM/error/test_metrics.txt', help='main test log file')
parser.add_argument('--test_epoch', default=300, help='test_epoch')

opt = parser.parse_args()
print(opt)

# 设置随机种子
manualSeed = 3407
torch.manual_seed(manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(manualSeed)

# 范围
L_ele = 0
H_ele = 990

# 数据集路径
folder_path = r"D:\dataset\output_2019_huabei"
mask_path = r"D:\mamba_GAN\code\mask\mask_huabei.txt"
mask_path = torch.tensor(np.loadtxt(mask_path, dtype=np.float32))

# 数据变换
transform = transforms.Compose([
    ResizeTransform(64),
    ToTensor16Bit(),
    transforms.Normalize((0.5,), (0.5,))
])

nc = 12
oc = 1
# 初始化数据集和数据加载器
dataset = CombinedDataset(
    folder_path=folder_path,
    num_images_per_sequence=nc,
    mask=mask_path,
    transform=transform,
    image_size=64,
    device='cuda' if opt.cuda else 'cpu'
)

dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, drop_last=True,
                        num_workers=0, pin_memory=True)

# 初始化模型
netG = Generator(nc, oc)
if opt.cuda:
    netG.cuda()

# 损失函数
criterionMSE = nn.MSELoss(reduction='mean')

# 设备
device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

# --- 新增：初始化 FID 指标 ---
print("Initializing metrics (FID)...")
# FID: feature=2048 是标准设置
fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
print("Metrics initialized.")

# 创建输出目录
os.makedirs(f'{opt.outf}/experiment/withoutCBAM/test_images', exist_ok=True)
os.makedirs(os.path.dirname(opt.logfile), exist_ok=True)

# 测试指定epoch
epoch = opt.test_epoch

netG_path = f'{opt.outf}/experiment/withoutCBAM/nets/netG_epoch_{epoch:03d}.pth'
if not os.path.exists(netG_path):
    print(f"Model for epoch {epoch} not found at {netG_path}. Exiting.")
    exit()

# 加载模型
netG.load_state_dict(torch.load(netG_path))
netG.eval()

# 存储结果
rmse_list = []

start_time = time.time()
with torch.no_grad():
    for i, (images, masked_last_image, sequence_files) in enumerate(dataloader):
        dems = images[:, -1, :, :, :].to(device)  # [B,1,H,W] - 真实标签
        pm = images[:, 0:nc - 1, :, :, :].squeeze(2).to(device)  # [B,nc-1,H,W] - 输入
        cpLayer = masked_last_image.squeeze(2).to(device)  # [B,1,H,W] - 掩码输入

        # 1. 生成预测图像
        fake = netG(pm, cpLayer)

        # ---------------- RMSE 计算 (基于物理数值) ----------------
        dems_denorm = (dems * 0.5 + 0.5) * H_ele
        fake_denorm = (fake * 0.5 + 0.5) * H_ele

        mse = criterionMSE(fake_denorm, dems_denorm).item()
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        # ---------------- FID 数据准备 ----------------
        # 扩展为 3 通道 [B, 3, H, W]
        real_3ch = dems.repeat(1, 3, 1, 1)
        fake_3ch = fake.repeat(1, 3, 1, 1)

        # ---------------- FID 更新 ----------------
        # 转换为 [0, 1]
        real_01 = (real_3ch * 0.5 + 0.5).clamp(0, 1)
        fake_01 = (fake_3ch * 0.5 + 0.5).clamp(0, 1)

        # 转换为 uint8 [0, 255]
        real_uint8 = (real_01 * 255).to(torch.uint8)
        fake_uint8 = (fake_01 * 255).to(torch.uint8)

        # 更新 FID 统计数据
        fid_metric.update(real_uint8, real=True)
        fid_metric.update(fake_uint8, real=False)

        # 打印进度
        print(f'[Epoch {epoch}][Batch {i + 1}/{len(dataloader)}] RMSE: {rmse:.4f}')

# ---------------- 计算最终指标 ----------------
# 1. 平均 RMSE
avg_rmse = float(np.mean(rmse_list)) if len(rmse_list) > 0 else float('nan')

# 2. 计算全局 FID
print("Computing FID...")
fid_score = fid_metric.compute().item()

# ---------------- 输出与日志 ----------------
result_str = (
    f'\n================ Test Results (Epoch {epoch}) ================\n'
    f'Average RMSE : {avg_rmse:.6f}\n'
    f'FID Score    : {fid_score:.6f}\n'
    f'===========================================================\n'
)

print(result_str)

# 写入日志文件
with open(opt.logfile, 'a') as f:
    f.write(result_str)

# 打印耗时
elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Testing completed. Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

# 如果需要释放显存
fid_metric.reset()