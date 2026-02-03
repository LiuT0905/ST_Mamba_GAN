import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
matplotlib.use('Agg')
# 设置Matplotlib后端
try:
    plt.switch_backend('TkAgg')
except:
    pass  # 如果没有TkAgg，使用默认后端

# 设置绘图风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 模型定义
# ==========================================
from sam_unet.unet_parts import OutConv
from sam_unet.unet_parts_depthwise_separable import DoubleConvDS, UpDS, DownDS
from sam_unet.layers import CBAM
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
            sequence_length=8,
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
        self._bottleneck_D = D  # original per-timestep dim

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

        # residual + dropout + norm
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
    def __init__(self, input_channels=8, output_channels=1, kernels_per_layer=2, bilinear=True, reduction_ratio=16):
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


# ==========================================
# 2. 数据集定义
# ==========================================
class CombinedDataset(Dataset):
    def __init__(self, folder_path, num_images_per_sequence, mask, transform=None, image_size=64):
        self.folder_path = folder_path
        self.num_images_per_sequence = num_images_per_sequence
        self.image_files = sorted([os.path.join(folder_path, f)
                                   for f in os.listdir(folder_path) if f.endswith(('jpg', 'png', 'tif'))])
        self.mask = mask.unsqueeze(0).unsqueeze(0)
        self.transform = transform

    def __len__(self):
        return len(self.image_files) - self.num_images_per_sequence + 1

    def __getitem__(self, idx):
        sequence_files = self.image_files[idx:idx + self.num_images_per_sequence]
        images = []
        H_ele = 2597
        for img_path in sequence_files:
            image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
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

def visualize_feature_map(features, save_path, layer_name, max_channels=64):

    if len(features.shape) == 4:
        features = features[0]

    num_channels = features.shape[0]
    display_channels = min(num_channels, max_channels)

    grid_size = int(math.ceil(math.sqrt(display_channels)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size),
                             gridspec_kw={'wspace': 0, 'hspace': 0})

    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(len(axes)):
        ax = axes[i]
        if i < display_channels:
            feat_img = features[i].cpu().detach().numpy()
            if feat_img.max() - feat_img.min() > 1e-5:
                feat_img = (feat_img - feat_img.min()) / (feat_img.max() - feat_img.min())
            ax.imshow(feat_img, cmap=plt.cm.YlGnBu, aspect='equal', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    print(f"Saving feature map for {layer_name} to {save_path}")
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_gray_image(tensor, save_path):

    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().detach().numpy()
    else:
        img = np.array(tensor)

    # 规范成 [H, W]
    if img.ndim == 3 and img.shape[0] == 1:
        img2 = img[0]
    elif img.ndim == 3 and img.shape[0] > 1:
        img2 = img[0]
    elif img.ndim == 2:
        img2 = img
    else:
        raise ValueError("Unexpected image tensor shape for save_gray_image")

    if img2.max() - img2.min() > 1e-5:
        imgn = (img2 - img2.min()) / (img2.max() - img2.min())
    else:
        imgn = np.zeros_like(img2)

    u8 = (imgn * 255.0).astype(np.uint8)
    cv2.imwrite(save_path, u8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='batch size')
    parser.add_argument('--outf', default=r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_8\VF", help='output folder')
    parser.add_argument('--netG', default=r"D:\mamba_GAN\code\experiment\ST_Mamba_GAN_8\nets\netG_epoch_300.pth",
                        help="path to netG")
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--dataset_path', default=r"D:\dataset\output_2013-2018_huabei", help='dataset path')
    parser.add_argument('--mask_path', default=r"D:\mamba_GAN\code\mask\mask_huabei.txt", help='mask path')

    parser.add_argument('--runs', type=int, default=3, help='number of samples to save')
    opt = parser.parse_args()
    print("Options:", opt)

    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    # 创建输出基础目录
    viz_base = os.path.join(opt.outf, 'feature_viz')
    os.makedirs(viz_base, exist_ok=True)

    # 1. 准备数据
    mask_path = torch.tensor(np.loadtxt(opt.mask_path, dtype=np.float32))
    transform = transforms.Compose([
        ResizeTransform(64),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    nc = 8
    dataset = CombinedDataset(
        folder_path=opt.dataset_path,
        num_images_per_sequence=nc,
        mask=mask_path,
        transform=transform,
        image_size=64
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    data_iter = iter(dataloader)

    # 2. 加载模型
    netG = Generator(input_channels=nc, output_channels=1)
    if opt.netG != '':
        print(f"Loading model from {opt.netG}")
        netG.load_state_dict(torch.load(opt.netG, map_location=device))
    netG.to(device)
    netG.eval()

    # 3. 注册 Hook
    activation = {}


    def get_activation(name):
        def hook(model, input, output):
            out = output
            if isinstance(output, (tuple, list)):
                out = output[0]
            activation[name] = out.detach().cpu().clone()

        return hook


    netG.sma_at_unet.inc.register_forward_hook(get_activation('L1_Encoder_Inc'))
    netG.sma_at_unet.up3.register_forward_hook(get_activation('L7_Decoder_Up3'))

    # ==========================================
    # 4. 智能筛选循环
    # ==========================================
    target_count = int(opt.runs)  # 目标保存数量
    saved_count = 0  # 当前已保存数量
    total_attempts = 0  # 总尝试次数
    max_attempts = 20000  # 【安全机制】最大尝试次数，防止死循环

    # 定义筛选阈值
    THRESHOLD_FEAT_STD = 1.0  # 特征图 Std 阈值
    THRESHOLD_GEN_STD = 0.05  # 生成图 Std 阈值

    print(
        f"开始筛选: 目标保存 {target_count} 张 (条件: Feat_Std > {THRESHOLD_FEAT_STD} & Gen_Std > {THRESHOLD_GEN_STD})")

    prev_last_layer_feat = None

    while saved_count < target_count:
        # 安全退出检测
        if total_attempts >= max_attempts:
            print(f"\n[Warning] 达到最大尝试次数 ({max_attempts})，提前停止。已保存 {saved_count} 张。")
            break

        total_attempts += 1
        activation.clear()

        # 获取数据
        try:
            images, masked_last_image, seq_files = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, masked_last_image, seq_files = next(data_iter)

        # 准备输入
        dems = images[:, -1, :, :, :].to(device)
        pm = images[:, 0:nc - 1, :, :, :].squeeze(2).to(device)
        cpLayer = masked_last_image.squeeze(2).to(device)

        # 推理
        with torch.no_grad():
            fake = netG(pm, cpLayer)

        # --- 筛选核心逻辑 ---

        # 1. 获取生成图指标
        gen_img_np = fake[0, 0].cpu().numpy()
        current_gen_std = gen_img_np.std()

        # 2. 获取特征图指标
        if 'L7_Decoder_Up3' in activation:
            feat_tensor = activation['L7_Decoder_Up3']
            current_feat_std = feat_tensor.std().item()
        else:
            current_feat_std = 0.0  # 异常情况

        # 3. 判断是否满足条件
        is_good_sample = (current_feat_std > THRESHOLD_FEAT_STD) and (current_gen_std > THRESHOLD_GEN_STD)


        if not is_good_sample:
            continue  # 跳过本次循环，不保存

        # --- 只有满足条件才执行以下保存逻辑 ---

        print(f"\n[Success] Found Match {saved_count + 1}/{target_count}! Processing visuals...")
        saved_count += 1

        # 命名处理
        try:
            candidate_str = str(seq_files[-1][0] if isinstance(seq_files[-1], (list, tuple)) else seq_files[-1])
            last_frame_name = os.path.basename(candidate_str).replace(' ', '_')[:50]
        except:
            last_frame_name = f"sample_{total_attempts}"

        run_folder = os.path.join(viz_base, f"run_{saved_count}_{last_frame_name}")
        os.makedirs(run_folder, exist_ok=True)

        # 保存特征图
        for layer_name, feature_map in activation.items():
            save_path = os.path.join(run_folder, f"{layer_name}.png")
            try:
                visualize_feature_map(feature_map, save_path, layer_name)
            except Exception as e:
                print(e)

        # 绘制并保存对比图 (含 Difference Map)
        gt_img = dems[0, 0].cpu().numpy()
        input_img = cpLayer[0, 0].cpu().numpy()

        diff_img = np.abs(gt_img - gen_img_np)
        mae_score = np.mean(diff_img)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))

        axs[0].imshow(input_img, cmap='gray')
        axs[0].set_title('Input (Masked)')
        axs[0].axis('off')

        axs[1].imshow(gen_img_np, cmap='gray')
        axs[1].set_title(f'Generated (Std: {current_gen_std:.3f})')
        axs[1].axis('off')

        axs[2].imshow(gt_img, cmap='gray')
        axs[2].set_title('real')
        axs[2].axis('off')

        im_diff = axs[3].imshow(diff_img, cmap='jet', vmin=0, vmax=diff_img.max())
        axs[3].set_title(f'Diff (MAE: {mae_score:.4f})')
        axs[3].axis('off')
        plt.colorbar(im_diff, ax=axs[3], fraction=0.046, pad=0.04)

        combined_path = os.path.join(run_folder, "comparison_with_diff.png")
        # plt.savefig(combined_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 保存单独的灰度图
        save_gray_image(torch.tensor(gen_img_np), os.path.join(run_folder, "generated.png"))
        save_gray_image(torch.tensor(gt_img), os.path.join(run_folder, "real.png"))

        # 保存反转后的 Input
        input_for_save = input_img
        if input_for_save.max() - input_for_save.min() > 1e-8:
            input_norm = (input_for_save - input_for_save.min()) / (input_for_save.max() - input_for_save.min())
        else:
            input_norm = input_for_save
        save_gray_image(torch.tensor(1.0 - input_norm), os.path.join(run_folder, "input_masked.png"))

        print(f"Saved results to {run_folder}")

    print("\nAll target runs complete.")
