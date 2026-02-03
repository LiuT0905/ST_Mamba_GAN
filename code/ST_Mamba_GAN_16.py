from __future__ import print_function
import argparse
import os
import random
import math
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.lr_scheduler import LambdaLR
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


class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf):
        super(Discriminator, self).__init__()
        self.layer1_image = nn.Sequential(
            nn.Conv2d(in_channels, int(ndf / 2), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer1_cp = nn.Sequential(
            nn.Conv2d(in_channels, int(ndf / 2), kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, cp, _cpLayer):
        out_1 = self.layer1_image(cp)
        out_2 = self.layer1_cp(_cpLayer)
        out = torch.cat((out_1, out_2), dim=1)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

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

# ----------------- Training script -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64)
    parser.add_argument('--imagewideSize', type=int, default=64)
    parser.add_argument('--imagehighSize', type=int, default=64)
    parser.add_argument('--nthread', type=int, default=0)
    parser.add_argument('--ncp', type=int, default=100)
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--outclass', type=int, default=1)
    parser.add_argument('--nk', type=int, default=1)
    parser.add_argument('--niter', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--outf', default='.')
    parser.add_argument('--manualSeed', type=int, default=3407)
    parser.add_argument('--dataset', default='PM25')
    parser.add_argument('--netD', default='')
    parser.add_argument('--netG', default='')
    parser.add_argument('--npre', type=int, default=0)
    parser.add_argument('--lambda_l1', type=float, default=5.0)
    parser.add_argument('--logfile', default='experiment/ST_Mamba_GAN_20/error/errlog.txt')
    parser.add_argument('--logfileD', default='experiment/ST_Mamba_GAN_20/error/errlog_D.txt')
    parser.add_argument('--logfileG', default='experiment/ST_Mamba_GAN_20/error/errlog_G.txt')
    parser.add_argument('--logfileMSE', default='experiment/ST_Mamba_GAN_20/error/errlog_M.txt')

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True

    L_ele = 0
    H_ele = 2597

    # dataset setup
    folder_path = r"D:\dataset\output_2013-2018_huabei"
    mask_path = r"D:\mamba_GAN\code\mask\mask_huabei.txt"
    mask_path = torch.tensor(np.loadtxt(mask_path, dtype=np.float32))

    batch_size = opt.batchSize
    num_images = 16

    transform = transforms.Compose([
        ResizeTransform(64),
        ToTensor16Bit(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CombinedDataset(folder_path=folder_path,
                              num_images_per_sequence=num_images,
                              mask=mask_path,
                              transform=transform,
                              image_size=64,
                              device='cuda')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=0, pin_memory=True)

    # init models
    ndf = opt.ndf
    ngf = opt.ngf
    nc = 16
    oc = 1
    netD = Discriminator(oc, ndf)
    netG = Generator(nc, oc)

    if (opt.netG != '' and opt.netD != ''):
        netG.load_state_dict(torch.load(opt.netG))
        netD.load_state_dict(torch.load(opt.netD))

    if (opt.cuda and torch.cuda.is_available()):
        device = torch.device("cuda")
        netG.to(device)
        netD.to(device)

    optimizerD = torch.optim.Adam(netD.parameters(), lr=opt.lr*0.35, betas=(opt.beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # LR schedulers
    warmup_epochs = 10
    total_epochs = opt.niter
    warmup_steps = warmup_epochs * len(dataloader)
    total_steps = total_epochs * len(dataloader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    schedulerD = LambdaLR(optimizerD, lr_lambda)
    schedulerG = LambdaLR(optimizerG, lr_lambda)

    criterion = nn.BCEWithLogitsLoss()
    criterionPM = nn.MSELoss(reduction='mean')

    device = torch.device("cuda" if torch.cuda.is_available() and opt.cuda else "cpu")

    # prepare placeholders
    cpLayer = torch.zeros(opt.batchSize, oc, opt.imagehighSize, opt.imagewideSize, device=device)
    last_pm = torch.zeros(opt.batchSize, oc, opt.imagehighSize, opt.imagewideSize, device=device)
    pm = torch.zeros(opt.batchSize, nc, opt.imagehighSize, opt.imagewideSize, device=device)

    print("Starting training...")
    start_time = time.time()
    current_step = 0

    for epoch in range(opt.npre + 1, opt.npre + opt.niter + 1):
        for i, (images, masked_last_image, sequence_files) in enumerate(dataloader):
            errlog = open(opt.logfile, 'a')
            errlog_D = open(opt.logfileD, 'a')
            errlog_G = open(opt.logfileG, 'a')
            errlog_M = open(opt.logfileMSE, 'a')

            # data preprocessing
            last_pm = images[:, -1, :, :, :].to(device)
            pm = images[:, 0:nc-1, :, :, :].squeeze(2).to(device)
            cpLayer = masked_last_image.squeeze(2).to(device)

            # --------- Train D ----------
            errD_real_sum = 0.0
            errD_fake_sum = 0.0

            for k in range(opt.nk):
                netD.zero_grad()

                # forward real
                output_real = netD(last_pm, cpLayer)
                real_labels = torch.full_like(output_real, 0.99, device=device)
                errD_real = criterion(output_real, real_labels)
                errD_real.backward()

                fake = netG(pm, cpLayer)
                output_fake = netD(fake.detach(), cpLayer)
                fake_labels = torch.zeros_like(output_fake, device=device)
                errD_fake = criterion(output_fake, fake_labels)
                errD_fake.backward()

                errD_real_sum += errD_real.item()
                errD_fake_sum += errD_fake.item()

                optimizerD.step()

            # average for logging
            if opt.nk > 0:
                errD_real_avg = errD_real_sum / float(opt.nk)
                errD_fake_avg = errD_fake_sum / float(opt.nk)
            else:
                errD_real_avg = 0.0
                errD_fake_avg = 0.0

            # --------- Train G ----------
            netG.zero_grad()

            output_for_g = netD(fake, cpLayer)
            real_labels_for_g = torch.ones_like(output_for_g, device=device)
            adv_loss_G = criterion(output_for_g, real_labels_for_g)

            l1_loss = F.l1_loss(fake,last_pm, reduction='mean')
            errG = adv_loss_G + opt.lambda_l1 * l1_loss

            errG1_val = adv_loss_G.item()
            l1_loss_val = l1_loss.item()
            errD_total_val = errD_real_avg + errD_fake_avg
            errG_val = errG.item()

            errG.backward()
            optimizerG.step()

            schedulerD.step(current_step)
            schedulerG.step(current_step)
            current_step += 1

            if ((i + 1) % 40 == 0):
                vutils.save_image(fake.data, '%s/experiment/ST_Mamba_GAN_20/image/epoch_%03d_batch_%03d_fake.png' % (
                opt.outf, epoch, i), normalize=True)
                vutils.save_image(last_pm.data, '%s/experiment/ST_Mamba_GAN_20/image/epoch_%03d_batch_%03d_real.png' % (
                opt.outf, epoch, i), normalize=True)
                last_pm.data.copy_(L_ele + (last_pm.data / 2 + 0.5) * (H_ele - L_ele))
                fake.data.copy_(L_ele + (fake.data / 2 + 0.5) * (H_ele - L_ele))
                errPM = criterionPM(fake, last_pm)

                errlog.write(
                    '[%d/%d][%d/%d] Loss_D: %.4f (D_real: %.4f, D_fake: %.4f) Loss_G: %.4f (G_adv: %.4f, L1: %.4f) MSE: %.2f\n' % (
                        epoch, opt.npre + opt.niter, i + 1, len(dataloader),
                        errD_total_val, errD_real_avg, errD_fake_avg,
                        errG_val, errG1_val, l1_loss_val, errPM.item()))
                errlog_D.write(('%.4f\n') % (errD_total_val))
                errlog_G.write(('%.4f\n') % (errG_val))
                errlog_M.write(('%.4f\n') % (errPM.item()))

                print(
                    '[%d/%d][%d/%d] Loss_D: %.4f (D_real: %.4f, D_fake: %.4f) Loss_G: %.4f (G_adv: %.4f, L1: %.4f) MSE: %.2f' % (
                        epoch, opt.npre + opt.niter, i + 1, len(dataloader),
                        errD_total_val, errD_real_avg, errD_fake_avg,
                        errG_val, errG1_val, l1_loss_val, errPM.item()))

            errlog.close()
            errlog_G.close()
            errlog_D.close()
            errlog_M.close()

        if epoch in (1,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300):
            torch.save(netG.state_dict(), f'{opt.outf}/experiment/ST_Mamba_GAN_20/nets/netG_epoch_{epoch:03d}.pth')

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"Epoch {epoch} completed. Cumulative training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        with open(opt.logfile, 'a') as f:
            f.write(f"Epoch {epoch} completed. Cumulative training time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}\n")