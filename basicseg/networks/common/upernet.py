import torch
import torch.nn as nn
import torch.nn.functional as F

class Upernet_head(nn.Module):
    def __init__(self, in_planes=[64, 128, 256, 512]):
        super().__init__()
        scale = [1, 2, 3, 6]
        fpn_dim = 256
        self.ppm_module = nn.ModuleList()
        for i in range(len(scale)):
            self.ppm_module.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale[i]),
                nn.Conv2d(in_planes[-1], 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(in_planes[-1] + len(scale) * 512, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )
        self.fpn_module = nn.ModuleList()
        self.fpn_smooth = nn.ModuleList()
        for i in range(len(in_planes) - 1):
            self.fpn_module.append(nn.Sequential(
                nn.Conv2d(in_planes[i], fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
            self.fpn_smooth.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))

        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(len(in_planes) * fpn_dim, fpn_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fpn_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat_maps = x
        feat_top = feat_maps[-1]
        ppm_size = feat_top.shape[-2:]
        ppm_out = []
        ppm_out.append(feat_top)
        for i in range(len(self.ppm_module)):
            out = self.ppm_module[i](feat_top)
            ppm_out.append(F.interpolate(out, size=ppm_size, mode='bilinear', align_corners=False))
        ppm_out = torch.cat(ppm_out, dim=1)
        ppm_out = self.ppm_fuse(ppm_out)

        fpn_out = []
        fpn_out.append(ppm_out)
        f = ppm_out
        for i in reversed(range(len(self.fpn_module))):
            size = feat_maps[i].shape[-2:]
            out = self.fpn_module[i](feat_maps[i])
            f = out + F.interpolate(f, size, mode='bilinear', align_corners=False)
            fpn_out.append(self.fpn_smooth[i](f))
        fpn_out.reverse()
        fpn_fush = []
        fpn_fush.append(fpn_out[0])
        for i in range(1, len(fpn_out)):
            size = fpn_out[0].shape[-2:]
            fpn_fush.append(F.interpolate(fpn_out[i], size, mode='bilinear', align_corners=False))
        out = torch.cat(fpn_fush, dim=1)
        out = self.fpn_fuse(out)
        return out