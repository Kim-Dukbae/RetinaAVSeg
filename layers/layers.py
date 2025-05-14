import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, count=2):
        super().__init__()
        layers = []
        for i in range(count):
            layers.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DownsampleConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.pool(x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, count=2):
        super().__init__()
        self.down = nn.Sequential(
            DownsampleConv(in_channels),
            ConvBlock(in_channels, in_channels * 2, count=count)
        )

    def forward(self, x):
        return self.down(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, r=2):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // r, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.squeeze(x).view(x.size(0), -1)
        return self.excitation(x)

class Reactivation(nn.Module):
    def __init__(self, in_channels, mid_channels, r=16, count=2):
        super().__init__()
        self.delta = in_channels
        self.mid_channels = mid_channels

        self.down = DownBlock(in_channels, count=count)
        self.se_block = SEBlock(mid_channels, r=r)

        self.bottom_conv = ConvBlock(in_channels, in_channels, count=1)
        self.final_conv = ConvBlock(in_channels * 2, in_channels, count=1)

    def _select_channels(self, x, idx):
        B, _, H, W = x.size()
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
        return torch.gather(x, dim=1, index=idx)

    def forward(self, x):
        x_feat = self.down(x)
        weights = self.se_block(x_feat)

        _, topk_idx = torch.topk(weights, k=self.delta, dim=1, largest=True)
        _, botk_idx = torch.topk(weights, k=self.delta, dim=1, largest=False)

        x_top = self._select_channels(x_feat, topk_idx)
        x_bot = self._select_channels(x_feat, botk_idx)

        x_bot = self.bottom_conv(x_bot)
        out = torch.cat([x_top, x_bot], dim=1)
        out = self.final_conv(out)

        return out

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, RA= False):
        super().__init__()

        if RA:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels*2, in_channels, count= 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, in_channels//2, count= 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Out_Conv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, num_classes, kernel_size=1, padding=0, bias= True),
        )

    def forward(self, x):
        return self.conv(x)
