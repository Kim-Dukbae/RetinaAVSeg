import torch
import torch.nn as nn
import torch.nn.functional as F


class Convolution(nn.Module):
    """
    Convolution Layer Block

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        count (int): Number of (convolution => [BN] => ReLU) blocks to apply after the initial layer.
        
    Structure:
        (convolution => [BN] => ReLU) * (count + 1)
    """

    def __init__(self, in_channels, out_channels, padding= 1, count=2):
        super().__init__()

        # Create the initial convolution layer
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        # Add 'count' number of Conv => BN => ReLU blocks
        for _ in range(count - 1):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        # Convert list of layers to a sequential block
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)
