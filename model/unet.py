import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    def __init__(self, in_ch, h_ch, num_class):
        super(Unet, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
          nn.Conv2d()
        )

        # decoder
        self.decoder = nn.Sequential(
          nn.Conv2d()
        )

        self.outc = nn.Conv2d(f_channels//2, n_classes, kernel_size= 3, padding=1)

    def forward(self, x):
        ''' encoder '''
        x = self.encoder(x)

        ''' decoder '''
        x = self.decoder(x)
        x = self.outc(fpn_outs)

        return x
