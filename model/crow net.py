
class CrowNet(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes=3):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, mid_channels, count=2)
        self.encoder2 = DownBlock(mid_channels, count=2)
        self.encoder3 = Reactivation(mid_channels * 2, mid_channels * 4, r=8, count=2)
        self.encoder4 = Reactivation(mid_channels * 2, mid_channels * 4, r=8, count=2)
        self.encoder5 = Reactivation(mid_channels * 2, mid_channels * 4, r=8, count=2)

        self.decoder4 = Up(mid_channels * 2, RA= True)
        self.decoder3 = Up(mid_channels * 2, RA= True)
        self.decoder2 = Up(mid_channels * 2, RA= True)
        self.decoder1 = Up(mid_channels * 2, RA= False)

        self.outs = Out_Conv(mid_channels, num_classes)

    def forward(self, x):
        en1 = self.encoder1(x)
        en2 = self.encoder2(en1)
        en3 = self.encoder3(en2)
        en4 = self.encoder4(en3)
        en5 = self.encoder5(en4)

        de4 = self.decoder4(en5, en4)
        de3 = self.decoder3(de4, en3)
        de2 = self.decoder2(de3, en2)
        de1 = self.decoder1(de2, en1)

        return self.outs(de1)
