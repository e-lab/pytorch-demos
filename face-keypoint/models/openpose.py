import torch
import torch.nn as nn

from torch.nn import functional as F

class ConvBN(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, dilation=1, bias=False,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        padding = kernel_size // 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        if activation: self.model.add_module(str(len(self.model)), activation)

    def forward(self, x):
        return self.model(x)

class ConvDepthWise(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, dilation=1, bias=False,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        padding = kernel_size // 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                      padding, dilation, bias=bias, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        if activation: self.model.add_module(str(len(self.model)), activation)

    def forward(self, x):
        return self.model(x)

class ConvDepthWiseT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, dilation=1, bias=False,
                 activation=nn.ReLU(inplace=True)):
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride // 2
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride,
                               padding=padding, dilation=dilation, output_padding=output_padding,
                               bias=bias, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, bias=bias),
            nn.BatchNorm2d(out_channels)
        )
        if activation: self.model.add_module(str(len(self.model)), activation)

    def forward(self, x):
        return self.model(x)


class OpenPose(nn.Module):

    def __init__(self, num_stages = 4):
        super().__init__()

        assert(num_stages > 0)

        # MobileNet, modified to hsave enc dec
        self.encoder = nn.ModuleList([
            nn.Sequential(
                ConvBN(3, 32, 3, 2),
                ConvDepthWise( 32,  64, 3, 1),
            ),
            nn.Sequential(
                ConvDepthWise( 64, 128, 3, 2),
                ConvDepthWise(128, 128, 3, 1)
            ),
            nn.Sequential(
                ConvDepthWise(128, 256, 3, 2),
                ConvDepthWise(256, 256, 3, 1)
            ),
            nn.Sequential(
                ConvDepthWise(256, 512, 3, 2),
                ConvDepthWise(512, 512, 3, 1)
            ),
            nn.Sequential(
                ConvDepthWise(512, 512, 3, 1),
                ConvDepthWise(512, 512, 3, 1),
                ConvDepthWise(512, 512, 3, 1),
                ConvDepthWise(512, 512, 3, 1)
            )
        ])

        self.decoder_start = nn.Sequential(
            ConvDepthWise(512, 256, 1, 1),
            ConvDepthWise(256, 256, 3, 1)
        )
        
        self.decoder = nn.ModuleList([])
        self.lateral = nn.ModuleList([])
        self.upsampler = nn.ModuleList([])

        for in_channels in reversed([64, 128, 256, 512]):
            d = ConvDepthWise(256, 256, 3, 1)
            u = ConvDepthWiseT(256, 256, 3, 2)
            l = ConvDepthWise(in_channels, 256, 1, 1)

            self.decoder.append(d)
            self.upsampler.append(u)
            self.lateral.append(l)

        self.classifier = nn.ModuleList([
            ConvDepthWise(256, 68, 3, 1)
        ])

    def _add(self, x, y):
        _,_,h,w = y.size()
        # upsample incase input is not power of 2
        # should only increase a couple of pixels (hopefully, verify)
        return F.upsample(x, size=(h,w), mode='bilinear') + y
    
    def forward(self, x):
        residuals = []
        for i, e in enumerate(self.encoder):
            x = e(x)
            residuals.append(x)
        x = self.decoder_start(x)
        
        result = []

        features = x
        f = x
        for i, c in enumerate(self.classifier):
            if i == 0: f = c(f)
            else: f = c(self._add(f, features))
        result.append(f)

        layers = zip(self.lateral, self.upsampler, self.decoder)
        for i, (l, u, d) in enumerate(layers):
            r = l(residuals[-(i+2)])
            x = u(x)
            x = self._add(x, r)
            x = d(x)
            features = x
            f = x
            for i, c in enumerate(self.classifier):
                if i == 0: f = c(f)
                else: f = c(self._add(f, features))
            result.append(f)
        return result
