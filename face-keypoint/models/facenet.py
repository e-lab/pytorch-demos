import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet

class FaceNet(nn.Module):

    def __init__(self):
        super().__init__()

        base_net = resnet.resnet34(pretrained=True)
        
        self.in_block = nn.Sequential(
            base_net.conv1,
            base_net.bn1,
            base_net.relu,
            base_net.maxpool
        )

        self.encoder = nn.ModuleList([
            base_net.layer1,
            base_net.layer2,
            base_net.layer3,
            base_net.layer4
        ])

        self.decoder_start = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder = nn.ModuleList([])
        self.lateral = nn.ModuleList([])
        self.upsampler = nn.ModuleList([])

        for in_channels in reversed([64, 128, 256]):
            d = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            u = nn.Sequential(
                nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            l = nn.Sequential(
                nn.Conv2d(in_channels, 256, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.decoder.append(d)
            self.upsampler.append(u)
            self.lateral.append(l)

        self.classifier = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Conv2d(256, 68, 3, padding=1)
        ])

    def _cat(self, x, y):
        _, _, h, w = y.size()
        return torch.cat((x[:, :, :h, :w], y), 1)

    def _add(self, x, y):
        _,_,h,w = y.size()
        return F.upsample(x, size=(h,w), mode='bilinear') + y
    
    def forward(self, x):
        x = self.in_block(x)
        residuals = []
        for e in self.encoder:
            x = e(x)
            residuals.append(x)
        x = self.decoder_start(x)
        result = []
        for i, (l, u, d) in enumerate(zip(self.lateral, self.upsampler, self.decoder)):
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
    
if __name__ == "__main__":

    from torch.autograd import Variable as V

    x = torch.zeros((1,3,256,256))
    m = FaceNet()
    m.forward(V(x))


