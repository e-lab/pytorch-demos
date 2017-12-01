import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import resnet

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        padding = kernel_size // 2
        output_padding = stride // 2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//4, 1, 1, padding=0, dilation=1),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels//4, in_channels//4, 3, stride, padding=1, dilation=1,
                               output_padding=output_padding),
            nn.BatchNorm2d(in_channels//4),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels//4, out_channels, 1, 1, padding=0, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.model(x)
        
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

        enc_channels = list(reversed([64, 128, 256, 512]))
        
        self.decoder = nn.ModuleList([])
        self.lateral = nn.ModuleList([])    
        for in_channel, out_channel in zip(enc_channels[:-1], enc_channels[1:]):
            d = DecoderBlock(in_channel, out_channel, 3, 2)
            l = nn.Sequential(
                nn.Conv2d(out_channel, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )
            self.decoder.append(d)
            self.lateral.append(l)
            
        self.classifier = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True)),
            nn.Conv2d(256, 68, 3, padding=1)
        ])

    def _add(self, x, y):
        _,_,h,w = y.size()
        return F.upsample(x, size=(h,w), mode='bilinear') + y
    
    def forward(self, x):
        x = self.in_block(x)
        
        residuals = []
        for e in self.encoder:
            x = e(x)
            residuals.append(x)

        result = []
        for i, (d, l) in enumerate(zip(self.decoder, self.lateral)):
            r = residuals[-(i+1)]
            x = d(x)

            features = l(x)
            f = features
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


