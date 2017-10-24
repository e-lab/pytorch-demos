import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as resnet
from torchvision.models import vgg19
from torch.autograd import Variable as B
import torch.nn.init as init

class Detector(nn.Module):

    def __init__(self, nclasses, npriors=1,
                 base_model_path='pretrained/alexnet-46.pth'):
        super().__init__()
        self.nclasses = nclasses
        self.npriors = npriors

        nout  = 256
        model = torch.load(base_model_path)['model_def']

        m = []
        for i in model.features.children():
            m.append(i)
            if type(i) is torch.nn.Conv2d:
                m.append(torch.nn.BatchNorm2d(i.out_channels))
        self.features_head = nn.Sequential(*m)

        self.prob = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear (9216, 512),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear (512, 512),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, nclasses)
        )
        self.bbox = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear (9216, 512),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear (512, 512),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 4),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        input_size = list(x.size())
        x = self.features_head(x)
        x = x.view(x.size()[0], -1)
        p = self.prob(x)
        b = self.bbox(x)

        return p, b
