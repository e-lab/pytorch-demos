
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18 as resnet
from torchvision.models import vgg19
from torch.autograd import Variable as B
import torch.nn.init as init

class SpatialDetector(nn.Module):

    def __init__(self, nclasses, npriors=1,
                 base_model_path='pretrained/alexnet-46.pth'):
        super().__init__()
        self.nclasses = nclasses
        self.npriors = npriors

        self.features_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.prob = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Conv2d(256, 4096, kernel_size=6),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Conv2d (4096, 4096, kernel_size=1),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Conv2d (4096, 1000, kernel_size=1),
            nn.ReLU (inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1000, nclasses, kernel_size=1)
        )
        self.bbox = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Conv2d (256, 4096, kernel_size=6),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Conv2d (4096, 4096, kernel_size=1),
            nn.ReLU (inplace=True),
            nn.Dropout(p = 0.5),
            nn.Conv2d (4096, 1000, kernel_size=1),
            nn.ReLU (inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(1000, 4, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        '''
        self.conv = nn.Sequential(
            nn.Conv2d(nout, 2*nout, 3),
            nn.BatchNorm2d(2*nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nout, 2*nout, 3),
            nn.BatchNorm2d(2*nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nout, 2*nout, 3, padding=1),
            nn.BatchNorm2d(2*nout),
            nn.ReLU(inplace=True),
        )
            
        self.prob = nn.Sequential(
            nn.Conv2d(2*nout, 2*nout, 3, padding=1),
            nn.BatchNorm2d(2*nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nout, nout, 3, padding=1),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nout, nclasses, 3, padding=1)
        )

        self.bbox = nn.Sequential(
            nn.Conv2d(2*nout, 2*nout, 3, padding=1),
            nn.BatchNorm2d(2*nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*nout, nout, 3, padding=1),
            nn.BatchNorm2d(nout),
            nn.ReLU(inplace=True),
            nn.Conv2d(nout, 4, 3, padding=1)
        )
        
        def xavier(param):
            nn.init.xavier_uniform(param)
            
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                xavier(m.weight.data)
                m.bias.data.zero_()


        self.conv.apply(weights_init)
        self.prob.apply(weights_init)
        self.bbox.apply(weights_init)
        '''
        
    def forward(self, x):
        x = self.features_head(x)
        p = self.prob(x)
        b = self.bbox(x)
        '''
        x = self.conv(x)
        p = self.prob(x)
        b = self.bbox(x)
        #p = p.view(-1, self.nclasses, p.size()[-2], p.size()[-1])
        #p = self.softmax(p)
        #p = p.view(-1, self.npriors*self.nclasses, p.size()[-2], p.size()[-1])
        '''
        return p, b

    '''
x = Detector(2)
print(x(B(torch.zeros(1,3,224,224))))'''
