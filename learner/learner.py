#! /usr/local/bin/python3

# E. Culurciello, February 2017
# Learner: learn new categories

import sys
import os
import time
import cv2 # http://seeb0h.github.io/howto/howto-install-homebrew-python-opencv-osx-el-capitan/
import numpy as np
from scipy.spatial import distance
import argparse
import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms

def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Learner Demo")
    # parser.add_argument('network', help='CNN model file')
    parser.add_argument('-i', '--input',  type=int, default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    return parser.parse_args()

# image pre-processing functions:
transformsImage = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # needed for pythorch ZOO models on ImageNet (stats)
    ])

print("Learner demo e-Lab")
print("keys 1-5 to learn new objects, esc to quit")

xres = 640
yres = 480
args = define_and_parse_args()
# categories = cat_file() # load category file

# setup camera input:
cam = cv2.VideoCapture(args.input)
cam.set(3, xres)
cam.set(4, yres)

# load CNN model:
# model = torch.load(args.network)
model = models.resnet18(pretrained=True)
# remove last fully-connected layer
model = nn.Sequential(*list(model.children())[:-1])
model.eval()
# print(model)

# remove last fc layer in ResNet definition also:
class ResNet(nn.Module):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

# some vars:
protos = np.zeros((512,5,1,1)) # array of previous templates
dists = np.zeros(5) # distance to protos

# main loop:
while True:
    startt = time.time()
    ret, frame = cam.read()
    if not ret:
        break

    if xres > yres:
        frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
    else:
        frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]
    
    pframe = cv2.resize(frame, dsize=(args.size, args.size))
    
    # prepare and normalize frame for processing:
    pframe = np.swapaxes(pframe, 0, 2)
    pframe = np.expand_dims(pframe, axis=0)
    pframe = transformsImage(pframe)
    pframe = torch.autograd.Variable(pframe) # turn Tensor to variable required for pytorch processing
    
    # process via CNN model:
    output = model(pframe)
    if output is None:
        print('no output from CNN model file')
        break

    output = output.data.numpy()[0] # get data from pytorch Variable, [0] = get vector from array
     
    # overlay on GUI frame
    # cv2.displayOverlay('win', text, 1000) # if Qt is present!
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(frame, text, (10, yres-20), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('win', frame)

    
    keyPressed = cv2.waitKey(1)
    if keyPressed == ord('1'):
        protos[:,0] = output
        print("Learned object 1")
    if keyPressed == ord('2'):
        protos[:,1] = output
        print("Learned object 2")
    if keyPressed == ord('3'):
        protos[:,2] = output
        print("Learned object 3")
    if keyPressed == ord('4'):
        protos[:,3] = output
        print("Learned object 4")
    if keyPressed == ord('5'):
        protos[:,4] = output
        print("Learned object 5")

    if keyPressed == 27: # ESC to stop
        break

    # compute distance between output and protos:
    for i in range(5):
        dists[i] = distance.euclidean( output, protos[:,i] )

    # print(dists)
    threshold = 1/3
    winner = np.argmin(dists)
    if dists[winner] < np.max(dists)*threshold:
        print("Detected proto", winner+1)

    # compute time and final info:
    endt = time.time()
    # sys.stdout.write("\r"+text+"fps: "+'{0:.2f}'.format(1/(endt-startt))) # text output 
    # sys.stdout.write("\rfps: "+'{0:.2f}'.format(1/(endt-startt)))
    sys.stdout.flush()

# end program:
cam.release()
cv2.destroyAllWindows()
