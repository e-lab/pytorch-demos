#! /usr/local/bin/python3

# E. Culurciello, February 2017
# Learner: learn new categories

import sys
import os
import time
import cv2 # install cv3, python3:  brew install opencv3 --with-contrib --with-python3 --without-python
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
from scipy.spatial import distance
import argparse
import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import tqdm # progress bar

def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Learner Demo")
    # parser.add_argument('network', help='CNN model file')
    parser.add_argument('-i', '--input', type=int, default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='detection threshold')
    parser.add_argument('-v', '--variance', type=bool, default=False, help='get variance for testset') # used to compute a variance matrix of representation values from a test dataset
    parser.add_argument('-p', '--path', type=str, default='', help='path for variance testset')
    parser.add_argument('-u', '--usevar', type=bool, default=False, help='use variance for distance calculations')
    return parser.parse_args()

# image pre-processing functions:
transformsImage = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]) # needed for pythorch ZOO models on ImageNet (stats)
    ])

print("Learner demo e-Lab")
print("keys 1-5 to learn new objects, esc to quit")

np.set_printoptions(precision=2)

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

def computeOutVar(path):
    valdir = os.path.join(path, 'val')
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=1, shuffle=True,
        num_workers=1, pin_memory=True)
    print('Loaded', len(val_loader), 'test images!')

    outs = np.zeros((512,len(val_loader),1,1))

    for i, (input, target) in tqdm(enumerate(val_loader)):
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        outs[:,i] = output.data.numpy()[0] # get data from pytorch Variable, [0] = get vector from array

    outvar = np.var(outs,1)

    np.save('outvar.npy', outvar)
    print('Variance computed and saved to "outvar.npy" file')
    return outvar

# run program:    
if args.variance:
    outvar = computeOutVar(args.path)
else: 
    if args.usevar: 
        outvar = np.load('outvar.npy')

# some vars:
protos = np.ones((512,5,1,1)) # array of previous templates
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
    pframe = transformsImage(pframe)
    pframe = torch.autograd.Variable(pframe) # turn Tensor to variable required for pytorch processing
    pframe = pframe.unsqueeze(0)
    
    # process via CNN model:
    output = model(pframe)
    if output is None:
        print('no output from CNN model file')
        break

    output = output.data.numpy()[0] # get data from pytorch Variable, [0] = get vector from array

    # detect key presses:
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
    if args.usevar:
        for i in range(5):
            dists[i] = distance.seuclidean( output, protos[:,i], outvar ) # uses a testset variance for better distance computations
    else:
        for i in range(5):
            dists[i] = distance.cosine( output, protos[:,i] )
    # print(dists)
    
    winner = np.argmin(dists)
    text2 = ""
    if dists[winner] < np.max(dists)*args.threshold:
        text2 = " / Detected: " + str(winner+1)

    # compute time and final info:
    endt = time.time()
    text = "fps: "+'{0:.2f}'.format(1/(endt-startt)) + text2

    # overlay on GUI frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (10, yres-20), font, 1, (255, 255, 255), 2)
    cv2.imshow('win', frame)

# end program:
cam.release()
cv2.destroyAllWindows()
