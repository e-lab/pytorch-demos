#! /usr/local/bin/python3

import sys
import os
import time
import cv2 # install cv3, python3:  http://seeb0h.github.io/howto/howto-install-homebrew-python-opencv-osx-el-capitan/
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
import argparse
import torch
import torchvision.transforms as transforms
import torch.legacy.nn as nn
#load_lua does not recognize SpatialConvolutionMM
nn.SpatialConvolutionMM = nn.SpatialConvolution
from torch.utils.serialization import load_lua

def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Visor Demo")
    parser.add_argument('model', help='model directory')
    parser.add_argument('-i', '--input',  default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    return parser.parse_args()


def cat_file():
    # load classes file
    categories = []
    try:
        f = open(args.model + '/categories.txt', 'r')
        for line in f:
            cat = line.split(',')[0].split('\n')[0]
            if cat != 'classes':
                categories.append(cat)
        f.close()
        print('Number of categories:', len(categories))
    except:
        print('Error opening file ' + args.model + '/categories.txt')
        quit()
    return categories

def patch(m):
    s = str(type(m))
    s = s[str.rfind(s, '.')+1:-2]
    if s == 'Padding' and hasattr(m, 'nInputDim') and m.nInputDim == 3:
        m.dim = m.dim + 1
    if s == 'View' and len(m.size) == 1:
        m.size = torch.Size([1,m.size[0]])
    if hasattr(m, 'modules'):
        for m in m.modules:
            patch(m)

print("Visor demo e-Lab - older Torch7 networks")
xres = 640
yres = 480
args = define_and_parse_args()
categories = cat_file() # load category file
print(categories)
font = cv2.FONT_HERSHEY_SIMPLEX

# setup camera input:
if args.input[0] >= '0' and args.input[0] <= '9':
    cam = cv2.VideoCapture(int(args.input))
    cam.set(3, xres)
    cam.set(4, yres)
    usecam = True
else:
    image = cv2.imread(args.input)
    xres = image.shape[1]
    yres = image.shape[0]
    usecam = False

# load old-pre-trained Torch7 CNN model:

# https://www.dropbox.com/sh/l0rurgbx4k6j2a3/AAA223WOrRRjpe9bzO8ecpEpa?dl=0
model = load_lua(args.model + '/model.net')
stat = load_lua(args.model + '/stat.t7')

#Patch Torch model to 4D
patch(model)

# image pre-processing functions:
transformsImage = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(stat.mean, stat.std)
    ])

while True:
    startt = time.time()
    if usecam:
        ret, frame = cam.read()
        if not ret:
            break
    else:
        frame = image.copy()

    if xres > yres:
        frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
    else:
        frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]
    
    pframe = cv2.resize(frame, dsize=(args.size, args.size))
    
    # prepare and normalize frame for processing:
    pframe = pframe[...,[2,1,0]]
    pframe = np.transpose(pframe, (2,0,1))
    pframe = transformsImage(pframe)
    pframe = pframe.view(1, pframe.size(0), pframe.size(1), pframe.size(2))
    
    # process via CNN model:
    output = model.forward(pframe)
    if output is None:
        print('no output from CNN model file')
        break

    # print(output)
    output = output.numpy()[0]

    # process output and print results:
    order = output.argsort()
    last = len(categories)-1
    text = ''
    for i in range(min(5, last+1)):
      text += categories[order[last-i]] + ' (' + '{0:.2f}'.format(output[order[last-i]]*100) + '%) '

    # overlay on GUI frame
    cv2.putText(frame, text, (10, yres-20), font, 0.5, (255, 255, 255), 1)
    cv2.imshow('win', frame)

    endt = time.time()
    # sys.stdout.write("\r"+text+"fps: "+'{0:.2f}'.format(1/(endt-startt))) # text output 
    sys.stdout.write("\rfps: "+'{0:.2f}'.format(1/(endt-startt)))
    sys.stdout.flush()
    
    if cv2.waitKey(1) == 27: # ESC to stop
        break

# end program:
if usecam:
    cam.release()
cv2.destroyAllWindows()
