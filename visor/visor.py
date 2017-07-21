#! /usr/local/bin/python3

import sys
import os
import time
import cv2 # install cv3, python3: brew install opencv3 --with-contrib --with-python3 --without-python
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
import argparse
import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms


def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Visor Demo")
    parser.add_argument('model', help='model directory')
    parser.add_argument('-i', '--input', default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('-t7', '--torch7', type=bool, default=False, help='Torch7 network or PyTorch (default)?')
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

print("Visor demo e-Lab")
xres = 640
yres = 480
args = define_and_parse_args()
categories = cat_file() # load category file
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

# load CNN moodels:
if args.torch7:
    print('Importing Torch7 model')
    import torch.legacy.nn as nn
    #load_lua does not recognize SpatialConvolutionMM
    nn.SpatialConvolutionMM = nn.SpatialConvolution
    from torch.utils.serialization import load_lua
    model = load_lua(args.model + '/model.net')
    stat = load_lua(args.model + '/stat.t7')
    patch(model)
else:
    # model = torch.load(args.network)
    model = models.resnet18(pretrained=True)
    model.eval()
    # print model
    softMax = nn.Softmax() # to get probabilities out of CNN


# image pre-processing functions:
transformsImage = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # needed for pythorch ZOO models on ImageNet (stats)
    ])

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
    if args.torch7:
        pframe = pframe[...,[2,1,0]]
        pframe = transformsImage(pframe)
        pframe = pframe.view(1, pframe.size(0), pframe.size(1), pframe.size(2))
        # process via CNN model:
        output = model.forward(pframe)
        output = output.numpy()[0]

    else:
        pframe = transformsImage(pframe)
        pframe = torch.autograd.Variable(pframe) # turn Tensor to variable required for pytorch processing
        pframe = pframe.unsqueeze(0)
        # process via CNN model:
        output = model(pframe)
        output = softMax(output) # convert CNN output to probabilities
        output = output.data.numpy()[0] # get data from pytorch Variable, [0] = get vector from array

    if output is None:
        print('no output from CNN model file')
        break
    
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
cam.release()
cv2.destroyAllWindows()
