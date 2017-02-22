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
from torch.utils.serialization import load_lua
from torch.legacy import nn # import torch.nn as nn


def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Visor Demo")
    # parser.add_argument('network', help='CNN model file')
    parser.add_argument('categories', help='text file with categories')
    parser.add_argument('-i', '--input',  type=int, default='0', help='camera device index or file name, default 0')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    # parser.add_argument('-S', '--stat', help='stat.txt file')
    return parser.parse_args()


def cat_file():
    # load classes file
    categories = []
    if hasattr(args, 'categories') and args.categories:
        try:
            f = open(args.categories, 'r')
            for line in f:
                cat = line.split(',')[0].split('\n')[0]
                if cat != 'classes':
                    categories.append(cat)
            f.close()
            print('Number of categories:', len(categories))
        except:
            print('Error opening file ' + args.categories)
            quit()
    return categories


print("Visor demo e-Lab - older Torch7 networks")
xres = 640
yres = 480
args = define_and_parse_args()
categories = cat_file() # load category file
print(categories)


# setup camera input:
cam = cv2.VideoCapture(args.input)
cam.set(3, xres)
cam.set(4, yres)

# load old-pre-trained Torch7 CNN model:

# https://www.dropbox.com/sh/l0rurgbx4k6j2a3/AAA223WOrRRjpe9bzO8ecpEpa?dl=0
model = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-alexowt-46/model.net')
stat = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-alexowt-46/stat.t7')
model.modules[13] = nn.View(1,9216)

# https://www.dropbox.com/sh/xcm8xul3udwo72o/AAC8RChVSOmgN61nQ0cyfdava?dl=0
# model = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-alextiny-46/model.net')
# stat = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-alextiny-46/stat.t7')
# model.modules[13] = nn.View(1,64)

# https://www.dropbox.com/sh/anklohs9g49z1o4/AAChA9rl0FEGixT75eT38Dqra?dl=0
# model = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-enet-demo-46/model.net')
# stat = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/elab-enet-demo-46/stat.t7')
# model.modules[41] = nn.View(1,1024)

# https://www.dropbox.com/sh/s0hwugnmhwkk9ow/AAD_abZ2LOav9GeMETt5VGvGa?dl=0
# model = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/enet128-demo-46/model.net')
# stat = load_lua('/Users/eugenioculurciello/Dropbox/shared/models/enet128-demo-46/stat.t7')
# model.modules[32] = nn.View(1,512)

# print(model)
# this now should work:
# model.forward(torch.Tensor(1,3,224,224)) # test

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
    # cv2.displayOverlay('win', text, 1000) # if Qt is present!
    font = cv2.FONT_HERSHEY_SIMPLEX
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
