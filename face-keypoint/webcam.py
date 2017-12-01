import cv2
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()
args.size = 512

cap = cv2.VideoCapture(0)
xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
tensor = transforms.ToTensor()

model = torch.load(args.model, map_location=lambda storage, loc: storage)
model = model.module
model = model.cuda()

while(True):

    ret, frame = cap.read()
    if not ret: break

    #if xres > yres:
    #    frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
    #else:
    #    frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]
    #frame = cv2.resize(frame, dsize=(args.size, args.size))
        
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = tensor(img)
    frame = img.permute(1,2,0).contiguous().numpy()
    #img = normalize(img)
    
    pred = model(V(img.unsqueeze(0).cuda()))[-1][0].cpu()
    
    pred = pred.data.max(0)[0].numpy()
    #pred = cv2.resize(pred, dsize=(args.size, args.size))
    
    pred = cv2.resize(pred, dsize=(xres, yres), interpolation=cv2.INTER_CUBIC)

    frame[:,:,1] += pred
    
    cv2.imshow('frame', frame)
    cv2.imshow('pred', pred)
    cv2.waitKey(1)
    #if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.release()
cv2.destroyAllWindows()
