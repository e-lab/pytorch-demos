import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms

import argparse
import numpy as np
import cv2

from models.spatial_detector import SpatialDetector
from utils.nms import nms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def spatialize(old, new):
    for i, j in zip(old.modules(), new.modules()):
        if not list(i.children()):
            if isinstance(i, nn.Linear):
                j.weight.data = i.weight.data.view(j.weight.size())
                j.bias.data = i.bias.data
            else:
                if len(i.state_dict()) > 0:
                    j.weight.data = i.weight.data
                    j.bias.data = i.bias.data

def parse_args():
    parser = argparse.ArgumentParser(description="Person Detection Demo")
    parser.add_argument('model', help='model path')
    parser.add_argument('-i', '--input', default='0', help='camera device index, default 0')
    parser.add_argument('-hd', type=bool, default=True, help='process full frame or resize to net eye size only')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='detection threshold')
    return parser.parse_args()

args = parse_args()
font = cv2.FONT_HERSHEY_PLAIN

model = torch.load(args.model)
s_model = SpatialDetector(2)
spatialize(model, s_model)
model = s_model
model.eval()

print(model)

# setup camera input:
cam = cv2.VideoCapture(int(args.input))
xres = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('camera width, height:', xres, ' x ', yres)

while(True):

    boxes = []

    # Capture frame-by-frame
    ret, frame = cam.read()

    s = time.time()

    if not args.hd:
        if xres > yres:
            frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
        else:
            frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]
        frame = cv2.resize(frame, dsize=(args.size, args.size))

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255
    img = img.permute(2,0,1).contiguous()
    img = normalize(img)
    _, ih, iw = img.size()
    p, b = model(V(img.unsqueeze(0)))
    p = p.squeeze(0).data
    b = b.squeeze(0).data
    c = p.max(0)[1]

    prob = F.softmax(p)[1].data
    for i in range(c.size()[0]):
        for j in range(c.size()[1]):
            if c[i,j] == 0: continue
            if prob[i,j] < args.threshold: continue
            x,y,w,h = b.numpy()[:,i,j]
            x = x * 224 + j* 32
            y = y * 224 + i* 32
            w = w * ih#iw
            h = h * iw#224#ih
            x1 = x
            y1 = y
            x2 = x1 + w
            y2 = y1 + h
            boxes.append([x1,y1,x2,y2,prob[i,j]])

    for (x1, y1, x2, y2, p) in boxes:
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 1, cv2.LINE_AA)

    boxes = nms(boxes, 0.3)

    for (x1, y1, x2, y2, p) in boxes:
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, str(round(p,3)), (int(x1), int(y1)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)

    e = time.time() - s
    frame = cv2.putText(frame, str(int(1/e))+' fps', (int(0), int(100)), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, releasecv2.resize(frame, (224,224)) the capture
cap.release()
cv2.destroyAllWindows()
