#! /usr/local/bin/python3

# Localization of frames in a video. Used to spatially localize where you are in a video given an input frame.
# this can be used in augmented realty applications to localize a user in one of multiple locations
# This code takes an input video and computes embedding for every frames of the video.
# you can then input a frame of another or same video, and it will tell you which frames are most similar to the one you presented.
#
# run as: python3 vloc.py -i your_video.mp4
#
# E. Culurciello, March 2017
#

import sys
import os
import time
import cv2 # install cv3, python3:  http://seeb0h.github.io/howto/howto-install-homebrew-python-opencv-osx-el-capitan/
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torchvision.models as models 
import torchvision.transforms as transforms
from annoy import AnnoyIndex # https://github.com/spotify/annoy


def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Video location Demo")
    parser.add_argument('-i', '--input', default='video.mp4', help='video file name')
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('--embfile', default='embeddings.npy', help='embedding file name')
    return parser.parse_args()


print("Video Localization demo e-Lab")
np.set_printoptions(precision=2)
args = define_and_parse_args()

# image pre-processing functions:
transformsImage = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]) # needed for pythorch ZOO models on ImageNet (stats)
    ])

# load CNN model:
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


def getFrameEmbedding(frame, xres, yres):

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

   return (output.data.numpy()[0]).reshape(512) # get data from pytorch Variable, [0] = get vector from array


def createVideoEmbeddings(filename):
  print('Creating video embeddings, please wait...')
  cap = cv2.VideoCapture(filename)
  xres = cap.get(3)
  yres = cap.get(4)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('This video has', frame_count, 'frames')

  # save embeddings in this:
  embeddings = np.zeros( (frame_count, 512) )

  for i in tqdm(range(frame_count)):
    ret, frame = cap.read()
    if not ret:
       break

    embeddings[i] = getFrameEmbedding(frame, xres, yres)
    
    cv2.imshow('frame', frame)
    # time.sleep(0.25)

    # if cv2.waitKey(1) == 27: # ESC to stop
       # break

  # save embeddings to file:
  np.save(args.embfile, embeddings)
  cap.release()

def getVideoFrame(filename, frame_num):
  cap = cv2.VideoCapture(filename)
  xres = cap.get(3)
  yres = cap.get(4)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('This video has', frame_count, 'frames')
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
  ret, frame = cap.read()
  # close:
  cap.release()
  return frame


def localizeInVideo(filename, frame_query, num_neighbors, n_trees=20):
  cap = cv2.VideoCapture(filename)
  xres = cap.get(3)
  yres = cap.get(4)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('This video has', frame_count, 'frames')

  # get embedding of query frame:  
  output = getFrameEmbedding(frame_query, xres, yres)

  # load embeddings:
  embeddings = np.load(args.embfile)
  print('Loaded', embeddings.shape, 'embeddings')

  # using Annoy library: https://github.com/spotify/annoy
  a = AnnoyIndex(embeddings.shape[1], metric='angular')
  for i in range(embeddings.shape[0]):
    a.add_item(i, embeddings[i])
  a.build(n_trees)
  neighbors = a.get_nns_by_vector(output, num_neighbors, search_k=-1, include_distances=False)
  print('Frame list of', num_neighbors, 'neighbors:', neighbors)
  
  # display results:
  frameNeigh = [] # table of frames
  for i in range(num_neighbors):
    cap.set(cv2.CAP_PROP_POS_FRAMES, neighbors[i])
    ret2, frame2 = cap.read()
    frameNeigh.append(frame2)

  # close:
  cap.release()
  return frameNeigh


def main():
  # video_file = '/Users/eugenioculurciello/Code/datasets/automotive/ped360p-cut-10fps.mp4'
  video_file = args.input
  if not Path(args.embfile).is_file(): # delete embedding file if you want it to be re-created
    createVideoEmbeddings(video_file)

  frame_num = 200
  frame_query = getVideoFrame(args.input, frame_num)
  num_neighbors = 10
  frameN = localizeInVideo(video_file, frame_query, num_neighbors)

  # display similar frames in video:
  cv2.imshow("Query frame", frame_query)
  for i in range(len(frameN)):
    print('Showing', i, 'th similar frame to query image')
    cv2.imshow("Similar frames", frameN[i])
    cv2.waitKey(500)
  

  # cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
