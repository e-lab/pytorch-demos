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
from scipy.spatial import distance
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
    parser.add_argument('-i', '--input', default='video.mp4', help='video file name', required=True)
    parser.add_argument('-s', '--size', type=int, default=224, help='network input size')
    parser.add_argument('--embfile', default='embeddings.npy', help='embedding file name')
    parser.add_argument('--save', default=False, help='saving similar frames to disk')
    parser.add_argument('--summarize', default=False, help='summarize a video file')
    parser.add_argument('--vst', type=float, default=0.1, help='video summarization threshold')
    parser.add_argument('--queryf', type=int, default=200, help='query frame number') # query frame to test localization in video
    return parser.parse_args()



def initModel():

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

  # load CNN model:
  model = models.resnet18(pretrained=True)
  # remove last fully-connected layer
  model = nn.Sequential(*list(model.children())[:-1])
  model.eval()
  # print(model)
  return model


def getFrameEmbedding(model, frame, xres, yres, newsize):

  # image pre-processing functions:
  transformsImage = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]) # needed for pythorch ZOO models on ImageNet (stats)
    ])

  if xres > yres:
    frame = frame[:,int((xres - yres)/2):int((xres+yres)/2),:]
  else:
    frame = frame[int((yres - xres)/2):int((yres+xres)/2),:,:]

  pframe = cv2.resize(frame, dsize=(newsize, newsize))

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


def openVideo(filename):
  cap = cv2.VideoCapture(filename)
  xres = cap.get(3)
  yres = cap.get(4)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print('This video has', frame_count, 'frames')
  return cap, frame_count, xres, yres


def createVideoEmbeddings(model, filename, newsize):
  print('Creating video embeddings, please wait...')
  cap, frame_count, xres, yres = openVideo(filename)

  # save embeddings in this:
  embeddings = np.zeros( (frame_count, 512) )

  for i in tqdm(range(frame_count-2)):
    ret, frame = cap.read()
    if not ret:
       break

    embeddings[i] = getFrameEmbedding(model, frame, xres, yres, newsize)
    
    cv2.imshow('video embeddings', frame)
    # time.sleep(0.25)

    # if cv2.waitKey(1) == 27: # ESC to stop
       # break

  # save embeddings to file:
  np.save(filename+'.emb', embeddings)
  cap.release()


def summarizeVideo(filename, threshold):
  print('Summarizing video, please wait...')
  video_summary = []
  cap, frame_count, xres, yres = openVideo(filename)

  # load embeddings:
  embeddings = np.load(filename+'.emb.npy')
  print('Loaded', embeddings.shape, 'embeddings')

  for i in tqdm(range(frame_count-2)):
    ret, frame = cap.read()
    if not ret:
       break
    output = embeddings[i]
    if i < 1:
      prevout = output
    else:
      d = distance.cosine(output, prevout)
      # print(d)
      if d > threshold:
        prevout = output
        cv2.imshow('video summarization', frame)
        cv2.waitKey(500)
        video_summary.append(i)

  # close:
  np.save(filename+'.sum', video_summary)
  cap.release()
  return video_summary


def getVideoFrame(filename, frame_num):
  cap, frame_count, xres, yres = openVideo(filename)
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
  ret, frame = cap.read()
  # close:
  cap.release()
  return frame


def localizeInVideo(model, filename, newsize, frame_query, num_neighbors, n_trees=20):
  cap, frame_count, xres, yres = openVideo(filename)

  # get embedding of query frame:  
  output = getFrameEmbedding(model, frame_query, xres, yres, newsize)

  # load embeddings:
  embeddings = np.load(filename+'.emb.npy')
  print('Loaded', embeddings.shape, 'embeddings')

  # prepare embedding search library:
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
  return frameNeigh, neighbors


def main():
  demo_name = "Video Localization demo e-Lab"
  print(demo_name)
  np.set_printoptions(precision=2)
  args = define_and_parse_args()

  # initialize neural network model to use:
  model = initModel() 

  # video_file = '/Users/eugenioculurciello/Code/datasets/automotive/ped360p-cut-10fps.mp4'
  video_file = args.input
  video_dir_name = os.path.dirname(video_file)
  video_file_name = os.path.basename(video_file)
  video_emb_file = video_file+'.emb.npy'
  
  if not Path(video_emb_file).is_file(): # delete embedding file if you want it to be re-created
    createVideoEmbeddings(model, video_file, args.size)

  if args.summarize:
    video_summary = summarizeVideo(video_file, args.vst) # summarize a video and get keyframes
    print('These', len(video_summary), 'frames are a summary of the video:', video_summary)
  else:
    query_frame_num = args.queryf
    frame_query = getVideoFrame(args.input, query_frame_num)
    num_neighbors = 10
    frameN, neighbors = localizeInVideo(model, video_file, args.size, frame_query, num_neighbors)

    # display similar frames in video:
    cv2.imshow("Query frame", frame_query)
    if args.save:
      cv2.imwrite('frame-query.jpg', frame_query)
    for i in range(len(frameN)):
      print('Showing', i, 'th similar frame to query image')
      cv2.imshow("Similar frames", frameN[i])
      cv2.waitKey(500)
      if args.save:
        cv2.imwrite('frame_sim'+str(i)+'.jpg', frameN[i])
  

  # cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
