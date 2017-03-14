#! /usr/local/bin/python3

# Demo of: Localization of frames in a video.
# please run vloc on video to get embedding file ready first!
#
# run as: python3 demo.py -c caption.srt -i video.mp4
#
# E. Culurciello, March 2017
#

import os
from pathlib import Path
import argparse
import vloc
import re
import datetime
import time
import cv2 # install cv3, python3:  http://seeb0h.github.io/howto/howto-install-homebrew-python-opencv-osx-el-capitan/
# add to profile: export PYTHONPATH=$PYTHONPATH:/usr/local/Cellar/opencv3/3.2.0/lib/python3.6/site-packages/
import numpy as np
from annoy import AnnoyIndex # https://github.com/spotify/annoy


def define_and_parse_args():
    # argument Checking
    parser = argparse.ArgumentParser(description="Demo captioning for Video location")
    parser.add_argument('-i', '--input', default='video.mp4', help='video file name', required=True)
    parser.add_argument('-c', '--captionfile', default='video.srt', help='caption file name', required=True)
    parser.add_argument('--size', type=int, default=224, help='network input size')
    parser.add_argument('-s', '--save', type=int, default=False, help='save video file of demo')
    parser.add_argument('--outfilename', default='output.avi', help='network input size')
    return parser.parse_args()


def main():
   demo_name = "Demo captioning for Video Localization"
   print(demo_name)
   np.set_printoptions(precision=2)
   args = define_and_parse_args()
   font = cv2.FONT_HERSHEY_SIMPLEX

   # load caption file:
   f=open(args.captionfile)
   content = f.read()
   captions = re.findall("(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\s+(.+)", content)
   # captions[x][0], captions[x][1] is times, captions[x][2] is caption text

   def getCaptionData(idx):
      x = time.strptime(captions[idx][0].split(',')[0],'%H:%M:%S')
      t1 = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
      x = time.strptime(captions[idx][1].split(',')[0],'%H:%M:%S')
      t2 = datetime.timedelta(hours=x.tm_hour,minutes=x.tm_min,seconds=x.tm_sec).total_seconds()
      textc = captions[idx][2]
      return (t1,t2,textc)

   # testing:
   # print("Captions are:")
   # for i in range(len(captions)):
   #    t1,t2,textc = getCaptionData(i)
   #    print(t1, t2, textc)


   # initialize neural network model to use:
   model = vloc.initModel() 

   video_file = args.input
   video_dir_name = os.path.dirname(video_file)
   video_file_name = os.path.basename(video_file)
   video_emb_file = video_file+'.emb.npy'

   cap, frame_count, xres, yres = vloc.openVideo(video_file)
   xres = cap.get(3)
   yres = cap.get(4)
   fourcc = cv2.VideoWriter_fourcc(*'XVID')
   out = cv2.VideoWriter(args.outfilename, fourcc, 20.0, (int(xres),int(yres))) # to write a movie output!
   fps = cap.get(cv2.CAP_PROP_FPS)
   print("Frames per second:", fps)

   # load embeddings:
   embeddings = np.load(video_file+'.emb.npy')
   print('Loaded', embeddings.shape, 'embeddings')
   
   # prepare embedding search library:
   # using Annoy library: https://github.com/spotify/annoy
   num_neighbors = 4
   n_trees = 20
   a = AnnoyIndex(embeddings.shape[1], metric='angular')
   for i in range(embeddings.shape[0]):
      a.add_item(i, embeddings[i])
   
   a.build(n_trees)

   # start processing:
   j = 0 # index in caption file
   for i in range(frame_count-2):
      ret, frame = cap.read()
      if i < 1:
         t1,t2,textc = getCaptionData(j)

      # get embedding of query frame (i):  
      output = vloc.getFrameEmbedding(model, frame, xres, yres, args.size)

      # get where is this frame in the video file?
      neighbors = a.get_nns_by_vector(output, num_neighbors, search_k=-1, include_distances=False)
      # print('Frame list of', num_neighbors, 'neighbors:', neighbors)

      # get appropriate caption based on recognized frame position in video
      tn = neighbors[0] * 1/fps # get time in seconds of this matched frame
      if not(tn >= t1 and tn < t2):
         j += 1
         t1,t2,textc = getCaptionData(j) # get next caption item
      # print(i, neighbors[0], j, t1, tn, t2, textc)

      # overlay on GUI frame
      cv2.putText(frame, textc, (10, int(yres-20)), font, 1, (255, 255, 255), 2)
      cv2.imshow(demo_name, frame)
      # cv2.imwrite('frame.jpg', frame)
      out.write(frame)
      cv2.waitKey(1)

   # close:
   cap.release()
   out.release()

   # cv2.destroyAllWindows()

if __name__ == "__main__":
  main()