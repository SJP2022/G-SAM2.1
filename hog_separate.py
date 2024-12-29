import os
import math
import cv2
import numpy as np
import torch
import json
from decord import VideoReader
import argparse
from multiprocessing import Process, Queue
import random
from time import time

video_dir = "/share_io03_ssd/common2/videos/AVA/clips/trainval"
shot_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_shots"

#vid = "00SfeRtiM2o"
#vid  = "kMy-6RtoOVU"
#vid = "VsYPP2I0aUQ"
#vid = "zlVkeKC6Ha8"
vid = "kMy-6RtoOVU"

height = 360
width = 480

threshold = 0.35

start_time = 902
end_time = 1798 #1000 #1798

def solve(hog, videos):

    for i, video in enumerate(videos):
        print("%s starts." % video)
        video_root = os.path.join(video_dir, video)
        shots = {}
        for tid in range(start_time, end_time+1):
            video_path = os.path.join(video_root, str(tid)+".mp4")
            #vr = VideoReader(video_path)
            vr = VideoReader(video_path, height=height, width=width, num_threads=1)
            indices = [i for i in range(len(vr))]
            frames = vr.get_batch(indices)
            # T H W C
            if isinstance(frames, torch.Tensor):
                images = frames.numpy().astype(np.uint8)
            else:
                images = frames.asnumpy().astype(np.uint8)
            #print(type(images))
            print(video_path, images.shape)

            hog_features1 = 0 #当前帧
            hog_features2 = 0 #上一帧
            for i in range(images.shape[0]):
                hog_features2 = hog_features1
                image = images[i,:,:,:]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                hog_features1 = hog.compute(gray)
                if i==0:
                    continue
                hog_features_diff = cv2.compareHist(hog_features1, hog_features2, cv2.HISTCMP_CORREL)
                #print(type(hog_features_diff))
                #print(hog_features_diff)
                if abs(hog_features_diff)<threshold:
                    shots[tid] = i
                    print("%s finished with shots: %s" % (video_path, i))
                    break

        with open(os.path.join(shot_dir, video+".json"), "w", encoding='utf-8') as f:
            json.dump(shots, f)

        print("%s finished with shots: %s" % (video, shots))


# 计算HOG特征
winSize = (64, 64)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = cv2.HOGDescriptor_L2Hys
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradient = True
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins,
                        derivAperture, winSigma, histogramNormType, L2HysThreshold,
                        gammaCorrection, nlevels, signedGradient)


solve(hog, [vid])