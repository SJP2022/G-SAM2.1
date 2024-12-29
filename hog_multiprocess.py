# command: python hog_separate.py --process 8 --height 360 --width 480 --threshold 0.35
import argparse
import yaml
import os
from multiprocessing import Process, Queue
from time import time
import cv2
import numpy as np
import torch
import json
from decord import VideoReader

config_path="paths.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
video_root = config["video_root"]
hog_shots_root = config["hog_shots_root"]

if not os.path.exists(hog_shots_root):
    os.makedirs(hog_shots_root)

def solve(hog, process_id, videos):
    start_time = time()
    check = 10

    for i, video in enumerate(videos):
        vid = video[:-4]
        json_path = os.path.join(hog_shots_root, vid+".json")
        if os.path.exists(json_path):
            print("%s exists." % video)
            continue

        print("%s starts." % video)
        video_path = os.path.join(video_root, video)
        #vr = VideoReader(video_path)
        vr = VideoReader(video_path, height=height, width=width)
        indices = [i for i in range(len(vr))]
        frames = vr.get_batch(indices)
        # T H W C
        if isinstance(frames, torch.Tensor):
            images = frames.numpy().astype(np.uint8)
        else:
            images = frames.asnumpy().astype(np.uint8)
        #print(type(images))
        print(video, images.shape)
        
        shots = []

        hog_features1 = 0 #当前帧
        hog_features2 = 0 #上一帧
        for i in range(images.shape[0]):
            hog_features2 = hog_features1
            image = images[i,:,:,:]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hog_features1 = hog.compute(gray)
            if i==0:
                shots.append(i+1)
                continue
            hog_features_diff = cv2.compareHist(hog_features1, hog_features2, cv2.HISTCMP_CORREL)
            #print(type(hog_features_diff))
            #print(hog_features_diff)
            if abs(hog_features_diff)<threshold:
                shots.append(i+1)
                #print(i+1, hog_features_diff)

        print("%s finished with shots: %s" % (video, shots))

        with open(json_path, "w") as f:
            json.dump(shots, f)

        # print(cmd)
        if i % check == check - 1:
            ET = time() - start_time
            ETA = ET / (i + 1) * (len(videos) - i - 1)
            print(
                "process:%d %d/%d ET: %.2fmin ETA:%.2fmin "
                % (process_id, i + 1, len(videos), ET / 60, ETA / 60)
            )

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', default=8, type=int)
    parser.add_argument('--height', default=360, type=int)
    parser.add_argument('--width', default=480, type=int)
    parser.add_argument('--threshold', default=0.35, type=float)
    args = parser.parse_args()

    num_process = args.process
    height = args.height #360
    width = args.width #480
    threshold = args.threshold #0.1
    sub_video_list = []
    videos = os.listdir(video_root)
    n = len(videos)
    step = n // num_process
    j = 0

    # random.shuffle(video_list)
    for i in range(0, n, step):
        j += 1
        if j == num_process:
            sub_video_list.append(videos[i:n])
            break
        else:
            sub_video_list.append(videos[i : i + step])

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

    process_list = []
    Q = Queue()
    for i, item in enumerate(sub_video_list):
        cur_process = Process(target=solve, args=(hog, i, item))
        process_list.append(cur_process)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
