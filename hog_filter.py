# command: python hog_filter.py --process 8
import argparse
import os
from multiprocessing import Process, Queue
import random
from time import time
import cv2
import numpy as np
import torch
import json
from decord import VideoReader

video_root = "/share/common/VideoDatasets/ActivityNet/videos"
hog_shots_root = "/share/test/shijiapeng/ActivityNet_shots_HOG_filter"
shots_root = "/share/test/shijiapeng/ActivityNet_shots_HOG_final"

if not os.path.exists(shots_root):
    os.makedirs(shots_root)

def solve(process_id, videos):
    start_time = time()
    check = 10

    for i, video in enumerate(videos):
        vid = video[:-4]
        hog_shots_path = os.path.join(hog_shots_root, vid+".json")
        json_path = os.path.join(shots_root, vid+".json")
        if not os.path.exists(hog_shots_path):
            print("%s shots don't exist." % vid) 
            continue
        if os.path.exists(json_path):
            print("%s exists." % video)
            continue
        
        print("%s starts." % video)

        video_path = os.path.join(video_root, video)
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()

        with open(hog_shots_path, 'r', encoding='utf-8') as f:
            frame_idx = json.load(f)
        frame_idx.append(len(vr))
        shots = []
        last_idx = frame_idx[:]
        point = -1
        
        for cnt, idx in enumerate(frame_idx):
            if idx==1:
                point = 0
            elif idx-last_idx[point]<fps: #merge
                if len(shots)>0:
                    point = cnt
            else: #note and separate
                #print(last_idx[point], idx)
                shots.append(last_idx[point])
                point = cnt
        if last_idx[point]!=len(vr):
            print("%s failed with shots: %s, fps = %s" % (video, shots, fps))
        else:
            print("%s finished with shots: %s, fps = %s" % (video, shots, fps))

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
    args = parser.parse_args()

    num_process = args.process
    sub_video_list = []
    videos = os.listdir(video_root)
    #videos = ["00SfeRtiM2o.mp4", "00S8I27qDU4.mp4", "0_1BQPWzRiw.mp4"]
    #videos = ["0_1BQPWzRiw.mp4"]
    n = len(videos)
    step = n // num_process
    j = 0

    random.shuffle(videos)
    for i in range(0, n, step):
        j += 1
        if j == num_process:
            sub_video_list.append(videos[i:n])
            break
        else:
            sub_video_list.append(videos[i : i + step])

    process_list = []
    Q = Queue()
    for i, item in enumerate(sub_video_list):
        cur_process = Process(target=solve, args=(i, item))
        process_list.append(cur_process)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
