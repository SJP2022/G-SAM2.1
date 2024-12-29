# command: python shot_filter.py --gpus 8 --threshold 0.9
import argparse
import os
import random
import cv2
import pdb
import sys
import time
import numpy as np
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import logging
logging.set_verbosity_error()
import threading
from concurrent.futures import ThreadPoolExecutor
import yaml

config_path="paths.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
video_root = config["video_root"]
hog_shots_root = config["hog_shots_root"]
shots_root = config["shots_root"]

def train_on_gpu(gpu_id, vid):

    #判断文件是否存在
    shots_path = os.path.join(shots_root, vid+".json") #结果保存的路径
    if os.path.exists(shots_path):
        print("%s result exists." % vid) 
        return

    #线程开始
    print("%s started with gpu %d." % (vid, gpu_id)) 

    video_path = os.path.join(video_root, vid+".mp4") #视频文件路径
    hog_shots_path = os.path.join(hog_shots_root, vid+".json") #原分段路径
    if not os.path.exists(hog_shots_path):
        print("%s shots don't exist." % vid) 
        return
    
    with open(hog_shots_path, 'r', encoding='utf-8') as f:
        frame_idx = json.load(f)
    shots = []
    cap = cv2.VideoCapture(video_path)
    for idx in frame_idx:
        clip_features = []
        if idx==1:
            shots.append(idx)
            continue
        for n in [idx-2, idx-1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=image, return_tensors="pt").pixel_values
            inputs = inputs.to(torch.device(f'cuda:{gpu_id}'))
    
            with torch.no_grad():
                feat = model[gpu_id](inputs)['image_embeds']
                feat = F.normalize(feat, dim=-1)
                clip_features.append(feat.cpu().numpy())

        sim = clip_features[0] @ clip_features[1].T
        #print(idx, sim[0,0].item())
        if sim[0,0].item()<threshold: #0.9
            shots.append(idx)

    #print(shots)

    with open(shots_path, "w") as f:
        json.dump(shots, f)

    #线程结束
    print("%s finished with gpu %d: %s" % (vid, gpu_id, shots)) 

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=8, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument('--feature_extractor', default=config["clip_model_path"], help='Select the feature extractor model for video segmentation')
    args = parser.parse_args()

    # 定义要使用的GPU数量
    num_gpus = args.gpus
    threshold = args.threshold
    
    # 创建线程列表
    threads = []

    #每个gpu上部署一个model
    processor = CLIPProcessor.from_pretrained(args.feature_extractor)
    model = [CLIPVisionModelWithProjection.from_pretrained(args.feature_extractor).to(torch.device(f'cuda:{i}')) for i in range(num_gpus)]
    print('%d clip loaded.' % (len(model)))

    #为每个gpu创建一个线程池
    pool = [ThreadPoolExecutor(max_workers=1) for i in range(num_gpus)]
    print('%d pool built.' % (len(pool)))

    #遍历数据集
    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    random.shuffle(videos)
            
    #对每个视频创建处理线程
    for i, vid in enumerate(videos):
        gpu_id = i%num_gpus
        t = pool[gpu_id].submit(train_on_gpu, gpu_id, vid)
        threads.append(t)
            
    # 等待所有线程完成
    flag = True
    while flag:
        flag = False
        for t in threads:
            if not t.done():
                flag = True
    print('All subprocesses done.')