# command: python KTS.py --gpus 8
import argparse

import os
import cv2
import pdb
import sys
import time
import numpy as np
import json
from transformers import logging
logging.set_verbosity_error()

from projects.VLog.models.kts_model import VideoSegmentor
from projects.VLog.models.clip_model import FeatureExtractor
from projects.VLog.utils.utils import logger_creator, format_time

import random
import threading
from concurrent.futures import ThreadPoolExecutor
import torch

videos_dir = "/share/common/VideoDatasets/ActivityNet/videos"
shot_dir = "/share/test/shijiapeng/ActivityNet_shots_KTS"
os.makedirs(shot_dir, exist_ok=True)

def train_on_gpu(gpu_id, vid):

    #判断文件是否存在
    shot_path = os.path.join(shot_dir, vid+".json") #转录结果保存的路径
    if os.path.exists(shot_path):
        print("%s exists." % vid) 
        return

    #线程开始
    print("%s started with gpu %d at %s." % (vid, gpu_id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))) 

    feature_extractor = extractors[gpu_id]
    video_segmenter = segmenters[gpu_id]

    video_path = os.path.join(videos_dir, vid+".mp4") #视频文件路径
    clip_features, video_length = feature_extractor(video_path, vid, save=False)
    seg_windows = video_segmenter(clip_features, video_length)

    shots = seg_windows

    print(vid, shots)

    json_path = os.path.join(shot_dir, vid+".json")
    with open(json_path, "w") as f:
        json.dump(shots, f)

    #线程结束
    print("%s finished with gpu %d  at %s." % (vid, gpu_id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='/vhome/shijiapeng/BenchV/demo_videos/00SfeRtiM2o.mp4')
    parser.add_argument('--alpha', default=1, type=int, help='Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.')
    parser.add_argument('--beta', default=1, type=int, help='The smallest time gap between successive clips, in seconds.')
    parser.add_argument('--data_dir', default='./examples', type=str, help='Directory for saving videos and logs.')
    parser.add_argument('--tmp_dir', default='./tmp', type=str, help='Directory for saving intermediate files.')
    
    # * Models settings *
    parser.add_argument('--openai_api_key', default='xxx', type=str, help='OpenAI API key')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--feature_extractor', default='openai/clip-vit-base-patch32', help='Select the feature extractor model for video segmentation')
    parser.add_argument('--feature_extractor_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu')
    parser.add_argument('--image_captioner', choices=['blip2-opt', 'blip2-flan-t5', 'blip'], dest='captioner_base_model', default='blip2-opt', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--image_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--audio_translator', default='large')
    parser.add_argument('--audio_translator_device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo'], default='gpt-3.5-turbo')
    parser.add_argument('--gpus', default=8, type=int)
    args = parser.parse_args()

    # 定义要使用的GPU数量
    num_gpus = args.gpus

    #feature_extractor = FeatureExtractor(args)
    #video_segmenter = VideoSegmentor(alpha=args.alpha, beta=args.beta)
    extractors = []
    for i in range(num_gpus):
        args.feature_extractor_device = torch.device(f'cuda:{i}')
        extractors.append(FeatureExtractor(args))
    segmenters = [VideoSegmentor(alpha=args.alpha, beta=args.beta) for i in range(num_gpus)]
    print('%d models loaded.' % (len(extractors)))
    threads = []
    pool = [ThreadPoolExecutor(max_workers=1) for i in range(num_gpus)]
    print('%d pool built.' % (len(pool)))

    #videos = {"xukun.mp4"}
    videos = os.listdir(videos_dir)
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