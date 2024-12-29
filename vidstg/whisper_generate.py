# command: python whisper_generate.py --gpus 8
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import json
import os, time

import whisper
import whisperx
import torch

video_root = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/videos"
ASR_root = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/asr"

if not os.path.exists(ASR_root):
    os.makedirs(ASR_root)

def video2asr(vid):

    #判断文件是否存在
    ASR_path = os.path.join(ASR_root, vid+".json") #转录结果保存的路径
    if os.path.exists(ASR_path):
        print("%s exists." % vid) 
        return

    #线程开始
    print("%s started." % (vid)) 

    video_path = os.path.join(video_root, vid+".mp4") #视频文件路径
    #whisper转录
    asr = model.transcribe(video_path, task="translate") 
    audio = whisperx.load_audio(video_path)
    aligned_asr = whisperx.align(asr["segments"], align_model, metadata, audio, device, return_char_alignments=False)
    #保存转录结果
    with open(ASR_path, 'w') as f: 
        json.dump(aligned_asr, f)

    #线程结束
    print("%s finished." % (vid)) 

if __name__=='__main__':

    MODEL_DIR = "TOFILL"
    device = torch.device('cuda:0')
    model = whisper.load_model("turbo", device)
    align_model, metadata = whisperx.load_align_model(language_code='en', device=device, model_dir=MODEL_DIR)
    
    videos = os.listdir(video_root)
    for i in range(len(videos)):
        if videos[i].endswith(".mp4"):
            videos[i] = videos[i][:-4]
    random.shuffle(videos)
            
    for vid in videos:
        video2asr(vid)