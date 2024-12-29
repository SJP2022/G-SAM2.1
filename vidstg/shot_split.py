import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd
from scenedetect import AdaptiveDetector, detect

# detector = ContentDetector()

video_root = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/videos"
out_root = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/shots"
#video  = "zlVkeKC6Ha8"


def video_split(video):
    out_path = os.path.join(out_root, f"{video}.json")
    if os.path.exists(out_path):
        print(video, "exists.")
        return
    
    video_path = os.path.join(video_root, video+".mp4")
    start_tidx = 895
    detector = AdaptiveDetector(
        adaptive_threshold=3.0,
        # luma_only=True,
    )
    try:
        scene_list = detect(video_path, detector, start_in_scene=True)
        timestamp = [(s.get_seconds(), t.get_seconds()) for s, t in scene_list]
    except Exception as e:
        print(f"Video '{video}' with error {e}")
        return
    #shots = [start_tidx]
    #for (s, t) in timestamp:
    #    shots.append(math.ceil(t)+start_tidx)

    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(timestamp, f, indent=4)

    print(video, "finished.", len(timestamp))


videos_mp4 = os.listdir(video_root)
videos = [video[:-4] for video in videos_mp4]
#videos = ["2794976541"]
random.shuffle(videos)
for video in videos:
    video_split(video)
    