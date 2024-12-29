import argparse
import json
import math
import os
import random

import numpy as np
import pandas as pd

# detector = ContentDetector()

ava_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_videos"
video_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_videos"
out_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_shots"
vid_path = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/vid.json"
#video  = "zlVkeKC6Ha8"

vid_list = []
videos = os.listdir(ava_root)
random.shuffle(videos)
for video in videos:
    video = video[:-4]
    out_path = os.path.join(out_root, f"{video}.json")
    with open(out_path, "r", encoding='utf-8') as f:
        out_data = json.load(f)
    if len(out_data)==1:
        vid_list.append(video)
with open(vid_path, "w", encoding='utf-8') as f:
    json.dump(vid_list, f, indent=4)

print(len(vid_list))