# command: python frame_extract.py --process 8
import argparse
import json
import os
from multiprocessing import Process, Queue
from time import time

vid_path = "3418738633.json"
with open(vid_path, "r", encoding='utf-8') as f:
    vid_list = json.load(f)
print(len(vid_list["trajectories"]))
cnt = 0
for track in vid_list["trajectories"]:
    for obj in track:
        if obj["generated"]==1:
            cnt += 1
            break
print(cnt)
with open("3418738633_indent.json", "w", encoding='utf-8') as f:
    json.dump(vid_list, f, indent=4)
