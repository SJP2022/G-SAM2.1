import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images
from utils.common_utils import CommonUtils
from utils.mask_dictionary_model import MaskDictionaryModel, ObjectInfo
import json
import copy
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import pycocotools.mask as mask_util
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP

# This demo shows the continuous object tracking plus reverse tracking with Grounding DINO and SAM 2
"""
Step 1: Environment settings and model initialization
"""
# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "/share_io03_ssd/common2/videos/AVA/clips/trainval"
#video_dir = "assets"
#annotation_dir = "assets"
frame_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_frames"
# 'output_dir' is the directory to save the annotated frames
output_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_tracking"

#vid = "00SfeRtiM2o"
#vid  = "kMy-6RtoOVU"
#vid = "VsYPP2I0aUQ"
#vid = "zlVkeKC6Ha8"
#vid = "kplbKz3_fZk"
vid = "kMy-6RtoOVU"
video_path = os.path.join(video_dir, vid)
frame_path = os.path.join(frame_dir, vid+"_full")
output_path = os.path.join(output_dir, vid+"_key_frames")
output_video_path = os.path.join(output_path, vid+"_tracking.mp4")
# create the output directory
mask_data_dir = os.path.join(output_path, "mask_data")
json_data_dir = os.path.join(output_path, "json_data")
result_dir = os.path.join(output_path, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
start_time = 902
end_time = 1798 #1000 #1798

"""
Custom video input directly using video files
"""
if not os.path.exists(frame_path):
    # saving video to frames
    source_frames = Path(frame_path)
    source_frames.mkdir(parents=True, exist_ok=True)
    offset = 0
    for tt in range(start_time, end_time+1):
        video = os.path.join(video_path, str(tt)+".mp4")
        video_info = sv.VideoInfo.from_video_path(video)  # get video info
        print(video_info)
        width = video_info.width
        height = video_info.height
        frame_rate = video_info.fps
        frame_generator = sv.get_video_frames_generator(video, stride=1, start=0, end=None)
        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=False, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame, image_name="{:05d}.jpg".format(offset))
                print("offset", "{:05d}.jpg".format(offset))
                offset += 1
else:
    video = os.path.join(video_path, str(start_time)+".mp4")
    video_info = sv.VideoInfo.from_video_path(video)  # get video info
    print(video_info)
    width = video_info.width
    height = video_info.height
    frame_rate = video_info.fps