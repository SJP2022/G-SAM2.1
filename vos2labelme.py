import os
import shutil
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
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

# init grounding dino model from huggingface
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


# setup the input image and text prompt for SAM 2 and Grounding DINO
# VERY important: text queries need to be lowercased + end with a dot
concatenated_text = "person."
#file_name = 'assets/coco-labels-2014_2017.txt'
file_name = 'assets/ytvis-labels.txt'
with open(file_name, 'r') as file:
    lines = file.readlines()
concatenated_text = '. '.join(line.strip() for line in lines)
concatenated_text = "person."

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`  
video_dir = "/share_io03_ssd/common2/videos/AVA/clips/trainval"
#video_dir = "assets"
#annotation_dir = "assets"
frame_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_frames"
# 'output_dir' is the directory to save the annotated frames
output_dir = "/share_io02_hdd/shijiapeng/AVA_annotations_24_10/AVA_tracking"

#vid = "00SfeRtiM2o"
#vid  = "kMy-6RtoOVU"
vid = "VsYPP2I0aUQ"
#vid = "zlVkeKC6Ha8"
video_path = os.path.join(video_dir, vid)
frame_path = os.path.join(frame_dir, vid+"_full")
input_path = os.path.join(output_dir, vid+"_key_frames_gdino")
output_path = os.path.join(output_dir, vid+"_key_frames_gdino_labelme")
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
                offset += 1
else:
    video = os.path.join(video_path, str(start_time)+".mp4")
    video_info = sv.VideoInfo.from_video_path(video)  # get video info
    print(video_info)
    width = video_info.width
    height = video_info.height
    frame_rate = video_info.fps

# scan all the JPEG frame names in this directory
all_frame_names = [
    p for p in os.listdir(frame_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
all_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

annotation = True
if annotation:

    with open("/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/ava_train_v2.2.json", "r", encoding='utf-8') as f:
        result_dict = json.load(f)

    print(result_dict[vid])

    with open(f"{vid}.json", "w", encoding='utf-8') as f:
        json.dump(result_dict[vid], f, indent=4)


for tt in range(start_time, end_time+1):

    tt_abs = tt-902 #change the index
    start_frame = tt_abs*frame_rate
    print("frame_names", all_frame_names[start_frame])
    img_path = os.path.join(frame_path, all_frame_names[start_frame])
    shutil.copy(img_path, os.path.join(output_path, f"{tt}.jpg"))
    
    with open(os.path.join(input_path, f"z_gdinobox_{tt}.json"), "r", encoding='utf-8') as f:
        vos_data = json.load(f)

    shapes = []
    for idx, item in enumerate(vos_data["annotations"]):
        shape_item = {
            "label": str(idx),
            "points": [
                [
                    item["bbox"][0],
                    item["bbox"][1],
                ],
                [
                    item["bbox"][2],
                    item["bbox"][3],
                ]
            ],
            "group_id": None,
            "description": "",
            "shape_type": "rectangle",
            "flags": {},
            "mask": None
        }
        shapes.append(shape_item)

    results = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": f"{tt}.jpg",
        "imageData": None,
        "imageHeight": vos_data["img_height"],
        "imageWidth": vos_data["img_width"]
    }

    with open(os.path.join(output_path, f"{tt}.json"), "w") as f:
        json.dump(results, f, indent=4)