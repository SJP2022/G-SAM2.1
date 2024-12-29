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
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from transformers import logging

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

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)

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
output_path = os.path.join(output_dir, vid+"_key_frames_gdino")
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
new_width = 480
new_height = 360

def cal_hog(img_list, threshold_hog=0.35, threshold_clip=0.9):
    shots = []
    hog_features1 = 0 #当前帧
    hog_features2 = 0 #上一帧
    for i, image in enumerate(img_list):
        hog_features2 = hog_features1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_features1 = hog.compute(gray)
        if i==0 or i==len(img_list)-1:
            shots.append(i)
            continue
        hog_features_diff = cv2.compareHist(hog_features1, hog_features2, cv2.HISTCMP_CORREL)
        #print(type(hog_features_diff))
        #print(i-1, i, hog_features_diff)
        if abs(hog_features_diff)<threshold_hog:
            clip_features = []
            for n in [i-1, i]:
                color = cv2.cvtColor(img_list[n], cv2.COLOR_BGR2RGB)
                inputs = processor(images=color, return_tensors="pt").pixel_values
                inputs = inputs.to(device)
        
                with torch.no_grad():
                    feat = model(inputs)['image_embeds']
                    feat = F.normalize(feat, dim=-1)
                    clip_features.append(feat.cpu().numpy())
            sim = clip_features[0] @ clip_features[1].T
            #print(idx, sim[0,0].item())
            if sim[0,0].item()<threshold_clip: #0.9
                shots.append(i)
    #print("%s finished with shots: %s" % (video_path, shots))
    return shots

img_list = []
tt_list = []

for tt in range(start_time, end_time+1):

    #if str(tt) not in result_dict[vid]:
    #    print("No object in the clip, skip this clip {}".format(tt))
    #    continue

    tt_abs = tt-902 #change the index
    start_frame = tt_abs*frame_rate
    print("frame_names", all_frame_names[start_frame])
    img_path = os.path.join(frame_path, all_frame_names[start_frame])
    img_list.append(cv2.resize(cv2.imread(img_path), (new_width, new_height)))
    tt_list.append(tt)

shots = cal_hog(img_list)
shot_list = [tt_list[i] for i in shots]
print(shot_list)
print(len(shot_list))
anno_list = set()
for i in range(len(shot_list)):
    if i==0 or i==len(shot_list)-1:
        anno_list.add(shot_list[i])
        continue
    #if shot_list[i]-shot_list[i-1]>1:
    #    anno_list.append((shot_list[i]+shot_list[i-1])//2)
    anno_list.add(shot_list[i]-1)
    anno_list.add(shot_list[i])
print(anno_list)
print(len(anno_list))