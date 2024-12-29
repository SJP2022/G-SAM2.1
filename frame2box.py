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
output_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_box"

#vid = "00SfeRtiM2o"
#vid = "kMy-6RtoOVU"
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
    '''
    annotation_path = "/share_io03_ssd/common2/videos/AVA/annotations/ava_train_v2.2.csv"
    headers = ['video_id', 'timestamp', 'x1', 'y1', 'x2', 'y2', 'action_id', 'person_id']
    df = pd.read_csv(annotation_path, header=None, names=headers)
    result_dict = {}
    for _, row in df.iterrows():
        video_id = row['video_id']
        timestamp = str(row['timestamp'])
        person_id = str(row['person_id'])

        if video_id not in result_dict:
            result_dict[video_id] = {}

        if timestamp not in result_dict[video_id]:
            result_dict[video_id][timestamp] = {}

        if person_id not in result_dict[video_id][timestamp]:
            result_dict[video_id][timestamp][person_id] = {}
        else:
            continue

        result_dict[video_id][timestamp][person_id] = [row['x1'], row['y1'], row['x2'], row['y2']]

    with open("/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/ava_train_v2.2.json", "w", encoding='utf-8') as f:
        json.dump(result_dict, f)
    '''

    with open("/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/ava_train_v2.2.json", "r", encoding='utf-8') as f:
        result_dict = json.load(f)

    print(result_dict[vid])

    with open(f"{vid}.json", "w", encoding='utf-8') as f:
        json.dump(result_dict[vid], f, indent=4)

for tt in range(start_time, end_time+1):

    if str(tt) not in result_dict[vid]:
        print("No object in the clip, skip this clip {}".format(tt))
        continue

    tt_abs = tt-902 #change the index
    start_frame = tt_abs*frame_rate
    print("frame_names", all_frame_names[start_frame])
    img_path = os.path.join(frame_path, all_frame_names[start_frame])
    image_base_name = all_frame_names[start_frame].split(".")[0]
    
    # gt box
    input_boxes = []
    objects = []
    pids = []
    for pid, pbox in result_dict[vid][str(tt)].items():
        input_boxes.append([pbox[0]*width, pbox[1]*height, pbox[2]*width, pbox[3]*height])
        objects.append('person')
        pids.append(pid)
    gt_dict = {
        "input_boxes": input_boxes,
        "objects": objects
    }

    input_boxes = np.vstack(input_boxes)

    #confidences = np.ones(masks.shape[0])
    confidences = np.array([int(i) for i in pids])
    scores = np.array([int(i) for i in pids])
    class_names = ['person' for i in range(len(pids))]
    class_ids = np.array([int(i) for i in pids])

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(class_names, confidences)
    ]

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=None,  # (n, h, w)
        class_id=class_ids
    )

    """
    Note that if you want to use default color map,
    you can set color=ColorPalette.DEFAULT
    """
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

    label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    cv2.imwrite(os.path.join(output_path, f"gtbox_{tt}.jpg"), annotated_frame)


    """
    Dump the results in standard format and save as json files
    """
    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    image = Image.open(img_path).convert("RGB")
    # save the results in standard format
    results = {
        "image_path": img_path,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "score": score,
            }
            for class_name, box, score in zip(class_names, input_boxes, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(output_path, f"z_gtbox_{tt}.json"), "w") as f:
        json.dump(results, f, indent=4)
