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

vid = "fi1EGafAVh0_1"
video_root = f"/share/test/xieyiweng/datasets/BenSMOT/BenSMOT/test/attending_zoo_museum/{vid}"
gt_root = os.path.join(video_root, "gt")
img_root = os.path.join(video_root, "imgs")
gt_path = os.path.join(gt_root, "gt.txt")
output_path = f"./{vid}"
if not os.path.exists(output_path):
    os.makedirs(output_path)
anno_dict = {}
with open(gt_path, 'r') as file:
    for line in file:
        data = line.strip().split(',')
        frame_id = str(data[0])
        if frame_id not in anno_dict:
            anno_dict[frame_id] = []
        object_id = int(data[1])
        x1 = float(data[2])
        y1 = float(data[3])
        x2 = float(data[4])+x1
        y2 = float(data[5])+y1
        anno_dict[frame_id].append({
            "id": object_id,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
for frame_id, obj_list in anno_dict.items():
    img_path = os.path.join(img_root, '{:03d}'.format(int(frame_id))+".jpg")
    input_boxes = []
    objects = []
    pids = []
    for obj in obj_list:
        input_boxes.append([obj["x1"], obj["y1"], obj["x2"], obj["y2"]])
        objects.append(' ')
        pids.append(obj["id"])
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
    cv2.imwrite(os.path.join(output_path, f"gtbox_{frame_id}.jpg"), annotated_frame)


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
    
    with open(os.path.join(output_path, f"z_gtbox_{frame_id}.json"), "w") as f:
        json.dump(results, f, indent=4)