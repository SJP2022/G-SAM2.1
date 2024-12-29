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
# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", device)

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
image_predictor = SAM2ImagePredictor(sam2_image_model)


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
output_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_tracking"

#vid = "00SfeRtiM2o"
#vid  = "kMy-6RtoOVU"
vid = "VsYPP2I0aUQ"
video_path = os.path.join(video_dir, vid)
frame_path = os.path.join(frame_dir, vid)
output_path = os.path.join(output_dir, vid+"_demo_4")
output_video_path = os.path.join(output_path, vid+"_tracking.mp4")
# create the output directory
mask_data_dir = os.path.join(output_path, "mask_data")
json_data_dir = os.path.join(output_path, "json_data")
result_dir = os.path.join(output_path, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)
start_time = 902
end_time = 1000 #1000 #1798

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

#sam2_masks = MaskDictionaryModel()
PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point
#objects_count = 0

def img2box(img_path, text):
    image = Image.open(img_path).convert("RGB")

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.2, #0.4
        text_threshold=0.2, #0.3
        target_sizes=[image.size[::-1]]
    )

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    # print("results[0]",results[0])
    objects = results[0]["labels"]

    return input_boxes, objects

def box2mask(img_path, input_boxes):
    image = Image.open(img_path).convert("RGB")

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 2:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)
    
    return masks

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection_area/(box1_area + box2_area - intersection_area)
    return iou

def gdino_with_gt(gdino_dict, gt_dict, iou_threshold=0.3):
    input_boxes_gdino = gdino_dict["input_boxes"]
    objects_gdino = gdino_dict["objects"]
    input_boxes_gt = gt_dict["input_boxes"]
    objects_gt = gt_dict["objects"]
    input_boxes = []
    objects = []
    for gtid in range(len(objects_gt)):
        gtbox = input_boxes_gt[gtid]
        flag = 0 
        max_iou = 0
        max_iou_id = -1
        have_same_class = False
        for gdid in range(len(objects_gdino)):
            gdbox = input_boxes_gdino[gdid]
            iou = calculate_iou(gtbox, gdbox)
            print("iou", iou)
            if iou > iou_threshold:
                if objects_gt[gtid]==objects_gdino[gdid]:
                    if (not have_same_class) or iou>max_iou:
                        max_iou_id = gdid
                        max_iou = iou
                    have_same_class = True
                elif not have_same_class:
                    if iou>max_iou:
                        max_iou_id = gdid
                        max_iou = iou

        flag = max_iou_id

        if flag>=0:
            input_boxes.append(input_boxes_gdino[flag])
            objects.append(objects_gdino[flag])
        else:
            input_boxes.append(input_boxes_gt[gtid])
            objects.append(objects_gt[gtid])

    return input_boxes, objects


for tt in range(start_time, end_time+1):

    if str(tt) not in result_dict[vid]:
        print("No object in the clip, skip this clip {}".format(tt))
        continue

    frame_object_count = {}
    tt_abs = tt-902 #change the index
    end_frame_inall = (tt_abs+1)*frame_rate-1
    if tt_abs>2:
        start_frame_inall = (tt_abs-3)*frame_rate
        start_frame = frame_rate*3
    elif tt_abs>1:
        start_frame_inall = (tt_abs-2)*frame_rate
        start_frame = frame_rate*2
    elif tt_abs>0:
        start_frame_inall = (tt_abs-1)*frame_rate
        start_frame = frame_rate
    else:
        start_frame_inall = tt_abs*frame_rate
        start_frame = 0
    print("start_frame_inall", start_frame_inall, end_frame_inall+1)
    frame_names = all_frame_names[start_frame_inall: end_frame_inall+1]
    text = concatenated_text
    # init video predictor state
    inference_state = video_predictor.init_state(video_path=frame_path, start_frame=start_frame_inall+1, end_frame=end_frame_inall+1)
    step = frame_rate # the step to sample frames for Grounding DINO predictor # 10
    print("frame_names", frame_names, frame_names[start_frame])
    img_path = os.path.join(frame_path, frame_names[start_frame])
    image_base_name = frame_names[start_frame].split(".")[0]
    #mask_dict_gdino
    mask_dict = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")

    # gdino box
    input_boxes, objects = img2box(img_path, text)
    gdino_dict = {
        "input_boxes": input_boxes,
        "objects": objects
    }
    
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
    
    input_boxes, objects = gdino_with_gt(gdino_dict=gdino_dict, gt_dict=gt_dict)

    try:
        masks = box2mask(img_path, input_boxes)
    except Exception:
        print("input_boxes", input_boxes)
        print("No object detected in the frame, skip the frame {}".format(start_frame))
        continue

    input_boxes = np.vstack(input_boxes)

    confidences = np.ones(masks.shape[0])
    scores = np.ones(masks.shape[0])
    class_names = ['person' for i in range(masks.shape[0])]
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
        mask=masks.astype(bool),  # (n, h, w)
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
    cv2.imwrite(os.path.join(output_path, f"groundingdino_annotated_image_{frame_names[start_frame][:5]}.jpg"), annotated_frame)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join(output_path, f"grounded_sam2_annotated_image_with_mask_{frame_names[start_frame][:5]}.jpg"), annotated_frame)


    """
    Dump the results in standard format and save as json files
    """

    def single_mask_to_rle(mask):
        rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        return rle

    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

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
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(output_path, f"grounded_sam2_hf_model_demo_results_{frame_names[start_frame][:5]}.json"), "w") as f:
        json.dump(results, f, indent=4)
