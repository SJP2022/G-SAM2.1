import os
import time
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
sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "sam2.1_hiera_l.yaml"
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
annotation_dir = "/share_io03_ssd/test2/shijiapeng/ActivityNet_annotations_24_8/ActivityNet_annotations_vicuna_sum"
video_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/videos"
#video_dir = "assets"
#annotation_dir = "assets"
frame_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/frames"
shot_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/shots"
# 'output_dir' is the directory to save the annotated frames
#output_dir = "/share_io03_ssd/test2/shijiapeng/AVA_annotations_24_10/AVA_tracking"
output_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/exp/tracking"

def img2box(img_path, text):
    image = Image.open(img_path).convert("RGB")

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4, #0.4
        text_threshold=0.3, #0.3
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

#vid = "00SfeRtiM2o"
vid  = "example"
#vid = "p9hJmlWGvFI"
#vid = "9XyrLUWZl40"
#vid = "gJKw6jGjiDE"
#vid = "0AjYz-s4Rek"
#vid = "aDrjDISgmLU"
#vid = "tracking_car_exp"
video_path = os.path.join(video_dir, vid+".mp4")
frame_path = os.path.join(frame_dir, vid)
output_path = os.path.join(output_dir, vid)
output_video_path = os.path.join(output_path, vid+"_tracking.mp4")
# create the output directory
mask_data_dir = os.path.join(output_path, "mask_data")
json_data_dir = os.path.join(output_path, "json_data")
result_dir = os.path.join(output_path, "result")
CommonUtils.creat_dirs(mask_data_dir)
CommonUtils.creat_dirs(json_data_dir)

"""
Custom video input directly using video files
"""
video_info = sv.VideoInfo.from_video_path(video_path)  # get video info
print(video_info)
width = video_info.width
height = video_info.height
#frame_rate = video_info.fps
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()

# scan all the JPEG frame names in this directory
all_frame_names = [
    p for p in os.listdir(frame_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
all_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

annotation = True
if annotation:
    shot_path = os.path.join(shot_dir, vid+".json")
    if os.path.exists(shot_path):
        with open(shot_path, 'r', encoding='utf-8') as f:
            try:
                shot_points = json.load(f)
            except json.decoder.JSONDecodeError:
                print("%s json.decoder.JSONDecodeError" % vid)
    else:
        print("annotations of %s don't exist. Can't summarize this video!!!" % vid)
else:
    shot_points = [[0, total_frames/frame_rate]]

PROMPT_TYPE_FOR_VIDEO = "mask" # box, mask or point

print(vid ,"start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for [st, ed] in shot_points:
    start_frame_inall = round(st*frame_rate)
    end_frame_inall = round(ed*frame_rate)
    start_frame = (start_frame_inall+end_frame_inall)//2
    print("start_frame_inall", start_frame_inall, end_frame_inall+1)
    frame_names = all_frame_names[start_frame_inall: end_frame_inall+1]
    text = concatenated_text
    # init video predictor state
    inference_state = video_predictor.init_state(video_path=frame_path, start_frame=start_frame_inall+1, end_frame=end_frame_inall+1)
    step_forward = end_frame_inall-start_frame+1
    step_backward = start_frame-start_frame_inall+1
    print("frame_names", frame_names[start_frame])
    print("forward", step_forward, "backward", step_backward)
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

    try:
        masks = box2mask(img_path, input_boxes)
    except Exception:
        print("input_boxes", input_boxes)
        print("No object detected in the frame, skip the frame {}".format(start_frame))
        continue

    """
    Step 3: Register each object's positive points to video predictor
    """

    #print(masks, input_boxes, objects)
    # If you are using point prompts, we uniformly sample positive points based on the mask
    if mask_dict.promote_type == "mask":
        mask_dict.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=objects)
    else:
        raise NotImplementedError("SAM 2 video predictor only support mask prompts")

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """

    if len(mask_dict.labels) == 0:
        print("No object detected in the frame, skip the frame {}".format(start_frame))
        continue
    video_predictor.reset_state(inference_state)

    for object_id, object_info in mask_dict.labels.items():
        frame_idx, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state,
                start_frame,
                object_id,
                object_info.mask,
            )
    
    video_segments = {}  # output the following {step} frames tracking masks
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step_forward, start_frame_idx=start_frame):
        frame_masks = MaskDictionaryModel()
        
        for i, out_obj_id in enumerate(out_obj_ids):
            out_mask = (out_mask_logits[i] > 0.0) # .cpu().numpy()
            object_info = ObjectInfo(instance_id = out_obj_id, mask = out_mask[0], class_name = mask_dict.get_target_class_name(out_obj_id), logit=mask_dict.get_target_logit(out_obj_id))
            object_info.update_box()
            frame_masks.labels[out_obj_id] = object_info
            image_base_name = frame_names[out_frame_idx].split(".")[0]
            frame_masks.mask_name = f"mask_{image_base_name}.npy"
            frame_masks.mask_height = out_mask.shape[-2]
            frame_masks.mask_width = out_mask.shape[-1]

        video_segments[out_frame_idx] = frame_masks
        #sam2_masks = copy.deepcopy(frame_masks) # maybe can't find object that dismissed in the middle time

    #print("video_segments:", len(video_segments))

    """
    Step 5: save the tracking masks and json files
    """

    for frame_idx, frame_masks_info in video_segments.items():
        mask = frame_masks_info.labels
        mask_img = torch.zeros(frame_masks_info.mask_height, frame_masks_info.mask_width)
        for obj_id, obj_info in mask.items():
            mask_img[obj_info.mask == True] = obj_id

        mask_img = mask_img.numpy().astype(np.uint16)
        np.save(os.path.join(mask_data_dir, frame_masks_info.mask_name), mask_img)

        json_data_path = os.path.join(json_data_dir, frame_masks_info.mask_name.replace(".npy", ".json"))
        #print(frame_masks_info)
        frame_masks_info.to_json(json_data_path) # 此处json文件无法保存mask这种tensor张量

    reverse = True
    if reverse:
        print("try reverse tracking")
        #start_object_id = 0
        object_info_dict = {}
        print("reverse tracking frame", start_frame, frame_names[start_frame])
        if start_frame != 0:
            video_predictor.reset_state(inference_state)
            image_base_name = frame_names[start_frame].split(".")[0]
            json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
            json_data = MaskDictionaryModel().from_json(json_data_path)
            mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
            mask_array = np.load(mask_data_path, allow_pickle=True)
            #for object_id in range(start_object_id+1, current_object_count+1):
            for object_id in json_data.labels.keys():
                print("reverse tracking object", object_id)
                object_info_dict[object_id] = json_data.labels[object_id]
                video_predictor.add_new_mask(inference_state, start_frame, object_id, mask_array == object_id)
            
            for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state, max_frame_num_to_track=step_backward,  start_frame_idx=start_frame, reverse=True):
                image_base_name = frame_names[out_frame_idx].split(".")[0]
                json_data_path = os.path.join(json_data_dir, f"mask_{image_base_name}.json")
                mask_data_path = os.path.join(mask_data_dir, f"mask_{image_base_name}.npy")
                if os.path.exists(json_data_path):
                    json_data = MaskDictionaryModel().from_json(json_data_path)
                    mask_array = np.load(mask_data_path, allow_pickle=True)
                else:
                    json_data = MaskDictionaryModel(promote_type = PROMPT_TYPE_FOR_VIDEO, mask_name = f"mask_{image_base_name}.npy")
                    mask_array = None
                    #json_data.add_new_frame_annotation(mask_list=torch.tensor(masks).to(device), box_list=torch.tensor(input_boxes), label_list=OBJECTS)

                # merge the reverse tracking masks with the original masks
                have_obj = False
                track_obj_ids = []
                for i, out_obj_id in enumerate(out_obj_ids):
                    out_mask = (out_mask_logits[i] > 0.0).cpu()
                    if out_mask.sum() == 0:
                        print("no mask for object", out_obj_id, "at frame", out_frame_idx)
                        continue
                    object_info = object_info_dict[out_obj_id]
                    object_info.mask = out_mask[0]
                    object_info.update_box()
                    object_box = [object_info.x1, object_info.y1, object_info.x2, object_info.y2]
                    has_been_tracked  = False
                    for history_id, history_info in json_data.labels.items():
                        history_box = [history_info.x1, history_info.y1, history_info.x2, history_info.y2]
                        #if calculate_iou(history_box, object_box)>0.7:
                        if history_id==out_obj_id or calculate_iou(history_box, object_box)>0.7:
                            has_been_tracked = True
                            break
                    if has_been_tracked:
                        print("object has been tracked", out_obj_id, "at frame", out_frame_idx)
                        continue
                    have_obj = True
                    track_obj_ids.append(out_obj_id)
                    json_data.labels[out_obj_id] = object_info
                    json_data.mask_height = out_mask.shape[-2]
                    json_data.mask_width = out_mask.shape[-1]
                    if mask_array is None:
                        mask_array = np.zeros((json_data.mask_height, json_data.mask_width))
                    mask_array = np.where(mask_array != out_obj_id, mask_array, 0)
                    mask_array[object_info.mask] = out_obj_id
                
                if have_obj:
                    print(json_data_path, track_obj_ids)
                    np.save(mask_data_path, mask_array)
                    json_data.to_json(json_data_path)
                elif out_frame_idx==start_frame:
                    continue
                else:
                    break

CommonUtils.draw_masks_and_box_with_supervision(frame_path, mask_data_dir, json_data_dir, result_dir)
create_video_from_images(result_dir, output_video_path, frame_rate=frame_rate)
print(vid, "end time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))