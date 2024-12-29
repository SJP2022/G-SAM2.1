import os
import random
import shutil

# 源目录和目标目录
target_dir = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_demo"
source_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_tracking"
action_names = os.listdir(source_root)
for action in action_names:
    source_dir = os.path.join(source_root, os.path.join(action, action+"_tracking.mp4"))
    shutil.copyfile(source_dir, os.path.join(target_dir, action+"_tracking.mp4"))