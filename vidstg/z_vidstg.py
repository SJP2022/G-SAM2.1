import os
import random
import shutil

#'''
# 源目录和目标目录
target_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/vidor_annos"
source_root = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/validation"
source_id = os.listdir(source_root)
for id in source_id:
    source_dir = os.path.join(source_root, id)
    source_video = os.listdir(source_dir)
    for vid in source_video:
        source_path = os.path.join(source_dir, vid)
        target_path = os.path.join(target_dir, vid)
        shutil.copyfile(source_path, target_path)
        '''
        try:
            os.symlink(source_path, target_path)
            print(f"软链接已创建：{target_path} -> {source_path}")
        except OSError as e:
            print(f"创建软链接失败：{e}")
        '''
#'''
'''
import json

with open("/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/10712559773.json", "r", encoding='utf-8') as f:
    vid_list = json.load(f)
with open("/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/10712559773.json", "w", encoding='utf-8') as f:
    json.dump(vid_list, f, indent=4)
'''
'''
target_dir = "/share_io02_hdd/shijiapeng/STVG_datasets/vidstg/vv"
source_root = "/share/test/xieyiweng/datasets/VIDOR/vidor/training"
source_id = os.listdir(source_root)
for id in source_id:
    source_dir = os.path.join(source_root, id)
    source_video = os.listdir(source_dir)
    for vid in source_video:
        if vid=="2794976541.json":
            print(source_dir)
'''