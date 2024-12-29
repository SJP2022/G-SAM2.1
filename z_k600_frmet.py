# command: python frame_extract.py --process 8
import argparse
import json
import os
from multiprocessing import Process, Queue
from time import time

from decord import VideoReader

vid_path = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/vid.json"
with open(vid_path, "r", encoding='utf-8') as f:
    vid_list = json.load(f)

video_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_videos"
frames_root = "/share_io02_hdd/shijiapeng/K600_annotations_24_11/K600_frames"

if not os.path.exists(frames_root):
    os.makedirs(frames_root)

def solve(process_id, videos):
    start_time = time()
    check = 10

    for i, video in enumerate(videos):
        frames_path = os.path.join(frames_root, video.split(".")[0])
        if not os.path.exists(frames_path):
            os.makedirs(frames_path)

        video_path = os.path.join(video_root, video)
        vr = VideoReader(video_path)
        fps = vr.get_avg_fps()
        duration = int(len(vr))
        cmd = f"ffmpeg -i {video_path} "
        cmd += f"{frames_path}/%05d.jpg -loglevel quiet"
        os.system(cmd)

        if duration != len(os.listdir(frames_path)):
            print(
                f"video: {video}, duration: {duration}, # of frames: {len(os.listdir(frames_path))}"
            )
        # print(cmd)
        if i % check == check - 1:
            ET = time() - start_time
            ETA = ET / (i + 1) * (len(videos) - i - 1)
            print(
                "process:%d %d/%d ET: %.2fmin ETA:%.2fmin "
                % (process_id, i + 1, len(videos), ET / 60, ETA / 60)
            )

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', default=8, type=int)
    args = parser.parse_args()

    num_process = args.process
    sub_video_list = []
    videos = [vid+".mp4" for vid in vid_list]
    n = len(videos)
    step = n // num_process
    j = 0

    # random.shuffle(video_list)
    for i in range(0, n, step):
        j += 1
        if j == num_process:
            sub_video_list.append(videos[i:n])
            break
        else:
            sub_video_list.append(videos[i : i + step])

    process_list = []
    Q = Queue()
    for i, item in enumerate(sub_video_list):
        cur_process = Process(target=solve, args=(i, item))
        process_list.append(cur_process)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()