import random
from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
from multiprocessing import Process, Queue
from time import time
import numpy as np

def sort_by_number(filename):
    number = int(''.join(filter(str.isdigit, filename)))
    return number

ava_root = "/share_io03_ssd/common2/videos/AVA/clips/trainval"
out_root = "/share_io02_hdd/shijiapeng/AVA_annotations_24_10/AVA_videos"
#video  = "zlVkeKC6Ha8"


def video_cat(video):
    out_path = os.path.join(out_root, video+".mp4")
    if os.path.exists(out_path):
        print(video, "exists.")
        return
    
    video_path = os.path.join(ava_root, video)
    sub_videos = os.listdir(video_path)
    sorted_sub_videos = sorted(sub_videos, key=sort_by_number)
    #print(sorted_sub_videos)

    sub_video_path = [os.path.join(video_path, sv) for sv in sorted_sub_videos]

    clips = [VideoFileClip(file) for file in sub_video_path]

    final_clip = concatenate_videoclips(clips, method="compose")

    final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac')

    print(video, "finished.")


videos = os.listdir(ava_root)
random.shuffle(videos)
for video in videos:
    video_cat(video)

'''
def video_cat(process_id, videos):
    start_time = time()
    check = 10

    for i, video in enumerate(videos):
        out_path = os.path.join(out_root, video+".mp4")
        if os.path.exists(out_path):
            print(video, "exists.")
            continue

        video_path = os.path.join(ava_root, video)
        sub_videos = os.listdir(video_path)
        sorted_sub_videos = sorted(sub_videos, key=sort_by_number)
        #print(sorted_sub_videos)

        sub_video_path = [os.path.join(video_path, sv) for sv in sorted_sub_videos]

        clips = [VideoFileClip(file) for file in sub_video_path]

        final_clip = concatenate_videoclips(clips, method="compose")

        final_clip.write_videofile(out_path, codec='libx264', audio_codec='aac')

        print(video, "finished.")

        # print(cmd)
        if i % check == check - 1:
            ET = time() - start_time
            ETA = ET / (i + 1) * (len(videos) - i - 1)
            print(
                "process:%d %d/%d ET: %.2fmin ETA:%.2fmin "
                % (process_id, i + 1, len(videos), ET / 60, ETA / 60)
            )

if __name__=='__main__':

    num_process = 8
    sub_video_list = []
    videos = os.listdir(ava_root)
    n = len(videos)
    step = n // num_process
    j = 0

    random.shuffle(videos)
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
        cur_process = Process(target=video_cat, args=(i, item))
        process_list.append(cur_process)

    for process in process_list:
        process.start()
    for process in process_list:
        process.join()
'''