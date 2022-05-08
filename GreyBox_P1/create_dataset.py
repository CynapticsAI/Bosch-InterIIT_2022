import os
import sys
import torchvision as torchvision

sys.path.append("./VST")
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
from VST import get_inference as f

video_file_paths = glob("/saved_videos/*.mp4")  # directory containing all .mp4 files
video_file_paths.sort()
train_frame_basedir = "train_dataset_frames/"
val_frame_basedir = "val_dataset_frames/"

for i in range(0, 400):
    if not os.path.exists(train_frame_basedir + str(i)):
        os.makedirs(train_frame_basedir + str(i))
    if not os.path.exists(val_frame_basedir + str(i)):
        os.makedirs(val_frame_basedir + str(i))

val_class_dict = np.zeros(400)
train_class_dict = np.zeros(400)
error_in_files = 0

for video_file_path in tqdm(video_file_paths):
    probs = []
    try:
        video = torchvision.io.read_video(video_file_path)[0]  # Frames * H * W * C
        logits_tuple = f.get_logits(video_file_path)
        probs.append([mm[1] for mm in logits_tuple])
        class_label = torch.argmax(torch.FloatTensor(probs), dim=1).detach().cpu().numpy()[0]

        # save at least 3 videos files from each class in validation dataset
        if val_class_dict[class_label] < 3:
            number_of_frames = video.shape[0]
            start_frame = int(number_of_frames // 3)
            end_frame = 2 * start_frame
            video_name = video_file_path.split("saved_videos/")[-1].split(".mp4")[0]
            video = video.permute(0, -1, 1, 2)
            for i in range(start_frame, end_frame):
                torchvision.utils.save_image(video[i] / 255., val_frame_basedir + str(class_label) + "/" +
                                             video_name + "_" + str(i) + ".png")
            val_class_dict[class_label] += 1
            continue

        number_of_frames = video.shape[0]
        start_frame = int(number_of_frames // 3)
        end_frame = 2 * start_frame
        video_name = video_file_path.split("saved_videos/")[-1].split(".mp4")[0]
        video = video.permute(0, -1, 1, 2)
        for i in range(start_frame, end_frame):
            torchvision.utils.save_image(video[i] / 255., train_frame_basedir + str(class_label) + "/" +
                                         video_name + "_" + str(i) + ".png")
        train_class_dict[class_label] += 1
    except:
        error_in_files += 1
        print(f"Error in file: {video_file_path}")

print("Task Complete")
