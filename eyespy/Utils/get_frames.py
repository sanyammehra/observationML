"""
Description:
    Converts .mp4 videos under the provided path and saves as .png images.
Usage:
    python get_frames.py [args/flags]
Args:
    --inpath  (str):  Path to the location of the videos
    --outpath (str):  Path to save the videos as frames
    --fps     (int):  30fps will be divided by this value; 30/fps will be the sampling rate
    -s             :  Flag to run code in simulate mode
"""

import argparse
import cv2
import os
import re

VIDEOPATH = '../Data/Videos/'
SAVEPATH = '../Data/Frames/'

def find_videos(path):
    """
    Enumerates all the videos and records their paths.
    Args:
        path                (str) : Path to the location of the videos
    Returns:
        video_paths_sorted  (list): List of all the paths as strings
        video_dict          (dict): Dict with (path, id) pairs
    """
    print('[INFO] Searching for .mp4 videos in ', path)
    video_paths = []
    video_dict = {}
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.find('.mp4') != -1:
                video_path = os.path.join(root, name)
                video_paths.append(video_path)
                match = re.search(r'(?P<video_id>\d+).mp4', video_path)
                video_id = match.group('video_id')
                video_dict[video_path] = video_id
    video_paths_sorted = sorted(video_paths, key=lambda x: int(video_dict[x]))
    print('[INFO] Videos located')
    return video_paths_sorted, video_dict

def crop_resize(image, size: int=224):
    """
    Centre crops and resizes an image.
    Args:
        image           (npy): Image to be cropped and resized
        size            (int): Image to be resized to [size, size, 3]
    Returms:
        image_resized   (npy): Image resized to (size, size, channels)
    """
    H, W, C = image.shape
    assert min(H, W) > size, 'Video/image resolution %ix%itoo low, size should be atleast %ix%i' % (H, W, size, size)
    if H > W:
        image_crop = image[H//2 - W//2 - 1:H//2 + W//2 - 2, ...]
    elif H < W:
        image_crop = image[:, W//2 - H//2 - 1:W//2 + H//2 - 2, ...]
    D, _, _ = image_crop.shape
    image_resized = cv2.resize(image_crop, (size, size), interpolation = cv2.INTER_AREA)
    return image_resized

def vid2frame(args, paths):
    """
    Converts videos in paths into frames and saves as .png files
    Args:
        args   (arg):  Args passed into the command line while running this script
        paths  (str):  List of paths of the video files
    Returns:
    """
    sample_rate = 30 // args.fps
    num_videos = len(paths)
    for i, path in enumerate(paths):
        video_id = path.split('.')[-2].split('/')[-1]
        video_dir = os.path.join(args.outpath, video_id)
        # make new dir for current video id
        if not args.s and not os.path.exists(video_dir):
            os.makedirs(video_dir)
        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        frame_id = 0
        success = True
        while success:
            try:
                success, image = vidcap.read()
            except Exception as e: print('Found exception: ', e)
            frame_id += 1
            # save every 5th frame as PNG image
            try:
                if frame_id % sample_rate == 0 and image is not None:
                    if not args.s: cv2.imwrite(os.path.join(video_dir, video_id + '_%d.png' % frame_id), crop_resize(image))
                    if frame_id % (sample_rate * 100) == 0:
                        print('[INFO] Saved Frame id: ', frame_id, ' from Video id: ', video_id)
            except Exception as e: print('Found exception: ', e)
        print(20 * '=' + '\n[INFO] Completed processing video %d/%d\n' % (i + 1, num_videos) + 20 * '=')

def main(args):
    video_paths, _ = find_videos(args.inpath)
    vid2frame(args, video_paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default=VIDEOPATH, help='Location of videos')
    parser.add_argument('--outpath', default=SAVEPATH, help='Location to save the frames')
    parser.add_argument('--fps', default=6, type=int, help='30/fps in integer; 30fps will be divided by this value')
    parser.add_argument('-s', action='store_true', help='Simulate mode')
    args = parser.parse_args()
    # print(args)
    main(args)
