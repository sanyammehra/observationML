"""
Description:
    Uses youtube-dl package to download .mp4 videos from URLs provided in a .txt file
Usage:
    python get_frames.py [args/flags]
Args:
    --inpath  (str):  Path to the .txt file with the URLs
    --outpath (str):  Path to save the downloaded videos
    -s             :  Flag to run code in simulate mode
"""

import argparse
import os
import time

SAVEPATH ='../Data/Videos/'
URLs = '../Data/URLs.txt'

def main(args):
    best_mp4 = """'bestvideo[ext=mp4]'"""
    # get number of videos in the text file
    with open(args.inpath) as f:
        num_videos = len(f.readlines())
    with open(args.inpath) as f:
        print('[INFO] URL file has ', num_videos, ' URLs')
        # loop through the URLs and download each video
        for i, line in enumerate(f):
            outpath = os.path.join(args.outpath, str(i + 1) + '.mp4')
            # command = 'youtube-dl --sleep-interval 1 -o ' + outpath + ' ' + line
            start = time.time()
            if args.s:
                command = 'youtube-dl -s -f ' + best_mp4 + ' -o ' + outpath + ' ' + line
            else:
                command = 'youtube-dl -f ' + best_mp4 + ' -o ' + outpath + ' ' + line
            print(10*'-')
            print('[INFO] Completed downloading ', i+1, '/', num_videos,
                    '\nTime taken to download from ', line, ' : %.5f' % (time.time() - start), 's')
            os.system(command)
            print('[INFO] Paused to catch a breath...')
            time.sleep(1)
            print('[INFO] Unpaused')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default=URLs, help='File with list of URLs')
    parser.add_argument('--outpath', default=SAVEPATH, help='Download location for the videos')
    # add flag to command if just want to simulate
    parser.add_argument('-s', action='store_true', help='Simulate mode')
    args = parser.parse_args()
    # print(args)
    main(args)
