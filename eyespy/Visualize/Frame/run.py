# from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import cv2
import os
import random
from eyespy.Visualize.Frame.states import *
import eyespy.Visualize.Frame.utils as utils


def main(opt):

  # initialize `task_state` for all tasks / videos
  task_state = TaskState(opt.actions, opt.data_path, opt.videos)
  job, video_state = 1, None

  while True:

    # Initialising `video_state` for current task / video
    if job == 1:
      image_dir = os.path.join(opt.data_path, str(task_state.tasks[task_state.task_idx]))
      # TODO: Prepare a csv file and list to avoid loading in every loop
      frame_paths, _, path_to_frame = utils.findpaths(image_dir)
      color = utils.COLORS[random.choice(list(utils.COLORS.keys()))]
      video_state = VideoState(opt.data_path, str(task_state.tasks[task_state.task_idx]),
                               image_dir, frame_paths, path_to_frame, color)

    # print info for terminal GUI
    utils.print_info(task_state, video_state)

    # load image and embed info
    image_path = video_state.frame_paths[video_state.image_idx]
    image = utils.imread(image_path, resize=2)
    utils.image_info(image, task_state, video_state)

    # record user input and implement action
    job = utils.read_key(cv2.waitKey(0), task_state, video_state)

    # quit condition
    if job == -1:
      break


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default='./eyespy/Data/Frames/', help='Path to the folder with images by task / video id (optional)')
  parser.add_argument('--videos', nargs='+', type=int, default=[idx+1 for idx in range(9)], help='Videos to start annotating (optional)')
  parser.add_argument('--actions', type=str, nargs='+', help='List of actions (optional)',
                      default=['act1',
                               'act2',
                               'act3',
                               'act4',
                               'act5',
                               'act6',
                               'act7',
                               'act8',
                               'act9'])

  args = parser.parse_args()

  main(args)
