# from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import cv2
from eyespy.EyeTribe.peyetribe import EyeTribe
import os
import random
from eyespy.Visualize.Task.states import *
import eyespy.Visualize.Task.utils as utils

DATA_PATH = './eyespy/Data/Tasks/'
TASKS = ['Shapes']
ACTIONS = ['act1', 'act2', 'act3', 'act4', 'act5',
          'act6', 'act7', 'act8', 'act9', 'act10']

def main(opt):
  # Instantiate tracker
  if opt.track:
    tracker = EyeTribe()
    tracker.connect()
    n = tracker.next()
    tracker.pushmode()

  # initialize `task_state` for all tasks
  task_state = TaskState(opt.actions, opt.data_path, opt.tasks)
  job, state = 1, None

  while True:
    # Initialising `state` for current task / video
    if job == 1:
      image_dir = os.path.join(opt.data_path, task_state.tasks[task_state.task_idx])
      # TODO: Prepare a csv file and list to avoid loading in every loop
      frame_paths, _, path_to_frame = utils.findpaths(image_dir)
      color = utils.COLORS[random.choice(list(utils.COLORS.keys()))]
      state = State(opt.data_path, str(task_state.tasks[task_state.task_idx]),
                               image_dir, frame_paths, path_to_frame, color)

    # print info for terminal GUI
    utils.print_info(task_state, state)

    # load image and embed info
    image_path = state.frame_paths[state.image_idx]
    image = utils.imread(image_path, resize=1)
    utils.image_info(image, task_state, state)

    done = False
    count = 0
    while not done:
      count += 1
      if opt.track:
        state.gaze.append(utils.getXY(tracker))
        key = cv2.waitKey(1)
      else:
        state.gaze.append((random.randint(0, 100), random.randint(0, 100)))
        # TODO: Get rid of / tune the delay
        key = cv2.waitKey(50)
      # TODO: Get rid of / tune the count threshold
      if key != -1 or count > 500:
        done = True
        if key != -1:
          # record user input and implement action
          job = utils.read_key(key, task_state, state)
          state.gaze = []

    # quit condition
    if job == -1:
      break


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the folder with images by task (optional)')
  parser.add_argument('--tasks', nargs='+', default=TASKS, help='Tasks to start performing (optional)')
  parser.add_argument('--track', action='store_true', help='Enable the gaze tracking mode; ensure H/w is setup to use this mode')
  parser.add_argument('--actions', type=str, nargs='+', help='List of actions (optional)',
                      default=ACTIONS)

  args = parser.parse_args()

  main(args)
