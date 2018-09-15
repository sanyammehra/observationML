import argparse
from eyespy.Visualize.Task.run import DATA_PATH, TASKS
from eyespy.Visualize.Task.utils import SCREEN_H, SCREEN_W
from functools import reduce
import numpy as np
import os
import _pickle as pickle

NUM_SAMPLES_THRESH = 12
MAX_DEV_X = 100
MAX_DEV_Y = 200
FRAC_FIX = 0.8

def LF_num_samples(s):
    """
    Number of gaze points recorded. Greater length expected for
    complex samples based on the task.
    """
    return len(s) < NUM_SAMPLES_THRESH if s is not None else None

def LF_max_deviation_x(s):
  gaze_x = list(zip(*s))[0]
  return max([abs(gaze_x[idx] - gaze_x[idx+1]) for idx in
  range(len(s)-1)]) < MAX_DEV_X if s is not None else None

def LF_max_deviation_y(s):
  gaze_y = list(zip(*s))[1]
  return max([abs(gaze_y[idx] - gaze_y[idx+1]) for idx in
  range(len(s)-1)]) < MAX_DEV_Y if s is not None else None

def load_labels(pickle_path):
  if os.path.exists(pickle_path) and os.path.getsize(pickle_path) > 0:
    with open(pickle_path, mode='rb') as f:
      labels = pickle.load(f)
    f.close()
    if len(labels.items()) > 0:
      return labels
    else:
      return {}
  else:
    return {}

def main(opt):
  LFs = [LF_num_samples, LF_max_deviation_x, LF_max_deviation_y]
  # TODO: change from 0 to task_id
  pickle_path = os.path.join(opt.data_path, opt.tasks[0], opt.tasks[0] + '_user_task_record.pkl')
  labels = load_labels(pickle_path)
  num_samples, num_funcs = len(labels), len(LFs)
  LF_matrix = np.zeros(shape=(num_samples, num_funcs))
  for idx, (sample, label) in enumerate(labels.items()):
    print(10*'-')
    series = label[-1]
    LF_matrix[idx] = np.array(list(map(lambda x: int(x(series)), LFs)))
    print(LF_matrix)
    print(10*'-')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_path', type=str, default=DATA_PATH, help='Path to the folder with images by task (optional)')
  parser.add_argument('--tasks', nargs='+', default=TASKS, help='Tasks to start performing (optional)')

  args = parser.parse_args()

  main(args)
