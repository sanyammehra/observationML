from __future__ import absolute_import, division, print_function, unicode_literals
import cv2
import itertools
import numpy as np
import os
import random
import re

# Colors
COLORS = {
  'WHITE'   : (255, 255, 255),
  'GREEN'   : (0, 255, 0),
  'BLUE'    : (255, 0, 0),
  'CYAN'    : (255, 255, 0),
  'YELLOW'  : (0, 255, 255),
  'MAGENTA' : (255, 0, 255),
  'RED'     : (0, 0, 255),
  'BLACK'   : (0, 0, 0),
}


KEYMAP = {
  'left' : [65361, 2, ord('a')],
  'up'   : [65362, 0, ord('w')],
  'right': [65363, 3, ord('d')],
  'down' : [65364, 1, ord('s')],
}


def findpaths(path):
  """
  Enumerate all the images.
  Args:
    path                (str) :  Path to the images
  Returns:
    frame_paths_sorted  (list):  List of all the paths as strings
    frame_to_path_dict  (dict):  Dict with (id, path) pairs
    path_to_frame_dict  (dict):  Dict with (path, id) pairs
  """
  print('[INFO] Searching for .png images in ', path)
  frame_paths = []
  frame_to_path_dict = {}
  path_to_frame_dict = {}
  for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
      if name.find('.png') != -1:
        frame_path = os.path.join(root, name)
        # NOTE: may want to change to deal with generic file names
        match = re.search(r'(?P<video_id>\d+)_(?P<frame_id>\d+).png', name)
        # video_id = int(match.group('video_id'))
        frame_id = int(match.group('frame_id'))
        frame_paths.append(frame_path)
        frame_to_path_dict[frame_id] = frame_path
        path_to_frame_dict[frame_path] = frame_id
  frame_paths_sorted = sorted(frame_paths, key=lambda x: int(path_to_frame_dict[x]))
  print('[INFO] %i frames located ' % (len(frame_paths)))
  return frame_paths_sorted, frame_to_path_dict, path_to_frame_dict


def imread(image_path, resize=1):
  """
  Read the image from given path.
  Args:
    image_path    (str) : path to the image to be loaded
    resize        (flt) : scaling factor
  Returns:
    image         (npy) : numpy array of the image
  """
  image = cv2.imread(image_path)
  # Removed option to adjust white balance and add colormap
  # image = np.clip(image, 0, white)/white*255
  # image = np.array(image, dtype=np.uint8)
  # image = cv2.applyColorMap(image, cv2.COLORMAP_OCEAN)
  image = cv2.resize(image, (0, 0), fx=resize, fy=resize, interpolation=cv2.INTER_NEAREST)
  return image

'''
def rotate(image, deg):
  if deg == 0:
    return image
  elif 1 <= deg <= 3:
    return np.rot90(image, deg).copy()
  else:
    raise NotImplementedError
'''


def grid_images(im_paths, w=4, h=4, margin=0, scale=0.5):
  """
  Display multiple images as a grid. Used for foresight and hindsight.
  Args:
    im_paths    (list): List containing paths of images to be displayed as a grid
    w           (int) : Width of the grid in no. of images
    h           (int) : Height of the grid in no. of images
    margin      (int) : Margin between images in the grid in px
    scale       (flt) : Scale factor to resize images to be displayed in the grid
  Returns:
    imgmatrix   (npy) : Combined image containing all images concatenated into a grid
  """
  n = w * h
  if len(im_paths) > n:
    raise ValueError('Number of images ({}) does not conform to '
                     'matrix size {}x{}'.format(w, h, len(im_paths)))
    return
# unscaled_imgs = [gray_to_map_and_flip(img=cv2.imread(fp)[..., 0], colormap='o', rotate=180) for fp in im_paths]
  unscaled_imgs = [cv2.imread(fp) for fp in im_paths]
  imgs = [cv2.resize(I, (int(I.shape[1] * scale), int(I.shape[0] * scale))) for I in unscaled_imgs]
  if any(i.shape != imgs[0].shape for i in imgs[1:]):
    raise ValueError('Not all images have the same shape')
    return
  img_h, img_w, img_c = imgs[0].shape
  m_x = 0
  m_y = 0
  if margin is not None:
    m_x = int(margin)
    m_y = m_x
  imgmatrix = np.zeros((int(img_h * h + m_y * (h - 1)),
                      int(img_w * w + m_x * (w - 1)),
                      img_c), dtype=np.uint8)
  imgmatrix.fill(255)
  positions = itertools.product(range(int(w)), range(int(h)))
  for (x_i, y_i), img in zip(positions, imgs):
    x = x_i * (img_w + m_x)
    y = y_i * (img_h + m_y)
    imgmatrix[y: y + img_h, x: x + img_w, :] = img
  return imgmatrix


def foresight(video_state, dist=16):
  """
  Display the next `dist` frames.
  Args:
    dist    (int): Number of images to be displayed in the foresight
  Returns:
  """
  assert dist < 100, 'Too much foresight requested, you\'re not that smart!'
  im_paths = video_state.frame_paths[video_state.image_idx + 1: video_state.image_idx + 1 + dist]
  if len(im_paths) > 0:
    image_grid = grid_images(im_paths=im_paths, w=np.ceil(np.sqrt(dist)), h=np.ceil(np.sqrt(dist)), margin=1)
    cv2.imshow('Foresight', image_grid)


def hindsight(video_state, dist=16):
  """
  Display the previous `dist` frames.
  Args:
    dist    (int): Number of images to be displayed in the hindsight
  Returns:
  """
  assert dist < 100, 'Too much hindsight requested, you\'re not that smart!'
  im_paths = video_state.frame_paths[max(video_state.image_idx - 1 - dist, 0): max(video_state.image_idx - 1, 0)]
  if len(im_paths) > 0:
    image_grid = grid_images(im_paths=im_paths, w=np.ceil(np.sqrt(dist)), h=np.ceil(np.sqrt(dist)), margin=1)
    cv2.imshow('Hindsight', image_grid)


def image_info(image, task_state, video_state):
  """
  Embed text info on the displayed image.
  Args:
    image       (npy) : numpy array of the image to be labeled
    task_state  (obj) : current TaskState object
    video_state (obj) : current VideoState object
  Returns:
  """
  image_info = 'Frame {}/{} ({})'.format(video_state.image_idx + 1, video_state.num_frames, video_state.get_image_name())
  cv2.putText(image, image_info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, video_state.color, 1)

  label_info = []
  if len(video_state.labels) > 0:
    label_info = ['{}'.format(a) for (f, a) in video_state.labels.items() if video_state.get_image_name().split('.')[0] == f]
  if len(label_info) == 0:
    label_info = ['None']
  for i, row in enumerate(label_info):
    cv2.putText(image, row, (5, 35 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, video_state.color, 1)
  cv2.imshow('Video', image)
  if video_state.look_ahead == 0:   # no lookahead
    cv2.destroyWindow('Foresight')
    cv2.destroyWindow('Hindsight')
  elif video_state.look_ahead == 1: # only foresight
    foresight(video_state)
  elif video_state.look_ahead == 2: # foresight and hindsight
    foresight(video_state)
    hindsight(video_state)


def orange_fg(text1, text2=''):
  print('\033[94m'+text1+'\033[0m'+text2)


def yellow_fg(text1, text2=''):
  print('\033[93m'+text1+'\033[0m'+text2)


def red_fg(text1, text2=''):
  print('\033[91m'+text1+'\033[0m'+text2)


def orange_fg(text1, text2='', newline=True):
  if newline:
    print('\033[33m'+text1+'\033[0m'+text2)
  else:
    print('\033[33m'+text1+'\033[0m'+text2, end='')


def blue_bg(text1, text2=''):
  print('\033[44m'+text1+'\033[40m'+text2)


def purple_fg(text1, text2=''):
  print('\033[45m'+text1+'\033[40m'+text2)


def red_bg(text1, text2=''):
  print('\033[41m'+text1+'\033[40m'+text2)


def print_info(task_state, video_state):
  """
  Print for terminal 'GUI'
  Args:
    task_state  (obj) : current TaskState object
    video_state (obj) : current VideoState object
  Returns:
  """
  os.system('clear')

  # instructions
  blue_bg('\n           Instructions           ')
  orange_fg('\u21e6 / \u21e8:\t', '1 frame back/forward')
  orange_fg('\u21e9 / \u21e7:\t', '10 frame back/forward')
  orange_fg('< / >:\t', '100 frame back/forward')
  orange_fg('[ / ]:\t', 'Previous/next task/video')
  orange_fg('Esc:\t', 'Exit')
  orange_fg('0-9:\t', 'Action ID')
  orange_fg('t / i:\t', '[User Input] Jump to Task/Image ID')
  orange_fg('Space:\t', 'Toggle text color')
  orange_fg('Tab:\t', 'Toggle lookahead mode')
  red_fg('Note:\t', '(a) Select image as active window  (b) Turn off Caps Lock  (c) Do not press shift key')

  # state information
  blue_bg('\n              State               ')
  orange_fg('Video ID: ', '{}\t'.format(task_state.tasks[task_state.task_idx]), newline=False)
  orange_fg('Frame ID: ', '{}'.format(video_state.get_image_name()))
  orange_fg('Image ID: ', '{}/{}'.format(video_state.image_idx + 1, video_state.num_frames))
  orange_fg('Action ID: ', video_state.get_image_label())

  # action dictionary and key mapping
  blue_bg('\n              Actions List              ')
  for a, action in enumerate(task_state.actions):
    orange_fg('Action {}: '.format(a + 1), action)

  # annotations
  blue_bg('\n           Actions Record            ')
  for frame_idx, (f, a) in enumerate(video_state.labels.items()):
    orange_fg('Label {}: '.format(frame_idx + 1), '{} --> {}'.format(f, a))


def get_user_input(prompt):
  """
  Prompt and get integer input from user.
  Args:
    prompt      (str) : prompt to be displayed to the user
  Returns:
    tmp         (int) : stored value from user's input
  """
  while True:
    user_input = input(prompt)
    try:
      tmp = int(user_input)
      return tmp
    except ValueError:
      print('Not a number')


def read_key(key, task_state, video_state):
  """
  Implement action based on key code.
  Args:
    task_state  (obj) : current TaskState object
    video_state (obj) : current VideoState object
  Returns:
    job         (int) : end annotation (-1); continue the task (0); new task (1)
  """
  if key in KEYMAP['left']:   # -1 frame
    video_state.image_idx -= 1
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key in KEYMAP['right']:    # +1 frame
    video_state.image_idx += 1
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key in KEYMAP['up']:   # -10 frames
    video_state.image_idx -= 10
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key in KEYMAP['down']:   # +10 frame
    video_state.image_idx += 10
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key == ord(','):  # <  # -100 frame
    video_state.image_idx -= 100
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key == ord('.'):  # >  # +100 frame
    video_state.image_idx += 100
    video_state.image_idx = min(max(0, video_state.image_idx), video_state.num_frames - 1)

  elif key == ord('\x1b'):  # esc
    video_state.save()
    return -1


  elif ord('0') <= key <= ord('9'):  # action idx
    tmp = key - ord('0') - 1
    if 0 <= tmp < task_state.num_actions:
      action_idx = tmp
      video_state.labels[video_state.get_image_name()] = task_state.actions[int(action_idx)]
      video_state.save()


  # elif key == ord('r'):  # remove
  #   tmp = get_user_input('Enter the Remove Clip ID: ') - 1
  #   if 0 <= tmp < len(video_state.clips):
  #     video_state.clips.pop(tmp)

  elif key == ord('i'):  # jump to image
    tmp = get_user_input('Enter the Frame ID: ') - 1
    if 0 <= tmp < video_state.num_frames:
      video_state.image_idx = tmp

  elif key == ord('t'):  # jump to task
    tmp = get_user_input('Enter the Task ID: ') - 1
    if 0 <= tmp < task_state.num_tasks:
      task_state.task_idx = tmp
    return 1

  elif key == ord(']'):  # next task / video
    task_state.task_idx += 1
    task_state.task_idx = min(max(0, task_state.task_idx), task_state.num_tasks - 1)
    video_state.save()
    return 1

  elif key == ord('['):  # previous task / video
    task_state.task_idx -= 1
    task_state.task_idx = min(max(0, task_state.task_idx), task_state.num_tasks - 1)
    video_state.save()
    return 1

  elif key == ord('\t'):   # TAB  #lookahead toggle
    video_state.look_ahead = (video_state.look_ahead + 1) % 3

  elif key == ord(' '):  # randomize text color
    color_key = random.choice(list(COLORS.keys()))
    video_state.color = COLORS[color_key]

  return 0
