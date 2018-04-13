from __future__ import absolute_import, division, print_function, unicode_literals
import csv
import os


class TaskState:
  """
  Record of the state of the current task vs. total number of tasks.
  One task means one video.
  Properties:
    task_idx    (int) : index of the current task in the list of tasks
    tasks       (list): list of task ids
    num_tasks   (int) : number of tasks loaded
    actions     (list): list of action names; these are class / label num_frames
    tasks_dir   (str) : path to the folder containing all images by video id
  """
  def __init__(self, actions, data_path, tasks):

    # initialize the properties
    self.task_idx    = 0
    self.tasks       = tasks
    self.num_tasks   = len(tasks)
    self.actions     = actions
    self.num_actions = len(actions)
    self.tasks_dir   = data_path


class State:
  """
  Record of the current state of the frame and eyespy.
  Properties:
    image_idx     (int) : index of the current image frame in the list of image frames
    labels        (dict): dict of annotations in the format `{file_name: action}`
    image_dir     (str) : path to folder conatining all frames of current task / video
    frame_paths   (list): list of paths to the image frames
    num_frames    (int) : number of frames in the current task / video
    csv_path      (str) : path to the csv file for reading / writing labels
    color         (str) : name of the color to be displayed on the image frame
    look_ahead    (int) : no llokahead (0); foresight only (1); foresight and hindsight (2)
  """
  def __init__(self, data_path, task_id, image_dir, frame_paths, path_to_frame, color):

    # set path to save the csv file
    csv_path = os.path.join(data_path, task_id, task_id + '_user_task_record.csv')
    # create empty csv files if does not already exist
    if not os.path.isfile(csv_path):
      with open(csv_path, 'w') as f:
        f = open(csv_path)
        pass
      f.close()

    # load labels from csv file
    labels = self._load(csv_path)
    # TODO: check data type, int or str

    # initialize the properties
    self.image_idx   = 0          # current image idx
    self.labels      = labels     # (file_name, action)
    self.image_dir   = image_dir
    self.frame_paths = frame_paths
    self.num_frames  = len(frame_paths)
    self.csv_path    = csv_path
    self.color       = color
    self.look_ahead  = 0
    self.gaze        = []

  @staticmethod
  def _load(csv_path):
    """
    Load the csv file
    Args:
      csv_path    (str) : path to csv file to be loaded
    Returns:
      labels      (dict): dict of `{file_name: action}` annotations
    """
    if os.path.isfile(csv_path):
      with open(csv_path, mode='r') as f:
        reader = csv.reader(f)
        # TODO: update to read in time series also
        labels = {rows[0]:rows[1] for rows in reader}
      if len(labels.items()) > 0:
        return labels
      else:
        return {}
    else:
      return {}


  def get_image_name(self):
    """
    Find the name of the current image
    Args:
    Returns:
      file_name:    (str) : name of the image file
    """
    # TODO: check data type, int or str
    file_name = self.frame_paths[self.image_idx].split('/')[-1].split('.')[0]
    return file_name


  def get_image_label(self):
    """
    Find the label of the current image
    Args:
    Returns:
      label:    (str) : name of the image file
    """
    label = [v[0] for (k, v) in self.labels.items() if self.get_image_name() == k]
    if len(label) == 1:
      label = label[0]
    elif len(label) > 1:
      raise Exception('[ERROR] More than one label must not be present for a single frame')
    else:
      label = 'None'
    return label

  def save(self):
    """
    Save the annotations as a csv file
    Args:
    Returns:
    """
    if len(self.labels.items()) > 0:
      # sort labels by frame ID
      sorted_labels = sorted(list(self.labels.items()), key=lambda x: int(x[0]))
      with open(self.csv_path, 'w') as f:
        for k, v in sorted_labels:
          gaze_series = ' '.join([str(gaze_coord[0]) + ':' + str(gaze_coord[1]) for gaze_coord in v[1]])
          f.write('{},{},{}\n'.format(k, v[0], gaze_series))
