import argparse
import csv
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import tkinter

DATAPATH = './eyespy/Data/Tasks/'
MASKPATH = os.path.join(DATAPATH, 'Masks')

def main(opt):
    if not os.path.exists(opt.mask_path):
        os.makedirs(opt.mask_path)
    # csv_path = os.path.join(opt.ask_path, 'Shapes_user_task_record.csv')
    csv_path = './eyespy/Data/Tasks/Shapes/Shapes_user_task_record.csv'
    print(csv_path)
    height, width = get_screen_resolution()
    mask = np.zeros((height, width))
    if os.path.isfile(csv_path):
        print('Found file at: {}'.format(csv_path))
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            pad = 30
            for c, row in enumerate(f):
                record = row.strip().split(',')
                image_name = record[0]
                gaze = record[-1].split()
                # TODO: vectorize this!
                for idx, xy in enumerate(gaze):
                    wh = xy.split(':')
                    w, h = int(wh[0]), int(wh[1])
                    print(h,w)
                    # mask[max(0, h-pad):min(height, h+pad)][max(0, w-pad):min(width, w+pad)] = idx
                    mask[max(0, h-pad):min(height, h+pad), max(0, w-pad):min(width, w+pad)] = 255
                    print('pad', max(0, h-pad))
                mask = mask/(len(gaze) - 1)
                if c == 5:
                    print(np.where(mask!=0))
                    plt.imshow(mask, cmap='OrRd')
                    plt.show()
                    break
        print(os.path.join(opt.mask_path, image_name))
        cv2.imwrite(os.path.join(opt.mask_path, image_name), mask)

def get_screen_resolution():
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return height, width

def random_image(height, width, max_num_shapes):
    image = np.ones((height, width, 3), np.uint8)
    num_shapes = random.randint(1, max_num_shapes)
    for _ in range(num_shapes):
        shape = random.choice(SHAPES)
        image = draw_shape(image, shape, height, width)
    return image

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

def draw_shape(image, shape, height, width):
    """
    Draws shapes on to the image.
    Args:
        image:  (npy): Image to which shapes are to be added
        shape:  (str): Name of shape to be drawn
        height  (int): y-dimension of the image
        width   (int): x-dimension of the image
    Returns:
        image:  (npy): Image to which shapes have been added
    """
    if shape == 'line':
        start = (random.randint(0, height - 1), random.randint(0, width - 1))
        end = (random.randint(0, height - 1), random.randint(0, width - 1))
        image = cv2.line(image, start, end, COLORMAP[random.choice(COLORS)], 4)
    elif shape == 'rectangle':
        leftTop = (random.randint(0, height - 2), random.randint(0, width - 2))
        rightBot = (random.randint(leftTop[0], height - 1), random.randint(leftTop[1], width - 1))
        image = cv2.rectangle(image, leftTop, rightBot, COLORMAP[random.choice(COLORS)], -1)
    elif shape == 'circle':
        centre = (random.randint(0, height - 1), random.randint(0, width - 1))
        radius = random.randint(0, int(min(height, width)//2))
        image = cv2.circle(image, centre, radius, COLORMAP[random.choice(COLORS)], -1)
    return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path', type=str, default=MASKPATH, help='Path to output folder to save images(optional)')
    args = parser.parse_args()
    # print(args)
    main(args)
