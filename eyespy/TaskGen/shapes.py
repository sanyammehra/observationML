import argparse
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import tkinter

TASKPATH = './eyespy/Data/Tasks/Shapes/'
SHAPES = ['line', 'rectangle', 'circle']
COLORS = ['red', 'blue', 'green']
COLORMAP = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)} # BGR

def main(opt):
    if not os.path.exists(opt.task_path):
        os.makedirs(opt.task_path)
    for idx in range(opt.num_images):
        print('Creating image number: {}'.format(idx + 1))
        image_name = str(idx + 1) + '.png'
        height, width = get_screen_resolution()
        image = random_image(height, width, opt.max_num_shapes)
        # cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Image",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # print(os.path.join(opt.task_path, image_name))
        cv2.imwrite(os.path.join(opt.task_path, image_name), image)

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

def crop_resize(image, size: int=224):
    """
    Centre crops and resizes an image.
    Args:
        image           (npy): Image to be cropped and resized
        size            (int): Image to be resized to [size, size, 3]
    Returns:
        image_resized   (npy): Image resized to (size, size, channels)
    """
    H, W, C = image.shape
    assert min(H, W) >= size, 'Video/image resolution %ix%itoo low, size should be atleast %ix%i' % (H, W, size, size)
    if H > W:
        image_crop = image[H//2 - W//2 - 1:H//2 + W//2 - 2, ...]
    elif H < W:
        image_crop = image[:, W//2 - H//2 - 1:W//2 + H//2 - 2, ...]
    else:
        image_crop = image
    D, _, _ = image_crop.shape
    image_resized = cv2.resize(image_crop, (size, size), interpolation = cv2.INTER_AREA)
    return image_resized

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_path', type=str, default=TASKPATH, help='Path to output folder to save images(optional)')
    parser.add_argument('--num_images', default=500, type=int, help='Number of task images to be created')
    parser.add_argument('--max_num_shapes', default=4, type=int, help='Maximum number of shapes per image')
    # parser.add_argument('--size', default=224, type=int, help='Number of task images to be created')
    args = parser.parse_args()
    # print(args)
    main(args)
