import argparse
import cv2
import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import tkinter

NUM_IMAGES = 100
MAX_NUM_SHAPES = 20
TASKPATH = './eyespy/Data/Tasks/Shapes/'
GROUND_TRUTH = 'ground_truths'
META_DATA = 'meta_data.txt'
# SHAPES = ['line', 'rectangle', 'circle', 'ellipse', 'polygon']
SHAPES = ['line', 'rectangle', 'circle', 'ellipse']
COLORS = ['red', 'blue', 'green']
COLORMAP = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)} # BGR

def main(opt):
    metadata = {}
    if not os.path.exists(opt.task_path):
        os.makedirs(opt.task_path)
    for idx in range(opt.num_images):
        print('Creating image number: {}'.format(idx + 1))
        image_name = str(idx + 1) + '.png'
        height, width = get_screen_resolution()
        image, shapes = random_image(height, width, opt.max_num_shapes)
        # if np.random.uniform(0, 1) > 0.5:
        #     image = np.rot90(image, 2, (0, 1))
        metadata[idx + 1] = shapes
        # cv2.namedWindow("Image", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("Image",cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('Image', image)
        # cv2.waitKey(0)
        # print(os.path.join(opt.task_path, image_name))
        cv2.imwrite(os.path.join(opt.task_path, image_name), image)
    with open(os.path.join(opt.task_path, META_DATA), 'w') as f:
        json.dump(metadata, f)

def get_screen_resolution():
    root = tkinter.Tk()
    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()
    return height, width

def random_image(height, width, max_num_shapes):
    image = np.ones((height, width, 3), np.uint8)
    num_shapes = random.randint(1, max_num_shapes)
    shapes = {}
    for idx in range(num_shapes):
        shape = random.choice(SHAPES)
        # TODO: document width and height vs. x and y
        # image, shape_info = draw_shape(image, shape, height, width)
        image, shape_info = draw_shape(image, shape, width, height)
        shapes[idx] = shape_info
    shapes['num_shapes'] = num_shapes
    return image, shapes

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
    color = random.choice(COLORS)
    if shape == 'line':
        start = (random.randint(0, height - 1), random.randint(0, width - 1))
        end = (random.randint(0, height - 1), random.randint(0, width - 1))
        image = cv2.line(image, start, end, COLORMAP[color], 4)
    elif shape == 'rectangle':
        leftTop = (random.randint(2, height - 2), random.randint(2, width - 2))
        rightBot = (random.randint(leftTop[0], min(height - 1, leftTop[0] + height//4)),
                    random.randint(leftTop[1], min(width - 1, leftTop[1] + width//4)))
        image = cv2.rectangle(image, leftTop, rightBot, COLORMAP[color], -1)
    elif shape == 'circle':
        radius = random.randint(2, int(min(height, width)//8))
        centre = (random.randint(radius, height - radius - 1), random.randint(radius, width - radius - 1))
        image = cv2.circle(image, centre, radius, COLORMAP[color], -1)
    elif shape == 'ellipse':
        minorAxis = random.randint(2, int(min(height, width)//12))
        majorAxis = random.randint(minorAxis, 2*minorAxis)
        axes = (minorAxis, majorAxis)
        centre = (random.randint(majorAxis, height - majorAxis - 1), random.randint(majorAxis, width - majorAxis - 1))
        image = cv2.ellipse(image, centre, axes, np.random.uniform(0, 360), 0.0, 360.0, COLORMAP[color], -1);
    elif shape == 'polygon':
        num_pts = random.randint(3, 6)
        pts = np.array([[random.randint(0, height - 1), random.randint(0, width -1)] for _ in range(num_pts)],
                np.int32)
        pts = pts.reshape((-1,1,2))
        image = cv2.polylines(image, [pts], True, COLORMAP[color])
    shape_info = {'shape': shape, 'color': color}
    return image, shape_info

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
    parser.add_argument('--num_images', default=NUM_IMAGES, type=int, help='Number of task images to be created')
    parser.add_argument('--max_num_shapes', default=MAX_NUM_SHAPES, type=int, help='Maximum number of shapes per image')
    args = parser.parse_args()
    # print(args)
    main(args)
