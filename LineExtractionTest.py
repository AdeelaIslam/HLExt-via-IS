#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 08:56:31 2019

@author: v1
"""
import os
import sys
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

from PIL import Image, ImageChops 


from os import listdir
import numpy as np
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn import visualize
import cv2
import json
import skimage 


os.makedirs("./Test_Results", exist_ok=True)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def display_instances(image_name, image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=False,
                      colors=None, captions=None):
                      
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    print("masks", masks.shape)
    pyplot.close('all') 

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = pyplot.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        #print("color",color)
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=5,
                                alpha=0.7, linestyle="solid",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        #ax.text(x1, y1 + 8, caption, color='b', size=30, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]

        # uncomment to crop lines
        """
        #   ADEELA START
        #   Converting mask from true false to 1, 0
        mask2 = np.where(mask == True, 1, 0)
        
        '''
        #   Uncomment to crop with a rectangles area
        i, j = np.where(mask2)
        indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
        cropped_arr = image[indices]
        '''
        #   creating masked array
        masked_arr = mask2[...,None]*image

        #crop edges
        true_points = np.argwhere(masked_arr)
        top_left = true_points.min(axis=0)
        bottom_right = true_points.max(axis=0)
        cropped_arr = masked_arr[top_left[0]:bottom_right[0]+1,top_left[1]:bottom_right[1]+1]
        
        #   convert black pixels to white
        #   saving cropped lined from mask
        im = Image.fromarray(cropped_arr.astype(np.uint8))
        im.save("./Cropped_lines/" + image_name[:-4]+'_'+str(i)+'.jpg')

        #   ADEELA END
        """

        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
 

    
    
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        pyplot.savefig("./Test_Results/" + image_name[:-4]+'_'+''+'.jpg', dpi=200, bbox_inches='tight', pad_inches=0)
        #pyplot.show()    

# define the prediction configuration
class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "line_extraction_cfg"
    # number of classes (background + line)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def separateRowsAndColumns(json_data):
    rows=[]
    cols=[]
    for box in json_data['boxes']:
       rowOrCol = box['label']
       if rowOrCol=="row":
           rows.append(box)
       if rowOrCol=="column":
           cols.append(box)
    return rows,cols

def sortRows(rows):
    sortedRows = sorted(rows, key=lambda k: k['row_start']) 
    #print(rows)
    #print(sortedRows)
    return sortedRows
    

def sortColumns(cols):
    sortedCols = sorted(cols, key=lambda k: k['col_end'], reverse=True) 
    #print(cols)
    #print(sortedCols)
    return sortedCols
 
#set paths
images_dir ='./test_images/'
model_directory_path='./Model/'            
weights_path='./Model/mask_rcnn_lineextraction_cfg_0030.h5' 
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir=model_directory_path , config=cfg)
# load model weights
model.load_weights(weights_path, by_name=True)

#Detect the rows and columns in the tables which are detected from tabeDetectionModel 
def plot_actual_vs_predicted(image, image_name, model, cfg, n_images=1):
  json_data={}
  print(image_name)
  json_data['page_name']=image_name
  boxes=[]
  class_names=['background','line']
  for i in range(n_images):
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = expand_dims(scaled_image, 0)
    # make prediction
    #yhat = model.detect(sample, verbose=0)[0]
    r = model.detect(sample, verbose=0)[0] 
    display_instances(image_name,image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
      
#plot_detectedTable(model,cfg)
def process_image(image):
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

for filename in listdir(images_dir):
    if filename[:-4]+'_'+''+'.jpg' not in listdir('./Test_Results'):
        try:
            img_path=images_dir + filename
            image = skimage.io.imread(img_path)
            image=process_image(image)
            plot_actual_vs_predicted(image,filename, model, cfg)
        except:
            print(f'{filename} not valid')
    