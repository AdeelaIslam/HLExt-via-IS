#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:15:09 2019

@author: v1
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:25:19 2019

@author: v1
"""

# plot one photograph and mask
from os import listdir
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
from matplotlib import pyplot
#from mrcnn.visualize import display_instances
#from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage
import numpy as np
import json
import cv2
######################### Prepare Dataset ##################################


# class that defines and loads the kangaroo dataset
class LineExtractionDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=0):
        # define one class
        self.add_class("dataset", 1, "line")
        # define data locations
        if is_train==0:
            images_dir=dataset_dir+'/train_images/'
            annotations_dir=dataset_dir+'/train_annots/'
        elif is_train==1:
            images_dir=dataset_dir+'/test_images/'
            annotations_dir=dataset_dir+'/test_annots/'
        '''
        elif is_train==2:
            images_dir=dataset_dir+'/val_images/'
            annotations_dir=dataset_dir+'/val_annots/'
        '''
        # find all images
        c=0
        for filename in listdir(images_dir):
            # extract image id
            image_id = filename[:-4] # filename is for instance 0001.jpg so 
            print("image_id : ",image_id)
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.json'
            c=c+1
            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annots_path=ann_path)  
    # extract bounding boxes from an annotation file
    def extract_boxes(self,image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location , xml file
        path_jsonFile = info['annots_path']
        path_image = info['path']
        #print(path_image)
        with open(path_jsonFile, 'r') as f:
            table0_dict = json.load(f)
        #print(table0_dict)
        #print(table0_dict['shapes'])
        lineObjects = list()
        for x in (table0_dict['shapes']):
            #print(x['label'])
            x_values=zeros(len(x['points']),dtype='int')
            y_values=zeros(len(x['points']),dtype='int')
            x_idx=0
            y_idx=0
            #print(len(x['points']))
            for y in x['points']:
                #print("next point")
        #        print(y[0])
        #        print(y[1])
                x_values[x_idx]=int(round(y[0]))
                y_values[y_idx]=int(round(y[1]))
                x_idx = x_idx + 1
                y_idx = y_idx + 1
                
            boxeswithlabel={"label":x['label'] , "x_coordinates":x_values , "y_coordinates":y_values}
            #print(boxeswithlabel)
            lineObjects.append(boxeswithlabel)
        
        #print(rowColObjects)
        
#        for obj in rowColObjects:
#            print(obj['label'])
#            print(obj['x_coordinates'])
#            for x in obj['x_coordinates']:
#                print(x)
#            print(obj['y_coordinates'])
#            for y in obj['y_coordinates']:
#                print(y)
        img=cv2.imread(path_image)
        height = img.shape[0]
        width = img.shape[1]
        return lineObjects, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        #info = self.image_info[image_id]
        # define box file location , xml file
        #print(img_id)
        #image = cv2.imread(info['path'])
        # load XML
        lineObj, w, h = self.extract_boxes(image_id)
        
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(lineObj)], dtype='uint8')
#        print(rows)
#        print(cols)
        # create masks
        class_ids = list()
        i=0
        for obj in lineObj:
            class_name = obj['label']
#            print(len(obj['y_coordinates']))
#            print(len(obj['x_coordinates']))
            rows=zeros(len(obj['y_coordinates']),dtype='int')
            cols=zeros(len(obj['x_coordinates']),dtype='int')
            #rows =obj['y_coordinates']
            r_idx=0
            for y in  obj['y_coordinates']:
                rows[r_idx]=y
                r_idx = r_idx + 1
                #print(type(y))
            #cols=obj['x_coordinates']
            c_idx=0
            for x in  obj['x_coordinates']:
                cols[c_idx] = x
                c_idx = c_idx + 1
                #print(type(x))
            #print(rows)
#            print(type(rows[0]))
            #print(cols)
#            print(type(cols[0]))
#            pyplot.imshow(image)
#            pyplot.scatter((cols[2]) , (rows[2]) , marker='o')
#            pyplot.show()
            rr, cc = skimage.draw.polygon(rows,cols)
#            print(rr)
#            print(cc)
            masks[rr, cc, i] = 1
#            box = boxes[i]
#            #print(box[0],box[1],box[2],box[3])
#            #row_s , col_s is the starting position (i,j) of box
#            #row_e , col_e is the ending position (i,j) of box
#            row_s, row_e = box[0], box[2]
#            col_s, col_e = box[1], box[3]
#            #print(row_s,col_s,row_e,col_e)
#            masks[row_s:row_e, col_s:col_e, i] = 1
#            #print(masks.shape)
            class_ids.append(self.class_names.index(class_name))
        
            i = i+1
  
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


####################### Testing Dataset Examples ############################


## train set
#train_set = KangarooDataset()
#train_set.load_dataset('/home/v1/Mask_RCNN/kangaroo', is_train=True)
#train_set.prepare()
## Training images have image_ids from 0 to 130
## Test/validation images have image_ids from 0 to 31
## load an image
#image_id = 0
#image = train_set.load_image(image_id)
#print(image.shape)
## load image mask
#mask, class_ids = train_set.load_mask(image_id)
#print(mask.shape)
## plot image
##pyplot.imshow(image)
## plot mask
#for i in range (mask.shape[2]):
#    pyplot.imshow(image)
#    pyplot.imshow(mask[:, :, i], cmap='gray', alpha=0.5)
#    pyplot.show()
#print(train_set.image_reference(image_id))



#test_set = KangarooDataset()
#test_set.load_dataset('/home/v1/Mask_RCNN/kangaroo', is_train=False)
#test_set.prepare()
#print('NO of Test images: %d' % len(test_set.image_ids))
#print('No of Train images: %d' % len(train_set.image_ids))


## plot first few images
#for i in range(9):
#    # define subplot
#    pyplot.subplot(330 + 1 + i)
#    # plot raw pixel data
#    image = train_set.load_image(i)
#    pyplot.imshow(image)
#    # plot all masks
#    mask, _ = train_set.load_mask(i)
#    for j in range(mask.shape[2]):
#        pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
## show the figure
#pyplot.show()



## enumerate all images in the dataset
#for image_id in train_set.image_ids:
#    # load image info
#    info = train_set.image_info[image_id]
#    # display on the console
#    print(info)

#Finally, the mask-rcnn library provides utilities for displaying images 
#and masks. We can use some of these built-in functions to confirm that the
# Dataset is operating correctly.
#For example, the mask-rcnn library provides the 
#mrcnn.visualize.display_instances() function that will show a photograph with
# bounding boxes, masks, and class labels. This requires that the bounding 
#boxes are extracted from the masks via the extract_bboxes() function.

#bbox = extract_bboxes(mask)
#display_instances(image, bbox, mask, class_ids, train_set.class_names)



################################ Start Training ##############################

#set paths
dataset_path ='./Project Data/'
model_directory_path='./Models/'
pretrained_weights_path='mask_rcnn_coco.h5'        


#define a configuration for the model
class LineExtractionConfig(Config):
    # Give the configuration a recognizable name
    NAME = "lineExtraction_cfg"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50
 
# prepare config
#config = GridConfig()

# prepare train set
print("############################# TRAIN #########################")
train_set = LineExtractionDataset()
train_set.load_dataset(dataset_path , is_train=0)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
##load an image
#image_id=39
#for image_id in range(len(train_set.image_ids)):
#    print(image_id)
#    print(train_set.image_reference(image_id))
#    image = train_set.load_image(image_id)
#    #print(image.shape)
#    # load image mask
#    mask, class_ids = train_set.load_mask(image_id)
#    #print(mask.shape)
#    # plot image
#    #pyplot.imshow(image)
#    #plot mask
#    #print(mask.shape[2])
#    for i in range (mask.shape[2]):
#        pyplot.imshow(image)
#        pyplot.imshow(mask[:, :, i], cmap='winter', alpha=0.3)
#        pyplot.show()
#print(train_set.image_reference(image_id))
# prepare test/val set
print("############################# TEST #########################")
test_set = LineExtractionDataset()
test_set.load_dataset(dataset_path , is_train=1)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))
#for image_id in range(len(test_set.image_ids)):
#    print(image_id)
#    print(test_set.image_reference(image_id))
#    image = test_set.load_image(image_id)
#    #print(image.shape)
#    # load image mask
#    mask, class_ids = test_set.load_mask(image_id)
#    #print(mask.shape)
#    # plot image
#    #pyplot.imshow(image)
#    #plot mask
#    #print(mask.shape[2])
#    for i in range (mask.shape[2]):
#        pyplot.imshow(image)
#        pyplot.imshow(mask[:, :, i], cmap='winter', alpha=0.3)
#        pyplot.show()
'''
print("############################# Validation #########################")
val_set = LineExtractionDataset()
val_set.load_dataset(dataset_path , is_train=2)
val_set.prepare()
print('Validation: %d' % len(val_set.image_ids))
'''
#for image_id in range(len(val_set.image_ids)):
#    print(image_id)
#    print(val_set.image_reference(image_id))
#    image = val_set.load_image(image_id)
#    #print(image.shape)
#    # load image mask
#    mask, class_ids = val_set.load_mask(image_id)
#    #print(mask.shape)
#    # plot image
#    #pyplot.imshow(image)
#    #plot mask
#    #print(mask.shape[2])
#    for i in range (mask.shape[2]):
#        pyplot.imshow(image)
#        pyplot.imshow(mask[:, :, i], cmap='winter', alpha=0.3)
#        pyplot.show()
# prepare config
config = LineExtractionConfig()
config.display()
# define the model
model = MaskRCNN(mode='training', model_dir=model_directory_path , config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights(pretrained_weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=30, layers='all')

