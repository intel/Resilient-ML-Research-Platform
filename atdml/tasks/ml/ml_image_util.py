#! /usr/bin/python
'''
#Copyright (C) 2018 Intel Corporation
#
#SPDX-License-Identifier: Apache-2.0
'''


# standard library imports
from argparse import ArgumentParser
import sys, ConfigParser, pickle
import os, glob
import re
import collections, gzip, mimetypes
import ujson, json, math, zipfile
import numpy as np
from scipy.sparse import csr_matrix
import subprocess

from scipy.stats import entropy
#import pydoop.hdfs as hdfs


#####matplotlib###############
import matplotlib, math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.colors as colors
import matplotlib.cm as cmx
from PIL import Image

from scipy.misc import imread, imresize

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'



# may chg to local dir for offline prediction ########################
CONF_FILE='/home/django/myml/app.config' # at the base dir of the web

config=ConfigParser.ConfigParser()
config.read(CONF_FILE)

# ===================================================== preprocess_image_batch =============
#           img_size, crop_size: tuple (256,256)
def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        # if color_mode=="bgr":
        #    img[:,:,[0,1,2]] = img[:,:,[2,1,0]]

        if crop_size:
            img = img[(img_size[0] - crop_size[0]) // 2 : (img_size[0] + crop_size[0]) // 2 \
                    , (img_size[1] - crop_size[1]) // 2 : (img_size[1] + crop_size[1]) // 2, :];

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch

# ===================================================== undo_image_avg =============
def undo_image_avg(img):
    img_copy = np.copy(img)
    img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
    img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
    img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    return img_copy

# ===================================================== create_imagenet_npy =============
def create_imagenet_npy(path_train_imagenet, len_batch=10000):
    # path_train_imagenet = '/datasets2/ILSVRC2012/train';

    sz_img = [224, 224]
    num_channels = 3
    num_classes = 1000

    im_array = np.zeros([len_batch] + sz_img + [num_channels], dtype=np.float32)
    num_imgs_per_batch = len_batch / num_classes

    dirs = [x[0] for x in os.walk(path_train_imagenet)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order (same as synset_words.txt)
    dirs = sorted(dirs)

    it = 0
    Matrix = [0 for x in range(1000)]

    for d in dirs:
        for _, _, filename in os.walk(os.path.join(path_train_imagenet, d)):
            Matrix[it] = filename
        it = it+1

    it = 0
    # Load images, pre-process, and save
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            print('Processing image number ', it)
            path_img = os.path.join(dirs[k], Matrix[k][u])
            image = preprocess_image_batch([path_img],img_size=(256,256), crop_size=(224,224), color_mode="rgb")
            im_array[it:(it+1), :, :, :] = image
            it = it + 1

    return im_array

# ===================================================== resize_image =============
def resize_image2file(image_fname, output_fname, tgt_size_tuple=(256,256)):
    # resize to 224x224 ==============
    img_in=Image.open(image_fname)
    print "INFO: before image resize=",img_in.size,", to size=", tgt_size_tuple
    #Image.ANTIALIAS  EOLed in pillow 4.1
    #re_img_in=img_in.resize(tgt_size_tuple,Image.ANTIALIAS)
    re_img_in=img_in.resize(tgt_size_tuple,Image.LANCZOS)
    re_img_in.save(output_fname)


# ===================================================== create_imagenet_npy =============
def save_image(image_arr, title, out_fname, dpi=96):
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.imshow(undo_image_avg( image_arr).astype(dtype='uint8'), interpolation=None)
    plt.title(title)
    # hide x,y axis
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    # lower dpi will decrease file size
    plt.savefig(out_fname, bbox_inches='tight', dpi=dpi)
    plt.show()
    
def main():
    print "in main()"
     
 
if __name__ == '__main__':
    __description__ = "utilties for ml"
    main()
