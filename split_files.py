#!/usr/bin/python

import os
import sys
from os.path import join, isfile
from shutil import move, copy
from tqdm import tqdm

train_size = 0.9

dir = './data/COCO/'
img_dir = './data/COCO/images'
masks_dir = './data/COCO/binary_masks'
if len(sys.argv) > 1:
    dir = sys.argv[1]

if not os.path.exists(join(dir, 'train')):
    os.mkdir(join(dir, 'train'))
    os.mkdir(join(dir, 'train', 'binary_masks'))
    os.mkdir(join(dir, 'train', 'images'))
if not os.path.exists(join(dir, 'val')):
    os.mkdir(join(dir, 'val'))
    os.mkdir(join(dir, 'val', 'binary_masks'))
    os.mkdir(join(dir, 'val', 'images'))

imgs = [f for f in os.listdir(img_dir) if isfile(join(img_dir, f)) and
                                      f.endswith('jpg')]
print(imgs[0])
print("Found {} images.".format(len(imgs)))
train_size = int(len(imgs)*train_size)

print("Creating train..")
for img in tqdm(imgs[:train_size]):
    #copy(join(img_dir, img), join(dir, 'train/images/', img))
    img = img.replace('jpg', 'png')
    #copy(join(masks_dir, img), join(dir, 'train/binary_masks/', img))
print("Creating val..")
for img in tqdm(imgs[train_size:]):
    #copy(join(img_dir, img), join(dir, 'val/images/', img))
    img = img.replace('jpg', 'png')
    #copy(join(masks_dir, img), join(dir, 'val/binary_masks/', img))

print("Checking amounts of files in each folder")
print("In val/images:", len(os.listdir(join(dir, 'val/images/'))))
print("In val/binary_masks:", len(os.listdir(join(dir, 'val/binary_masks/'))))
print("In train/images:", len(os.listdir(join(dir, 'train/images/'))))
print("In train/binary_masks:", len(os.listdir(join(dir, 'train/binary_masks/'))))
