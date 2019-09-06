# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:36:10 2019

@author: ruggi
"""

import numpy as np
import cv2
from cv2 import imread
from numpy import reshape
try:
    from ..ImageProcessing.lbp import lbp_calculated_pixel
except:
    from ImageProcessing.lbp import lbp_calculated_pixel
    
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

def readImageAsGray8(filename,new_dim=None):
    image = cv2.cvtColor(imread(filename), cv2.COLOR_BGR2GRAY)
    
    if new_dim is not None:
        image = image.resize(new_dim[0], new_dim[1])
    
    return image

def readImageAsGray(filename,new_dim=None,preprocessFunction=lambda img : img):
    image = im2double(preprocessFunction(cv2.cvtColor(imread(filename), cv2.COLOR_BGR2GRAY)))
    
    if new_dim is not None:
        image = image.resize(new_dim[0], new_dim[1])
    
    return image

def flattenImage(image):
    m,n = image.shape
    
    new_image = reshape(image, m*n)
    return new_image

def lbpPreProcess(image_gray):
    [height, width] = image_gray.shape[0:2]
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
             img_lbp[i, j] = lbp_calculated_pixel(image_gray, i, j)
        
    return cv2.cvtColor(img_lbp, cv2.COLOR_BGR2GRAY)