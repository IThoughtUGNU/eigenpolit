# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 18:54:55 2019

@author: ruggi
"""

import os

input_path = r'C:\dev\python\nm4cs\eigenfaces\dataset\raw\dimaio'

supported_exts = ['.jpg', '.jpeg', '.png']

# ------------------ File gathering ------------------ 

files_to_process = []

for root, subdirs, files in os.walk(input_path):
    for filename in files:
        ext = os.path.splitext(filename)[1]
        if ext in supported_exts:
            files_to_process.append(os.path.join(root, filename))
            
print(files_to_process)

i = 1

for imagePath in files_to_process:
    #import ntpath
    #image_filename = os.path.basename(imagePath)
    path = os.path.dirname(imagePath)
    
    length_int_places = 3
    frameNum = format(i, '0%dd' % length_int_places)
    os.rename(imagePath, os.path.join(path, "dimaio%s.jpg" % frameNum))
    
    i += 1