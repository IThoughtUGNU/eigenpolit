# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:25:31 2019

@author: ruggi
"""

import cv2
import os
 
def resizedImageAtPath(image_path, dim=(100,000)):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
     
    #print('Original Dimensions : ',img.shape)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

if __name__ == "__main__":
    from pathlib import Path
    input_path = r'C:\dev\python\nm4cs\eigenfaces\dataset\output\conte'
    save_path = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test'
    
    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir() ]
    if len(subfolders) == 0:
        subfolders = [input_path]
    
    for folder_path in subfolders:
        folder_simple_name = os.path.basename(folder_path)
        image_paths = [f.path for f in os.scandir(folder_path) if not f.is_dir() ]
        
        image_save_path = os.path.join(save_path, folder_simple_name)
        assert(save_path in image_save_path)
        
        if not os.path.exists(image_save_path):
            os.makedirs(image_save_path)
        
        for image_path in image_paths:
            new_image = resizedImageAtPath(image_path, dim=(100,100))
            
            new_image_path = os.path.join(image_save_path, os.path.basename(image_path))
            assert(image_save_path in new_image_path)
            
            if (Path(new_image_path).is_file()):
                continue
            cv2.imwrite(new_image_path, new_image)