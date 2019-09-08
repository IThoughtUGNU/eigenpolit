# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 01:55:51 2019

@author: ruggi
"""

import os
import cv2

# Costanti 

PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\raw\renzi'
SAVE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output\renzi'
SAVE_PATH_CNN = r'C:\dev\python\nm4cs\eigenfaces\dataset\output\salvini_cnn'

supported_exts = ['.jpg', '.jpeg', '.png']

cascPath = "haarcascade_frontalface_default.xml"

# ------------------ ------------ ESECUZIONE ------------------ ---------------

input_path = PATH # fare poi con args
save_path = SAVE_PATH
save_path_cnn = SAVE_PATH_CNN
super_input_path = None

if __name__ == "__main__":
    from pathlib import Path
    import argparse
    parser = argparse.ArgumentParser(description="Face extractor from images")
    
    # ------------------ Input parsing  ------------------ 
    parser.add_argument('--superinputpath', type=str,
                        help='Folder which contains folders of images, all taken')
    
    parser.add_argument('--inputpath', type=str,
                        help='Folder which contains folders of images, all taken')
    
    parser.add_argument('--rootsavepath', type=str, 
                        help='(Root) Folder to save folders of extracted faces')
    
    parser.add_argument('--savepath', type=str, 
                        help='Folder to save extracted faces')
    
    parser.add_argument('--overwrite', action='store_true',
                default=False,
                dest='overwrite',
                help='Apply Local Binary Pattern function')
    args = parser.parse_args()
    
    if args.superinputpath is None:
        if args.inputpath is None:
            print("[WARNING: None inputpath. Going with %s]" % PATH)
    else:
        super_input_path = args.superinputpath
    
    root_save_path = args.rootsavepath
    if root_save_path is None:
        if args.savepath is None:
            print("[WARNING: None savepath. Going with %s]" % SAVE_PATH)
    
    input_path = args.inputpath if args.inputpath is not None else PATH
    save_path  = args.savepath  if args.savepath  is not None else PATH
    overwrite = args.overwrite
    
    if super_input_path is not None:
        subfolders = [f.path for f in os.scandir(super_input_path) if f.is_dir() ]
    else:
        subfolders = [input_path]
    
    for j, folder in enumerate(subfolders):
        input_path = folder
        if root_save_path is not None:
            save_path = os.path.join(root_save_path, os.path.basename(folder))
            
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print(save_path)
        # ------------------ File gathering ------------------ 
    
        files_to_process = []
        
        for root, subdirs, files in os.walk(input_path):
            for filename in files:
                ext = os.path.splitext(filename)[1]
                if ext in supported_exts:
                    files_to_process.append(os.path.join(root, filename))
                    
        print(files_to_process)
        
            
        # ------------------ Face Detection ------------------ 
        
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)
        
        
        for imagePath in files_to_process:
            #import ntpath
            image_filename = os.path.basename(imagePath)
            
            # Read the image
            image = cv2.imread(imagePath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1, # 1.1
                minNeighbors=5,
                minSize=(30, 30)
                #flags = cv2.CV_HAAR_SCALE_IMAGE
            )
            
            print("Found {0} faces!".format(len(faces)))
            
            count = 0
            filename_without_ext = os.path.splitext(image_filename)[0]
            ext = os.path.splitext(image_filename)[1]
            file_save_path = os.path.join(save_path, filename_without_ext + ext)
            if (not overwrite) and Path(file_save_path).is_file():
                continue
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                image_to_save = gray[y:(y+h),x:(x+w)]# gray[x-x//2:w*2,y-y//2:h*2].copy()
                #print(image_filename)
                #print(x, y, w, h)
                #cv2.imshow("Faces found", image)
                #cv2.waitKey(0)
                
                
                if count > 0:
                    filename_without_ext += ("-%d" % count)
                
                cv2.imwrite(file_save_path, image_to_save)
                count+=1
        
        #cv2.imshow("Faces found", image)
        
    cv2.destroyAllWindows()    
        
        
    















