# -*- coding: utf-8 -*-

from numpy import zeros

class DatasetModel(object):
    def __init__(self, folderpath,shallowMirrorObject=None):
        import numpy as np
        self.folderpath = folderpath
        
        self.partitionIndexes = None
        if shallowMirrorObject is None:
            if type(folderpath) != list:
                self.images, self.files, self.m, self.n = _readFilesRecursively(folderpath)
            else:
                self.partitionIndexes = [0]
                for i, folder in enumerate(folderpath):
                    if i == 0:
                        self.images, self.files, self.m, self.n = _readFilesRecursively(folder)
                            
                        self.partitionIndexes.append(len(self.images))
                        continue
                    
                    images, files, self.m, self.n = _readFilesRecursively(folder)
                    self.partitionIndexes.append(len(images)+self.partitionIndexes[-1])
                    self.images = np.concatenate((self.images, images))
                    self.files.extend(files)
                
                
            self.labels = zeros(len(self.images))
        else:
            self.images, self.files = shallowMirrorObject.images, shallowMirrorObject.files
            self.labels = shallowMirrorObject.labels
            self.partitionIndexes = shallowMirrorObject.partitionIndexes
            self.m, self.n = shallowMirrorObject.m, shallowMirrorObject.n
        
    def labeledByFolderOfFiles(self,namedClassesList):
        newShallowDatasetModel = DatasetModel(self.folderpath, self)
        from pathlib import Path
        for i, file in enumerate(self.files):
            filepath = Path(file)
            just_path = filepath.parents[0]
            just_folder_name = just_path.relative_to(just_path.parents[0])
            
            candidate_name = just_folder_name.__str__().lower() 
            if candidate_name in namedClassesList:
                # Possibly more efficient to use a dictionary or binary tree if the classes are many
                newShallowDatasetModel.labels[i] = namedClassesList.index(candidate_name)
            else:
                print("[WARNING | DatasetModel.labeledByFolderOfFiles] UNKNOWN CLASS FOR ITEM %d" % i)
                print("---> at: %s" % file)
            
            
        return newShallowDatasetModel
    
    def exportedAsClassicDataset(self):
        return self.images, self.labels # X, y = model.exportedAsClassicDataset()
    
    def getDim(self):
        return self.m, self.n

def _readFilesRecursively(folderpath,supported_exts=['.jpg','.jpeg','.png'],preprocessFunction=lambda img : img):
    import os
    from cv2 import cvtColor, imread, COLOR_BGR2GRAY
    from numpy import reshape, array    
    try:
        from .ImageUtils import im2double
    except:
        from FaceRecognition.ImageUtils import im2double
    #try:
    #    from ImageUtils import im2double
    #except ModuleNotFoundError:
    #    from .ImageUtils import im2double
    
    files_list = []
    for root, subdirs, files in os.walk(folderpath):
        for file in files:
            ext = os.path.splitext(file)[1]
            if ext in supported_exts:
                files_list.append(os.path.join(root, file))
    
    #n_images = len(files_list)
    images = []
    for i, filename in enumerate(files_list):
        
        #print(filename)
        image_read = imread(filename)
        if image_read is None:
            print("[Warning] None image; ", end="")
            del files_list[i]
            
            continue
        image = im2double(preprocessFunction(cvtColor(image_read, COLOR_BGR2GRAY)))
        #print(image.size)
        [m,n] = image.shape
        images.append(reshape(image, m*n))
    
    return array(images), files_list, m,n

def readFilesRecursively(folderpath,supported_exts=['.jpg','.jpeg','.png'],preprocessFunction=lambda img : img):
    return _readFilesRecursively(folderpath, supported_exts, preprocessFunction)[0]















