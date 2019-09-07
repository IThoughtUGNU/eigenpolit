# -*- coding: utf-8 -*-


import cv2
import numpy as np
from numpy import zeros
from numpy.linalg import norm

# Idea:
# - Try to classify the test face from the dataset, with variable eigen components
# - If it fails to converge (to an image number) after eigenvector 20 it's probably not a face
# - If it fails to converge after eigenvector 25 it's even more unlikely to be a face

class KnownFaceClassifier(object):
    def __init__(self, model_images, threshold = 20, notConvergenceTol = 3, strategy='lbp'):
        from .FaceEigenClassifier import FaceEigenClassifier
        #self.u_list = []
        #self.A = None
        #self.mean_face = None
        #self.weights_lists = []
        #self.images = model_images
        #self.threshold = threshold
        
        self.faceEigenClassifier = FaceEigenClassifier(model_images, threshold)
        self.strategy = strategy
        self.threshold = threshold
        self.notConvergenceTol = notConvergenceTol
    
    def MakeFromLbpModel(model_images, threshold = 20):
        classifier = KnownFaceClassifier(model_images, threshold=20,strategy='lbp')
        return classifier
        
    def MakeFromGenericModel(model_images, threshold = 20, strategy='lbp'):
        from ImageUtils import lbpPreProcess
        new_model_images = []
        if strategy == 'lbp':
            for image in model_images:
                new_model_images.append(lbpPreProcess(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)))
                
        classifier = KnownFaceClassifier(new_model_images, threshold=20,strategy='lbp')
        return classifier
    
    def fit(self, nmin: int, nmax: int,log=False):
        self.faceEigenClassifier.fit(nmin, nmax, log)
            
    def test(self, test_image, target_dim=(100,100), double=False, gray=True,log=False,already_processed=False):
        from .ImageUtils import lbpPreProcess
        if self.strategy == 'lbp' and not already_processed:
            test_image = lbpPreProcess(cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY))
        
        v, mins = self.faceEigenClassifier._test(test_image,target_dim, double, gray, log)
        
        #   #for i in range(threshold,self.nmax-self.nmin):
        unique_mins = set(mins[self.threshold:])   
        return len(unique_mins) <= self.notConvergenceTol
        #return v

if __name__ == "__main__":
    from cv2 import imread
    from FaceRecognition import readFilesRecursively
    from FaceRecognition import faceRecognition, weights
    # ------------------ CONSTANTS --------------------------
    
    INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
    
    TEST_IMAGE_PATH0 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\salvini\salvini003.jpg'
    TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\io\rugg001.png'
    TEST_IMAGE_PATH2 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\io\rugg001.png'
    TEST_IMAGE_PATH3 = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\renzi\renzi001.jpg'
    
    MAX_N_EIGENVECTORS = 50
    # ------------------ Actual code ------------------------
    input_path = INPUT_PATH
    
    images = readFilesRecursively(input_path)
    knownFaceClassifier = KnownFaceClassifier.MakeFromLbpModel(images,threshold=20)
    
    knownFaceClassifier.fit(3, 50, log=True)
    print("FIT!")
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH0), already_processed=True)
    print("is test a known face:", is_a_known_face)
    
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH), already_processed=True)
    print("is test a known face:", is_a_known_face)

    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH2), already_processed=False)
    print("is test a known face:", is_a_known_face)
    is_a_known_face = knownFaceClassifier.test(imread(TEST_IMAGE_PATH3), already_processed=True)
    print("is test a known face:", is_a_known_face)
    









