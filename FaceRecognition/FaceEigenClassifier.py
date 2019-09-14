# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 16:59:56 2019

@author: ruggi
"""

#if __name__ == "__main__":
#    import os
#    
#    abspath = os.path.abspath(__file__)
#    dname = os.path.dirname(abspath)
#    os.chdir(dname)


import cv2
import numpy as np
from numpy import zeros
from numpy.linalg import norm

# Idea:
# - Try to classify the test face from the dataset, with variable eigen components
# - If it fails to converge (to an image number) after eigenvector 20 it's probably not a face
# - If it fails to converge after eigenvector 25 it's even more unlikely to be a face

class FaceEigenClassifier(object):
    def __init__(self, model_images, threshold = 20):
        self.u_list = []
        self.A = None
        self.mean_face = None
        self.weights_lists = []
        self.images = model_images
        self.threshold = threshold
        
    def fit(self, nmin: int, nmax: int,log=False):
        assert (nmax > nmin)
        from .FaceRecognition import buildFacespace, weights
        
        self.nmin = nmin
        self.nmax = nmax
        
        for n in range(nmin, nmax):
            u, A, mean_face = buildFacespace(self.images, n)
            u = u[:,nmin:]
            
            self.u_list.append(u)
            if self.A is None:
                self.A = A
            
            if self.mean_face is None:
                self.mean_face = mean_face
            
            # Calculate weights of Training Images
            #u = self.u_list[n-self.nmin]
            weights_images = weights(A,u, train=True)
            self.weights_lists.append(weights_images)
            if log:
                print("[LOG | FaceEigenClassifier.fit] Iterate #%d" % n)
            
    def _test(self, test_image, target_dim=(100,100), double=False, gray=True,log=False):
        from .ImageUtils import im2double
        from .FaceRecognition import weights
        
        if len(test_image.shape) == 3 and gray:
            # If the test image is not gray, gray it out
            test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
            
        if not double:
            test_image = im2double(test_image)
        
        if len(test_image.shape) == 2:
            if test_image.shape[0] != target_dim[0] or test_image.shape[1] != target_dim[1]:
                # Resize image if is not well sized
                test_image = test_image.resize(target_dim[0], target_dim[1])

            # Flatten the image
            test_image = np.reshape(test_image, target_dim[0] * target_dim[1])
            
        test_minus_mean = test_image - self.mean_face
        # A = self.A
       
        error_plot = zeros(self.nmax-self.nmin)
        mins = zeros(self.nmax-self.nmin)
        
        mins_differences = zeros(self.nmax-self.nmin)
        
        for n in range(self.nmin, self.nmax):
            weights_images = self.weights_lists[n-self.nmin]
            u = self.u_list[n-self.nmin]
            # Calculate weights of Test Images
            weights_test_image = weights(test_minus_mean, u, train=False)
            
            # Find number of images in training dataset
            n_images = weights_images.shape[1]
            
            err = zeros(n_images)
            
            for i in range(n_images):
                err[i] = norm(weights_test_image - weights_images[:,i])
        
            min_pos = err.argmin()
            min_error = err[min_pos]
            
            if log: 
                print("Number of eigen components: %d --> Error %d --> Image # %d.\n" % (n,min_error,min_pos))
                      
            i = n - self.nmin
            error_plot[i] = min_error
            mins[i] = min_pos

        threshold = self.threshold
        assert(self.nmax > threshold) # "Can't make a prediction; insufficient # of eigenfaces"
        for j in range(threshold-self.nmin, self.nmax-self.nmin):
            mins_differences[j] = mins[j] - mins[j-1]
            
        changeCount = 0
        for i in range(threshold-self.nmin, self.nmax-self.nmin):
            if mins_differences[i] != 0:
                changeCount += 1
            
        if (mins_differences[threshold-self.nmin:] == 0).all():
            return True, mins, changeCount
        else:
            return False, mins, changeCount

    def test(self, test_image, target_dim=(100,100), double=False, gray=True,log=False):
        return self._test(test_image, target_dim, double, gray,log)[0]
    

if __name__ == "__main__":
    from cv2 import imread
    
    from DatasetModel import readFilesRecursively
    #from FaceRecognition import readFilesRecursively
    # ------------------ CONSTANTS --------------------------
    
    INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
    
    TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\salvini\salvini003.jpg'
    TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\io\rugg001.png'
    
    MAX_N_EIGENVECTORS = 50
    # ------------------ Actual code ------------------------
    input_path = INPUT_PATH
    
    images = readFilesRecursively(input_path)
    faceEigenClassifier = FaceEigenClassifier(images,threshold=17)
    
    faceEigenClassifier.fit(3, 50, log=True)
    print("FIT!")
    is_a_face = faceEigenClassifier.test(imread(TEST_IMAGE_PATH))
    
    print("is test a face:", is_a_face)



















