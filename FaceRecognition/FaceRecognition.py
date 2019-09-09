# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 23:04:38 2019

@author: ruggi
"""

import os
import cv2
#import numpy as np

supported_exts = ['.jpg', '.jpeg', '.png']
from oct2py import Oct2Py
oc = Oct2Py()

def c(arr): # given a list or an array, return it as column vector (NumPy)
    from numpy import array
    
    return array([arr]).T


def no_preprocess_f(image):
    return image


def readFiles(folderpath):
    from cv2 import cvtColor, imread
    from numpy import reshape, array
    try:
        from ImageUtils import im2double
    except ModuleNotFoundError:
        from .ImageUtils import im2double
    
    files_list = [f.path for f in os.scandir(folderpath) if not f.is_dir() ]
    n_images = len(files_list)
    images = []
    for i in range(n_images):
        filename = files_list[i]
        ext = os.path.splitext(filename)[1]
        if ext in supported_exts:
        
            image = im2double(cvtColor(imread(filename), cv2.COLOR_BGR2GRAY))
            [m,n] = image.shape
            images.append(reshape(image, m*n))
    
    return array(images)

def faceRecognition(images, nc,m=None,n=None):
    """
    Function faceRecognition
    
    INPUT:
        - images images of dataset
        - nc     number of best eigenvectors to selet
        
    OUTPUT:
        - Eigenfaces
        - Image Matrix
        - Average Face Vector
    
    ALGORITHM 
    
    1) Calculate mean of all images
    2) Subtract mean from all images
    3) Stack all the image vectors as columns in a matrix (image matrix)
    4) Calculate the best k eigenvectors
    
    """

    import numpy as np
    from numpy import zeros, sqrt, diag #, identity
    from numpy.linalg import eig, svd
    
    n_images, mn = images.shape
    
    if m == None:
        m = n = int(sqrt(mn)) # assumo immagini quadrate
    
    sum_images = zeros((m * n), dtype=np.float32)
    
    for i in range(n_images):
        sum_images[:] += images[i,:]
        
    mean_face = sum_images / n_images
    
    A = zeros(images.T.shape)
    A[:,:] = images.T[:,:]
    A[:,:] -= c(mean_face)
    
    [v,D] = oc.eigs(A.T.dot(A), nc,nout=2)
    #w,V = eig(A.T.dot(A))
    
    # # D = identity(A.shape[1]) * c(w) # COLUMN-WISE MULTIPLICATION
    
    u = A.dot(v)
    
    #U, S, Vh = svd(A)
        
    return u, A, mean_face
    #return U, S, Vh, mean_face


def selectKComponents(eigenvalues, target_variance):
    assert(target_variance <= 1)
    
    k = 0
    # evaluate the number of principal components needed to represent 
    #   (target_variance*100)% Total variance.
    eigsum = sum(eigenvalues)
    csum = 0
    for i in range(len(eigenvalues)):
        csum += eigenvalues[i]
        total_variance = csum / eigsum
        if total_variance > target_variance:
            k = i
            break
        
    return k

def weights(A, u, train: bool, height=100, width=100):
    from numpy import zeros
    output = None
    if train:
        n_images = A.shape[1]
        # Ω of some image = array [M']
        # where Ω = [ω1, ω2, ... ω_k] for k = 1,...,M'
        # and M' = number of eigenfaces (eigenvectors)
        
        # super-Ω matrix: array [M', n]
        # where n = number of images
        M_eig = u.shape[1]
        Weights_matrix = zeros((M_eig,A.shape[1])) # 
        
        superΩ = zeros((M_eig, n_images))
        for i in range(n_images):
            Ω_i = zeros(M_eig)
            for k in range(M_eig): # no. of eigenvectors
                image_vector = A[:,i]
                ω_k = u[:,k].T.dot(image_vector)
                Ω_i[k] = ω_k
                
            superΩ[:,i] = Ω_i
            #image_vector = A[:,i]
            #Weights_matrix[:, i] = u.T.dot(image_vector)
            #image_vector_eigenfaces = u.dot(Weights_matrix[:,i])
            #image_matrix = image_vector_eigenfaces.reshape(height,width)
            ##image = cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
        
        output = superΩ
    else:
        image_vector = A
        weights_vector = u.T.dot(image_vector)
        #image_vector_eigenfaces = u.T.dot(weights_vector)
        output = weights_vector
    return output






