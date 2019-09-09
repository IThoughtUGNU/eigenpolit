# -*- coding: utf-8 -*-

import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

class EigenfaceModel(object):
    def __init__(self,images,labels, nmin:int = 3, nmax:int=20,m=None,n=None):
        from .FaceRecognition import faceRecognition, weights
        
        u, A, mean_face = faceRecognition(images, nmax,m,n)
        u = u[:,nmin:]
        self.u = u
        self.A = A
        self.mean_face = mean_face
        self.labels = labels
        
        # Calculate weights of Training Images
        self.weights_images = weights(A,u, train=True,height=m,width=n)
        self.nmin, self.nmax = nmin, nmax
        self.m, self.n = m,n
        
    def __str__(self):
        return "EigenfaceModel(nmin: %d, nmax %d)" % (self.nmin, self.nmax)
        
    def projectOntoEigenspace(self, test_image):
        from .FaceRecognition import weights
        u,mean_face = self.u, self.mean_face
        
        # Subtract average face from test image
        test_minus_mean = test_image - mean_face
        # Calculate weights of Test Images
        weights_test_image = weights(test_minus_mean, u, train=False,height=self.m, width=self.n)
        return weights_test_image
        
    def exportTrainPca(self):
        return self.weights_images.T

    def noFitTest(self,test_image,k=1):
        return self._noFitTest(test_image, k)[0]
    
    def _noFitTest(self, test_image,k=1):
        from .ImageUtils import flattenImage
        from .FaceRecognition import weights
        from numpy import zeros, argsort
        from numpy.linalg import norm
        
        assert(k>=1)
        
        u,mean_face = self.u, self.mean_face
        if len(test_image.shape) > 1:
            # If the test image is not given already flatten, let's flatten it.
            test_image = flattenImage(test_image)
        
        # Subtract average face from test image
        test_minus_mean = test_image - mean_face
        
        
        # Calculate weights of Test Images
        weights_test_image = weights(test_minus_mean, u, train=False,height=self.m, width=self.n)
        
        # Find number of images in training dataset
        n_images = self.weights_images.shape[1]
        
        err = zeros(n_images)
        
        for i in range(n_images):
            err[i] = norm(weights_test_image - self.weights_images[:,i])
        
        arg_sorted_err = argsort(err) 
        #sorted_err = err[arg_sorted_err]
        min_positions = arg_sorted_err[:k]
        
        #min_pos = err.argmin()
        #min_error = err[min_pos]
        
        if k == 1:
            match_class = self.labels[min_positions[0]]
        elif k > 1:
            matching_labels = self.labels[min_positions]
            match_class = most_common(matching_labels)
        
        return match_class, err[min_positions[0]]
    
    