# -*- coding: utf-8 -*-


class EigenfaceModel(object):
    def __init__(self,images,labels, nmin:int = 3, nmax:int=20):
        from .FaceRecognition import faceRecognition, weights
        
        u, A, mean_face = faceRecognition(images, nmax)
        u = u[:,nmin:]
        self.u = u
        self.A = A
        self.mean_face = mean_face
        self.labels = labels
        
        # Calculate weights of Training Images
        self.weights_images = weights(A,u, train=True)

    def projectOntoEigenspace(self, test_image):
        from .FaceRecognition import weights
        u,mean_face = self.u, self.mean_face
        
        # Subtract average face from test image
        test_minus_mean = test_image - mean_face
        # Calculate weights of Test Images
        weights_test_image = weights(test_minus_mean, u, train=False)
        return weights_test_image
        
    def exportTrainPca(self):
        return self.weights_images.T

    def noFitTest(self, test_image):
        from .ImageUtils import flattenImage
        from .FaceRecognition import weights
        from numpy import zeros
        from numpy.linalg import norm
        
        
        u,mean_face = self.u, self.mean_face
        if len(test_image.shape) > 1:
            # If the test image is not given already flatten, let's flatten it.
            test_image = flattenImage(test_image)
        
        # Subtract average face from test image
        test_minus_mean = test_image - mean_face
        
        
        # Calculate weights of Test Images
        weights_test_image = weights(test_minus_mean, u, train=False)
        
        # Find number of images in training dataset
        n_images = self.weights_images.shape[1]
        
        err = zeros(n_images)
        
        for i in range(n_images):
            err[i] = norm(weights_test_image - self.weights_images[:,i])
        
        min_pos = err.argmin()
        #min_error = err[min_pos]
        
        #match_img = images[min_pos]
        #match_img = match_img.reshape(100,100)
        match_class = self.labels[min_pos]
        return match_class
    
    