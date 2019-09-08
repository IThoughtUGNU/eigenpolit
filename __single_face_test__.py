# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 14:55:44 2019

@author: ruggi
"""

# ------------------ MODULES ----------------------------


from FaceRecognition.DatasetModel import readFilesRecursively
from FaceRecognition.FaceRecognition import faceRecognition
from FaceRecognition.FaceRecognition import selectKComponents, weights
from FaceRecognition.ImageUtils import readImageAsGray, flattenImage, lbpPreProcess


from cv2 import imshow, imread
import cv2

from numpy import fliplr, zeros
from numpy.linalg import eig, norm
import numpy as np

# ------------------ CONSTANTS --------------------------

INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'

TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\gentiloni\gent001.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\salvini\salvini003.jpg'

n_eigenvectors = 80
# ------------------ Actual code ------------------------
input_path = INPUT_PATH

images = readFilesRecursively(input_path)

u, A, mean_face = faceRecognition(images, n_eigenvectors)

u = u[:,3:]

# Let's take a test image matrix
# and vectorize it.
#TODO: test_image = flatten_image('test.jpg');
test_image_original = readImageAsGray(TEST_IMAGE_PATH)
test_image = flattenImage(test_image_original)

# Subtract average face from test image
test_minus_mean = test_image - mean_face

# Calculate weights of Training Images
weights_images = weights(A,u, train=True)

# Calculate weights of Test Images
weights_test_image = weights(test_minus_mean, u, train=False)

# Find number of images in training dataset
n_images = weights_images.shape[1]

err = zeros(n_images)

for i in range(n_images):
    err[i] = norm(weights_test_image - weights_images[:,i])

min_pos = err.argmin()
min_error = err[min_pos]

match_img = images[min_pos]
match_img = match_img.reshape(100,100)

dim = (200,200)
cv2.imshow("query face", cv2.resize(test_image_original, dim, interpolation = cv2.INTER_AREA) )
cv2.imshow("match_img", cv2.resize(match_img, dim, interpolation = cv2.INTER_AREA) )
cv2.waitKey(0)

#cv2.destroyAllWindows() 

#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

#ax1.imshow(mean_face.reshape(100,100), extent=[100,100])
#ax1.set_title('Mean Face')

#ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
#ax2.set_title('Auto-scaled Aspect')

#ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
#ax3.set_title('Manually Set Aspect')

#plt.tight_layout()
#plt.show()


    

#figure,subplot(4,4,1)
#imagesc(reshape(mean_matrix, [h,w]))
#colormap gray
#for i = 1:15
#    subplot(4,4,i+1)
#    imagesc(reshape(V(:,i),h,w))
#end

cv2.destroyAllWindows() 


