# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 17:53:54 2019

@author: ruggi
"""

# ------------------ MODULES ----------------------------


from FaceRecognition.DatasetModel import readFilesRecursively
from FaceRecognition.FaceRecognition import buildFacespace
from FaceRecognition.FaceRecognition import selectKComponents, weights
from FaceRecognition.ImageUtils import readImageAsGray, flattenImage

from cv2 import imshow, imread
import cv2

from numpy import fliplr, zeros
from numpy.linalg import eig, norm
import numpy as np

# ------------------ CONSTANTS --------------------------

INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100\dimaio'
INPUT_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'

TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\_baddetections\matt022-2.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\io\rugg001.png'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\salvini\salvini003.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\gentiloni\gent001.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test\_baddetections\matt022-2.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\salvini\salvini002.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\io\rugg001.png'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\salvini\salvini003.jpg'
TEST_IMAGE_PATH = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test\renzi\renzi002.jpg'

MAX_N_EIGENVECTORS = 40 #55
# ------------------ Actual code ------------------------
input_path = INPUT_PATH

images = readFilesRecursively(input_path)

u, A, mean_face = buildFacespace(images, MAX_N_EIGENVECTORS)

imshow("Mean face", mean_face.reshape(100,100))

cv2.waitKey(0)
cv2.destroyAllWindows() 

#raise RuntimeError

eigenval,V = eig(A.T.dot(A))

eigenval = np.sort(eigenval)[::-1] # Ordina gli autovalori in modo discendente

V = fliplr(V)

#Omega = u.T.dot(some_face - mean_face) 

"""
U, S, Vh, mean_face = faceRecognition(images, 10)

eigenvals = np.diagonal(S) ** 2 # Autovalori di A.dot(A.T); con A = U.dot(S).dot(Vh)

eigenvals = np.sort(eigenvals)[::-1] # Ordina gli autovalori in modo discendente
"""

#imshow("Mean face", mean_face.reshape(100,100))
#cv2.waitKey(0)

error_plot = zeros(MAX_N_EIGENVECTORS)


mins = zeros(MAX_N_EIGENVECTORS-3)
mins_differences = zeros(MAX_N_EIGENVECTORS-3)

for n in range(3,MAX_N_EIGENVECTORS):
    u, A, mean_face = buildFacespace(images, n)
    u = u[:,3:]
    
    # Let's take a test image matrix
    # and vectorize it.
    #TODO: test_image = flatten_image('test.jpg');
    test_image = flattenImage(readImageAsGray(TEST_IMAGE_PATH))
    
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
    mins[n-3] = min_pos
    
    print("Number of eigen components: %d --> Error %d --> Image # %d.\n" % (n,min_error,min_pos))
    error_plot[n] = min_error

threshold = 15    
for i in range(threshold-3, MAX_N_EIGENVECTORS-3):
    j = i #- 3
    mins_differences[j] = mins[j] - mins[j-1]
    
changeCount = 0
for i in range(threshold-3,len(mins_differences)):
    if mins_differences[i] != 0:
        changeCount += 1
#cv2.destroyAllWindows() 

x = np.arange(0,MAX_N_EIGENVECTORS,1)


# show mean and 1th through 15th principal eigenvectors
import matplotlib.pyplot as plt

plt.plot(x, error_plot)
plt.title("Error Analysis")
plt.xlabel("No. of eigenvectors")
plt.ylabel("Error")
plt.show()

#fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(6,10))

#ax1.imshow(mean_face.reshape(100,100), extent=[100,100])
#ax1.set_title('Mean Face')

#ax2.imshow(grid, extent=[0,100,0,1], aspect='auto')
#ax2.set_title('Auto-scaled Aspect')

#ax3.imshow(grid, extent=[0,100,0,1], aspect=100)
#ax3.set_title('Manually Set Aspect')

#plt.tight_layout()
#plt.show()

n_eig = 10
cv2.imshow("Eigenface %d" % n_eig, u[:,n_eig].reshape((100,100)))
cv2.waitKey(0)

#figure,subplot(4,4,1)
#imagesc(reshape(mean_matrix, [h,w]))
#colormap gray
#for i = 1:15
#    subplot(4,4,i+1)
#    imagesc(reshape(V(:,i),h,w))
#end

cv2.destroyAllWindows() 


