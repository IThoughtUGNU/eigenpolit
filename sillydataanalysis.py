# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:05:25 2019

@author: ruggi
"""

import cv2
import os
import numpy as np

input_path = r'C:\dev\python\nm4cs\eigenfaces\dataset\faces96-fe'

xs = []
ys = []

for root, subdirs, files in os.walk(input_path):
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        xs.append(img.shape[1])
        ys.append(img.shape[0])
        
import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=xs, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Size in pixels (only width)')
plt.ylabel('Frequency')
plt.title('Faces95 (x) images size Histogram')
mu = np.mean(xs)
std = np.std(xs)
median = np.median(xs)
plt.text(23, -20, r'$\mu={}, std={}$, median = {}'.format("%.2f" % mu, "%.2f" % std, "%.2f" % median))
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig("hist_x.svg")
plt.show()

# An "interface" to matplotlib.axes.Axes.hist() method
n, bins, patches = plt.hist(x=ys, bins='auto', color='#aa0405',
                            alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Size in pixels (only height)')
plt.ylabel('Frequency')
plt.title('Faces95 (y) images size Histogram')

mu = np.mean(ys)
std = np.std(ys)
median = np.median(ys)
plt.text(23, -25, r'$\mu={}, std={}$, median = {}'.format("%.2f" % mu, "%.2f" % std, "%.2f" % median))
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.savefig("hist_y.svg")
plt.show()
        
xs = np.array(xs)
ys = np.array(ys)