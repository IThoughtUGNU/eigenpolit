# -*- coding: utf-8 -*-

import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

if __name__ == "__main__":
    # Goal
    # - Test the performance of FaceEigenClassifier (test if each test image is a face or not)
    #       --> Use output100 as training set of images; output100-test as test set.
    #
    # - Test the performance of KnownFaceClassifier (test if each test image represents a person in the dataset or not)
    #       --> Use output100lbp as training set; output100lbp-test as test set.
    #
    # Find the best parameter for the 2 classifiers for such recognitions
    #
    # - Test the performance, given a pass of KnownFaceClassifier before,
    #     of the Eigenface model to recognize people BELONGING to the people dataset,
    #     but with images not already present in the training dataset.
    #       --> Mix all the KNOWN people from output100lbp and output100lbp-test (leave out foreigners)
    #           Select 80/20 or 90/10 randomly to build training set and test set.
    pass