# -*- coding: utf-8 -*-

import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from datasetpaths import *


from FaceRecognition.EigenfaceModel import EigenfaceModel
from FaceRecognition.DatasetModel import DatasetModel
from numpy import zeros, sqrt, argmax
from numpy.linalg import norm

configs = {
        0: [output100lbp, output100lbp_test_strict, output100strict_classes],
        1: [faces94, faces94_complement, faces94_ALL_classes],
        2: [faces94, None, faces94_own_classes]
        }

def plotHist(xs,xlabel,title,color='#0504aa'):
    import numpy as np
    import matplotlib.pyplot as plt
    
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=xs, bins='auto', color=color,
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    mu = np.mean(xs)
    std = np.std(xs)
    median = np.median(xs)
    #plt.text(23, 45, r'$\mu={}, std={}$, median = {}'.format("%.2f" % mu, "%.2f" % std, "%.2f" % median))
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    #plt.ylim(top=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def BuildModel():
    pass
    
def checkKnownPeople(config):
    from sklearn.model_selection import train_test_split

    #model = DatasetModel([output100lbp, output100lbp_test_strict])
    model = DatasetModel(config[0])
    model = model.labeledByFolderOfFiles(config[2])
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2,  # 0.2
                                                        random_state=0,
                                                        stratify=y) 
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin = 4, nmax=150,m=m,n=n)
    
    epsilons = zeros(len(y_test))
    for i, X_i in enumerate(X_test):
        Phi_i = X_i - eigenfaceModel.mean_face
        Ω_i = eigenfaceModel.projectOntoEigenspace(X_i)
        #Phi_f = zeros(len(Phi_i))
        #Phi_f[:len(Ω_i)] = Ω_i
        #epsilons[i] = norm(Phi_i - Phi_f)
        epsilons[i] = sqrt(abs(Phi_i.T.dot(Phi_i) - Ω_i.T.dot(Ω_i))) #sqrt(norm(X_i - Ω_i))
    
    # 4 possibilities (from paper)
    # (1) near face space and near a face class
    # (2) near face space but not near to a known face class
    # (3) distant from face space and near a face class
    # (4) distant from face space and not near a known face class
    
    plotHist(epsilons, "Distance between image and face-space", "Distance Hist for faces that should be known")
    return X_train, y_train, X_test, y_test, eigenfaceModel, model
    
def checkUnknownPeople(config,color="#ab1749"):
    from sklearn.model_selection import train_test_split

    model = DatasetModel([config[0], config[1]])
    model = model.labeledByFolderOfFiles(config[2])
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()

    pI = model.partitionIndexes
    X_train = X[:pI[1]]
    y_train = y[:pI[1]]
    X_test  = X[pI[1]:]
    y_test  = y[pI[1]:]
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin = 4, nmax=150,m=m,n=n)
    
    
    epsilons = zeros(len(y_test))
    for i, X_i in enumerate(X_test):
        Phi_i = X_i - eigenfaceModel.mean_face
        Ω_i = eigenfaceModel.projectOntoEigenspace(X_i)
        
        #Phi_f = zeros(len(Phi_i))
        #Phi_f[:len(Ω_i)] = Ω_i
        #epsilons[i] = norm(Phi_i - Phi_f)
        epsilons[i] = sqrt(abs(Phi_i.T.dot(Phi_i) - Ω_i.T.dot(Ω_i))) #sqrt(norm(X_i - Ω_i))
    
    #print(argmax(epsilons))
    #print(model.files[argmax(epsilons)])
    # 4 possibilities (from paper)
    # (1) near face space and near a face class
    # (2) near face space but not near to a known face class
    # (3) distant from face space and near a face class
    # (4) distant from face space and not near a known face class
    
    plotHist(epsilons, "Distance between image and face-space", "Distance Hist for faces that should NOT be known",color)
    
    return X_train, y_train, X_test, y_test, eigenfaceModel, model
    
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, eigenfaceModel, model = checkKnownPeople(configs[2])
    X_train2, y_train2, X_test2, y_test2, eigenfaceModel2, model2 = checkUnknownPeople(configs[1])