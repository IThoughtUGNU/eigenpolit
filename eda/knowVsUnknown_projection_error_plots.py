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
        2: [faces94, None, faces94_own_classes],
        3: [output100lbp, output100_test_complement, output100lbp_classes],
        4: [faces95, faces95_complement, faces95_all_classes],
        'faces95-fe97': [faces95_fe97, faces95_complement_fe97, faces95_all_classes],
        'faces96': [faces96, faces96_complement, faces96_all_classes],
        #'faces96-fe': [faces96_fe, faces96_complement, faces96_all_classes],
        'faces96-fe75': [faces96_fe75, faces96_complement_fe75, faces96_all_classes],
        'faces96-fe100': [faces96_fe100, faces96_complement_fe100, faces96_all_classes],
        'faces96-fe100lbp': [faces96_fe100lbp, faces96_complement_fe100lbp, faces96_all_classes],
        'polit100': [polit100, polit100_complement, polit100_all_classes],
        'polit100lbp' : [polit100lbp, polit100lbp_complement, polit100_all_classes]
        }

blue_color = '#0504aa'
pink_color = "#ab1749"

def plotHist(xs,xlabel,title,color=blue_color):
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
    
def checkKnownPeople(config,color=blue_color):
    from sklearn.model_selection import train_test_split
    import numpy as np
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
    print(max(epsilons))
    sd = np.std(epsilons, ddof=1, dtype=np.float64)
    mean = np.mean(epsilons)
    print("mean:", mean)
    print("sd:", sd)
    print("suggested cut threshold (mean+2sd):", mean + 2*sd)
    return X_train, y_train, X_test, y_test, eigenfaceModel, model
    
def checkUnknownPeople(config,color=pink_color):

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
    
    print(max(epsilons))
    return X_train, y_train, X_test, y_test, eigenfaceModel, model
    
def checkAllPeople(config,color="#ab1749"):
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    model = DatasetModel([config[0], config[1]])
    model = model.labeledByFolderOfFiles(config[2])
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()

    pI = model.partitionIndexes
    X0 = X[:pI[1]]
    y0 = y[:pI[1]]
    
    X_train, X_test1, y_train, y_test1 = train_test_split(X0, 
                                                        y0, 
                                                        test_size=0.2,  # 0.2
                                                        random_state=0,
                                                        stratify=y0) 
    
    X_test2  = X[pI[1]:]
    y_test2  = y[pI[1]:]
    
    print(X_test1.shape, X_test2.shape)
    X_test = np.concatenate((X_test1, X_test2))
    y_test = np.concatenate((y_test1, y_test2))
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin = 4, nmax=150,m=m,n=n)
    
    epsilons = zeros(len(y_test))
    for i, X_i in enumerate(X_test):
        Phi_i = X_i - eigenfaceModel.mean_face
        Ω_i = eigenfaceModel.projectOntoEigenspace(X_i)
        epsilons[i] = sqrt(abs(Phi_i.T.dot(Phi_i) - Ω_i.T.dot(Ω_i))) #sqrt(norm(X_i - Ω_i))
    
    # 4 possibilities (from paper)
    # (1) near face space and near a face class
    # (2) near face space but not near to a known face class
    # (3) distant from face space and near a face class
    # (4) distant from face space and not near a known face class
    
    plotHist(epsilons, "Distance between image and face-space", "Distance Hist for faces that should be known", color)
    print(max(epsilons))
    sd = np.std(epsilons, ddof=1, dtype=np.float64)
    mean = np.mean(epsilons)
    print("mean:", mean)
    print("sd:", sd)
    print("suggested cut threshold (mean+2sd):", mean + 2*sd)
    return X_train, y_train, X_test, y_test, eigenfaceModel, model
    
def showMinDistancesBetweenInVsOutTrainingDataset(config):
    from sklearn.model_selection import train_test_split
    import numpy as np
    
    model = DatasetModel([config[0], config[1]])
    model = model.labeledByFolderOfFiles(config[2])
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()

    pI = model.partitionIndexes
    X0 = X[:pI[1]]
    y0 = y[:pI[1]]
    
    X_train, X_test1, y_train, y_test1 = train_test_split(X0, 
                                                        y0, 
                                                        test_size=0.2,  # 0.2
                                                        random_state=0,
                                                        stratify=y0) 
    
    X_test2  = X[pI[1]:]
    y_test2  = y[pI[1]:]
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin = 4, nmax=150,m=m,n=n) # 4 | 150
    
    # People that should be KNOWN
    epsilons1 = zeros(len(y_test1))
    for i, X_i in enumerate(X_test1):
        y_pred_i, min_err_i = eigenfaceModel._noFitTest(X_i)
        epsilons1[i] = min_err_i

    # People that should NOT be KNOWN
    epsilons2 = zeros(len(y_test2))
    for i, X_i in enumerate(X_test2):
        y_pred_i, min_err_i = eigenfaceModel._noFitTest(X_i)
        epsilons2[i] = min_err_i
        
    plotHist(epsilons1, "Minimum error vs other samples", "(min) Error Hist for faces that should be known", 
             blue_color)
    
    sd, mean = np.std(epsilons1, ddof=1, dtype=np.float64), np.mean(epsilons1)
    print("mean:", mean," | sd:", sd, " |max:", max(epsilons1))
    print("suggested cut threshold (mean+2sd):", mean + 2*sd)
    
    plotHist(epsilons2, "Minimum error vs other samples", "(min) Error Hist for faces that should NOT be known", 
             pink_color)
    
    sd, mean = np.std(epsilons2, ddof=1, dtype=np.float64), np.mean(epsilons2)
    print("mean:", mean," | sd:", sd, " |max:", max(epsilons2))
    print("suggested cut threshold (mean+2sd):", mean + 2*sd)
    
    # return model, eigenfaceModel, epsilons1, epsilons2
    

if __name__ == "__main__":
    #checkKnownPeople(configs[2])
    #checkUnknownPeople(configs[1])
    #checkAllPeople(configs[1],color="#b0d3bf")
    
    
    # Minimum distances in Faces94, of KNOWN faces vs UNKNOWN faces. (Good for telling who is unknown)
    #showMinDistancesBetweenInVsOutTrainingDataset(configs[1])
    
    # Minimum distances in my dataset, of KNOWN faces vs UNKNOWN faces. (Good for telling who is unknown)
    #showMinDistancesBetweenInVsOutTrainingDataset(configs[0])
    #showMinDistancesBetweenInVsOutTrainingDataset(configs[3])
    
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    