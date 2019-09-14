# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-



import os
import sys
from numpy import zeros, array

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from datasetpaths import faces95_fe97, faces95_own_classes, faces95_complement_fe97, faces95_all_classes

from plotting import plotConfusionMatrix
from FaceRecognition.DatasetModel import DatasetModel
from FaceRecognition.EigenfaceModel import EigenfaceModel
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    from sklearn.metrics import classification_report
    
    model = DatasetModel([faces95_fe97, faces95_complement_fe97])
    model = model.labeledByFolderOfFiles(faces95_all_classes)
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()
    
    # Let's do a first split between KNOWN people and UNKNOWN people
    pI = model.partitionIndexes
    X0 = X[:pI[1]]
    y0 = y[:pI[1]]
    
    # Let's do a second split between just known people (to mix later)
    X_train, X_test1, y_train, y_test1 = train_test_split(X0, 
                                                        y0, 
                                                        test_size=0.2,  # 0.2
                                                        random_state=0,
                                                        stratify=y0) 
    
    # Unknown people
    X_test2  = X[pI[1]:]
    y_test2  = y[pI[1]:]
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=4,nmax=150,m=m, n=n)
    print(eigenfaceModel)
 
    X_test = np.concatenate((X_test1, X_test2))
    y_test = np.concatenate((y_test1, y_test2))
        
    # People that should be KNOWN
    #epsilons1 = zeros(len(y_test1))
    #for i, X_i in enumerate(X_test1):
    #    y_pred_i, min_err_i = eigenfaceModel._noFitTest(X_i)
    #    epsilons1[i] = min_err_i

    # People that should NOT be KNOWN
    #epsilons2 = zeros(len(y_test2))
    #for i, X_i in enumerate(X_test2):
    #    y_pred_i, min_err_i = eigenfaceModel._noFitTest(X_i)
    #    epsilons2[i] = min_err_i
    
    w_true = zeros(len(y_test))
    w_true[:len(y_test1)] = 1 # We know that the first part of the split is of known people
    
    w_predict = zeros(len(y_test)) # Array to fill with predictions
    epsilons = zeros(len(y_test))
    
    for i, X_i in enumerate(X_test):
        y_pred_i, min_err_i = eigenfaceModel._noFitTest(X_i)
        epsilons[i] = min_err_i
    
    # Method 1: Fixed threshold = 2000 (number deducted from the plots in EDA submodule)
    w_predict[epsilons <= 250] = 1
    
    print(classification_report(w_true, w_predict, target_names=['Known', 'Unknown']))
    
    plotConfusionMatrix(w_true, w_predict, labels = ['Known', 'Unknown'])
    
    # Method 2: Threshold calculated as mean + 2sd from the test1 split taken from the known dataset
    sub_eps = epsilons[:len(y_test1)]
    threshold = np.mean(sub_eps) + 2 * np.std(sub_eps,ddof=1,dtype=np.float64)
    w_predict2 = zeros(len(y_test))
    w_predict2[epsilons <= threshold] = 1
    print(classification_report(w_true, w_predict2, target_names=['Known', 'Unknown']))
    plotConfusionMatrix(w_true, w_predict2, labels = ['Known', 'Unknown'])
    
    # Method 3: Threshold calculated as mean + 3*sd from the test1 split taken from the known dataset
    sub_eps = epsilons[:len(y_test1)]
    threshold = np.mean(sub_eps) + 3 * np.std(sub_eps,ddof=1,dtype=np.float64)
    w_predict3 = zeros(len(y_test))
    w_predict3[epsilons <= threshold] = 1
    print(classification_report(w_true, w_predict3, target_names=['Known', 'Unknown']))
    plotConfusionMatrix(w_true, w_predict3, labels = ['Known', 'Unknown'])
    
    
    
    
    
    #y_predict = zeros(len(y_test))
    #for i, X_i in enumerate(X_test):
    #    y_predict[i] = eigenfaceModel.noFitTest(X_i,1)
    