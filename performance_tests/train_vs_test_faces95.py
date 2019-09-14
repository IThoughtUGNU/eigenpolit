# -*- coding: utf-8 -*-


import os
import sys
from numpy import zeros, array

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from datasetpaths import faces95, faces95_own_classes, faces95_fe97

from plotting import plotConfusionMatrix
from FaceRecognition.DatasetModel import DatasetModel
from FaceRecognition.EigenfaceModel import EigenfaceModel
from sklearn.model_selection import train_test_split

doPlot = False # Warning, confusion matrices very big!

if __name__ == "__main__":
    model = DatasetModel(faces95_fe97)
    model = model.labeledByFolderOfFiles(faces95_own_classes)
    m,n = model.getDim()
    X, y = model.exportedAsClassicDataset()
    
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2,  # 0.2
                                                        random_state=0,
                                                        stratify=y) 
    
    #X_test  = X_test[y_test < output100_all_classes.index('io')]
    #y_test  = y_test[y_test < output100_all_classes.index('io')]
    
    eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=4,nmax=150,m=m, n=n)
    print(eigenfaceModel)
    
    y_predict = zeros(len(y_test))
    for i, X_i in enumerate(X_test):
        y_predict[i] = eigenfaceModel.noFitTest(X_i,1)
    
    from sklearn.metrics import classification_report
    
    print(classification_report(y_test, y_predict, target_names=faces95_own_classes))
    if doPlot:
        plotConfusionMatrix(y_test, y_predict, labels = faces95_own_classes)
    
    # ---------------------- OTHER ALGORITHMS -----------------------------
    from sklearn.neural_network import MLPClassifier         
    X_train_pca = eigenfaceModel.exportTrainPca()
    
    X_test_pca = []
    
    for X_i in X_test:
        X_test_pca.append(eigenfaceModel.projectOntoEigenspace(X_i))
    
    X_test_pca = array(X_test_pca)
    # train a neural network
    print("Fitting the classifier to the training set")
    clf = MLPClassifier(hidden_layer_sizes=(1024,1024,), batch_size='auto', verbose=True, early_stopping=True).fit(X_train_pca, y_train)
    print(clf)
    y_pred = clf.predict(X_test_pca)
    print("#--------------------------------------------------------------")
    print("#-                      NEURAL NETWORKS                       -")
    print("#--------------------------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=faces95_own_classes))
    if doPlot:
        plotConfusionMatrix(y_test, y_pred, labels = faces95_own_classes)