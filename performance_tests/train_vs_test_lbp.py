# -*- coding: utf-8 -*-

import os
import sys
from plotting import plotConfusionMatrix

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

"""
def plotConfusionMatrix(y_test,y_predict, labels):
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_predict, y_test)
    
    import numpy as np
    fig = plt.figure(figsize=(len(labels),len(labels)))
    #width = np.shape(conf_mat)[1]
    #height = np.shape(conf_mat)[0]
    
    res = plt.imshow(np.array(conf_mat), cmap=plt.cm.summer, interpolation='nearest')
    for i, row in enumerate(conf_mat):
        for j, c in enumerate(row):
            if c>0:
                plt.text(j-.2, i+.1, c, fontsize=16)
                
    fig.colorbar(res)
    plt.title('Confusion Matrix')
    _ = plt.xticks(range(len(labels)), labels, rotation=90)
    _ = plt.yticks(range(len(labels)), labels)
    plt.show()
"""     
if __name__ == "__main__":
    from pathlib import Path
    
    output100         = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
    output100_test    = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test'
    output100lbp      = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
    output100lbp_test = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test'
    
    output100_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test-strict'
    output100lbp_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test-strict'
    
    output200         = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200'
    output200_test    = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200-test'
    output200lbp      = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200lbp'
    output200lbp_test = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200lbp-test'
    
    output200_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200-test-strict'
    output200lbp_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output200lbp-test-strict'
    
    output100strict_classes = ['berlusconi', 'calenda', 'conte', 'dimaio',
                         'gentiloni', 'mattarella', 'meloni', 'renzi', 'salvini',
                         'zingaretti']
    
    output100_classes = ['berlusconi', 'calenda', 'conte', 'dimaio',
                         'gentiloni', 'mattarella', 'meloni', 'renzi', 'salvini',
                         'zingaretti','io','_baddetections']
    
    dirs_at_dir = [Path(f.path).stem for f in os.scandir(r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test') if f.is_dir() ]
    output100_all_classes = output100_classes.copy()
    output100_all_classes.extend(dirs_at_dir)
    output100lbp_classes = list(set(output100_all_classes))
    
    from FaceRecognition.DatasetModel import DatasetModel
    from FaceRecognition.FaceEigenClassifier import FaceEigenClassifier
    from FaceRecognition.KnownFaceClassifier import KnownFaceClassifier
    from numpy import zeros, ones, array
    
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
    
    # ----------------------------Actual Running-------------------------------
    
    
    # IMPLEMENTATION
    # - Test the performance of FaceEigenClassifier (test if each test image is a face or not)
    #       --> Use output100 as training set of images; output100-test as test set.
    
    if 0:
        model = DatasetModel([output100, output100_test])
        model = model.labeledByFolderOfFiles(output100_all_classes)
        
        X, y = model.exportedAsClassicDataset()
        
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
        
        X_notfaces = X_test[y_test == output100_classes.index('_baddetections')]
        y_notfaces = y_test[y_test == output100_classes.index('_baddetections')]
        
        w_true    = zeros(len(y_test))                               # Imposta di default tutto a 0
        w_true[y_test != output100_classes.index('_baddetections')] = 1  # Imposta a 1 le labels degli esempi che sono facce
        
        w_predict = zeros(len(y_test))
        
        faceEigenClassifier = KnownFaceClassifier(X_train,threshold=6,strategy='none',notConvergenceTol=1)
        faceEigenClassifier.fit(3, 50,log=True)
        
        for i, X_i in enumerate(X_test):
            w_predict[i] = int(faceEigenClassifier.test(X_i))
        
        
        
        from sklearn.metrics import classification_report
        print(classification_report(w_true, w_predict, target_names=['Not a face', 'Face']))
    
    if 0:
#   IMPLEMENTATION
# - Test the performance of KnownFaceClassifier (test if each test image represents a person in the dataset or not)
#       --> Use output100lbp as training set; output100lbp-test as test set.

        model = DatasetModel([output100lbp, output100lbp_test])
        model = model.labeledByFolderOfFiles(output100_all_classes)
        X, y = model.exportedAsClassicDataset()
        
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
        X_test  = X_test[y_test != output100_all_classes.index('_baddetections')]
        y_test  = y_test[y_test != output100_all_classes.index('_baddetections')]
        
        w_true    = zeros(len(y_test))        
        w_true[y_test < output100_classes.index('io')] = 1 
        
        w_predict = zeros(len(y_test))
        
        knownFaceClassifier = KnownFaceClassifier.MakeFromLbpModel(X_train,threshold=20)
        knownFaceClassifier.notConvergenceTol = 6
        knownFaceClassifier.fit(2, 40,log=True) # nmax 35 to best unknowns; nmax 45 to best knowns
        
        for i, X_i in enumerate(X_test):
            w_predict[i] = int(knownFaceClassifier.test(X_i,already_processed=True))
        
        
        from sklearn.metrics import classification_report
        print(classification_report(w_true, w_predict, target_names=['Unknown', 'Known']))
    
    configs = { 0: [output100lbp, output100lbp_test_strict, output100strict_classes], 
                1: 0 }    
    if 1:
#   IMPLEMENTATION
# - Test the performance, given a pass of KnownFaceClassifier before,
#     of the Eigenface model to recognize people BELONGING to the people dataset,
#     but with images not already present in the training dataset.
#       --> Mix all the KNOWN people from output100lbp and output100lbp-test (leave out foreigners)
#           Select 80/20 or 90/10 randomly to build training set and test set.
    
        from FaceRecognition.EigenfaceModel import EigenfaceModel
        from sklearn.model_selection import train_test_split

        model = DatasetModel([output100lbp, output100lbp_test_strict])
        model = model.labeledByFolderOfFiles(output100strict_classes)
        X, y = model.exportedAsClassicDataset()

        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y, 
                                                            test_size=0.1,  # 0.2
                                                            random_state=0,
                                                            stratify=y) 

        #X_test  = X_test[y_test < output100_all_classes.index('io')]
        #y_test  = y_test[y_test < output100_all_classes.index('io')]
        
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=150)
        print(eigenfaceModel)
        
        y_predict = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_predict[i] = eigenfaceModel.noFitTest(X_i,1)
    
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        
        print(classification_report(y_test, y_predict, target_names=output100strict_classes))

        plotConfusionMatrix(y_test, y_predict, labels = output100strict_classes)
        
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
        print(classification_report(y_test, y_pred, target_names=output100strict_classes))
        plotConfusionMatrix(y_test, y_pred, labels = output100strict_classes)
        
        if 1:
    # #############################################################################

            
    # #############################################################################
            # Train a SVM classification model
            from time import time
            from sklearn.model_selection import GridSearchCV
            from sklearn.svm import SVC, LinearSVC, NuSVC
            
            print("#--------------------------------------------------------------")
            print("#-               SUPPORT VECTOR MACHINES                      -")
            print("#--------------------------------------------------------------")
            print("Fitting the classifier to the training set")
            t0 = time()
            #param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            #              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
            #clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
            #                   param_grid, cv=5, iid=False)
            #clf = clf.fit(X_train_pca, y_train)
            #print("done in %0.3fs" % (time() - t0))
            #print("Best estimator found by grid search:")
            #print(clf.best_estimator_)
            
            # decent
            clf = LinearSVC(max_iter=20000)
            clf = clf.fit(X_train_pca, y_train)
            
            # #############################################################################
            # Quantitative evaluation of the model quality on the test set
            print("----------------------- SVM TESTING -------------------------")
            print("Predicting people's names on the test set")
            t0 = time()
            y_pred = clf.predict(X_test_pca)
            print("done in %0.3fs" % (time() - t0))
            
            print(classification_report(y_test, y_pred, target_names=output100strict_classes))
            plotConfusionMatrix(y_test, y_pred, output100strict_classes)
            #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
        
                    
            
    
    
    
    
    
    