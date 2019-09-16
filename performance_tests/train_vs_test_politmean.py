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
    import cv2
    
    if 1:
#   IMPLEMENTATION
# - Test the performance, given a pass of KnownFaceClassifier before,
#     of the Eigenface model to recognize people BELONGING to the people dataset,
#     but with images not already present in the training dataset.
#       --> Mix all the KNOWN people from output100lbp and output100lbp-test (leave out foreigners)
#           Select 80/20 or 90/10 randomly to build training set and test set.
    
        from FaceRecognition.EigenfaceModel import EigenfaceModel
        from sklearn.model_selection import train_test_split

        model = DatasetModel([output100, output100_test_strict],preprocess_func = lambda img : img)
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
        
        # ---------------------- ANGLE ----------------------------------------
        
        
        y_predict = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_predict[i] = eigenfaceModel.angleTest(X_i,1)
    
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

            
    
    
    
    
    
    