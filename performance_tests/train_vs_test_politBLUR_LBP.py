# -*- coding: utf-8 -*-

import os
import sys
from plotting import plotConfusionMatrix

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

 
if __name__ == "__main__":
    from FaceRecognition.DatasetModel import DatasetModel
    from FaceRecognition.FaceEigenClassifier import FaceEigenClassifier
    from FaceRecognition.KnownFaceClassifier import KnownFaceClassifier
    from numpy import zeros, ones, array
    import cv2
    from datasetpaths import polit100blur_lbp, polit100blur_lbp_complement, polit100_all_classes, polit100_own_classes, polit100,\
                                polit100_complement, polit100med_lbp, polit100med_lbp_complement
    
    if 1:
#   IMPLEMENTATION
# - Test the performance, given a pass of KnownFaceClassifier before,
#     of the Eigenface model to recognize people BELONGING to the people dataset,
#     but with images not already present in the training dataset.
#       --> Mix all the KNOWN people from output100lbp and output100lbp-test (leave out foreigners)
#           Select 80/20 or 90/10 randomly to build training set and test set.
        labels = polit100_all_classes
        
        from FaceRecognition.EigenfaceModel import EigenfaceModel
        from sklearn.model_selection import train_test_split

        origModel = DatasetModel([polit100, polit100_complement],preprocess_func = lambda img : img )
        origModel = origModel.labeledByFolderOfFiles(labels)
        Xo, yo = origModel.exportedAsClassicDataset()
        
        model = DatasetModel([polit100med_lbp, polit100med_lbp_complement],preprocess_func = lambda img : cv2.medianBlur(img,5))
        model = model.labeledByFolderOfFiles(labels)
        X, y = model.exportedAsClassicDataset()
        
        indices = range(len(y))
        X_train, X_test, y_train, y_test, i_train, i_test = train_test_split(X, 
                                                                y, 
                                                                indices,
                                                                test_size=0.1,  # 0.2
                                                                random_state=0,
                                                                stratify=y) 

        #X_test  = X_test[y_test < output100_all_classes.index('io')]
        #y_test  = y_test[y_test < output100_all_classes.index('io')]
        
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=150) # old config 2 | 150 (documentation)
        print(eigenfaceModel)
        
        y_predict2 = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_predict2[i] = eigenfaceModel.noFitTest(X_i,1)
    
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        
        print(classification_report(y_test, y_predict2, target_names=labels))

        plotConfusionMatrix(y_test, y_predict2, labels = labels)
        
        # ---------------------- ANGLE ----------------------------------------
        
        
        y_predict = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_predict[i] = eigenfaceModel.angleTest(X_i,1)
    
        from sklearn.metrics import classification_report
        import matplotlib.pyplot as plt
        
        print(classification_report(y_test, y_predict, target_names=labels))

        plotConfusionMatrix(y_test, y_predict, labels = labels)
        
        # ---------------------- OTHER ALGORITHMS -----------------------------
        from sklearn.neural_network import MLPClassifier         
        X_train_pca = eigenfaceModel.exportTrainPca()
        
        X_test_pca = []
        
        for X_i in X_test:
            X_test_pca.append(eigenfaceModel.projectOntoFacespace(X_i))
        
        X_test_pca = array(X_test_pca)
        # train a neural network
        print("Fitting the classifier to the training set")
        clf = MLPClassifier(hidden_layer_sizes=(1024,1024,), batch_size='auto', verbose=True, early_stopping=True).fit(X_train_pca, y_train)
        print(clf)
        y_pred = clf.predict(X_test_pca)
        print("#--------------------------------------------------------------")
        print("#-                      NEURAL NETWORKS                       -")
        print("#--------------------------------------------------------------")
        print(classification_report(y_test, y_pred, target_names=labels))
        plotConfusionMatrix(y_test, y_pred, labels = labels)

        import matplotlib.gridspec as gridspec
        # Visualization
        def plot_gallery(images, titles, h, w, rows=3, cols=4):
            plt.figure()
            for i in range(rows * cols):
                ax = plt.subplot(rows, cols, i + 1)
                plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
                plt.title(titles[i])
                plt.xticks(())
                plt.yticks(())
                ax.update_layout(height=600, width=800, title_text="Subplots")
                plt.savefig('preds.png', bbox_inches='tight')
         
        from mpl_toolkits.axes_grid1 import ImageGrid
        def better_plot_gallery(images, titles, h, w, rows=3, cols=4):
            fig = plt.figure(1, (12., 8.))
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(3, 4),
                             axes_pad=0.55,
                             )
                        
            for i, axes in enumerate(grid):
                axes.set_title(titles[i], fontdict=None, loc='center', color = "k")
                axes.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
            plt.show()

        def titles(y_pred, y_test, target_names):
            y_pred = y_pred.astype('int')
            y_test = y_test.astype('int')
            for i in range(y_pred.shape[0]):
                pred_name = target_names[y_pred[i]].split(' ')[-1]
                true_name = target_names[y_test[i]].split(' ')[-1]
                yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)
         
        prediction_titles = list(titles(y_pred, y_test, labels))
        better_plot_gallery(X_test, prediction_titles, h=100, w=100)
        better_plot_gallery(Xo[i_test], prediction_titles, h=100, w=100)
            
            
            
    
    
    