# -*- coding: utf-8 -*-

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
    
if __name__ == "__main__":
    output100      = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100'
    output100_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-test-strict'
    output100_test_junk = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100-junk'
    
    output100lbp      = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp'
    output100lbp_test_strict = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test-strict'
    output100lbp_test_complement = r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test-complement'
    
    output100strict_classes = ['berlusconi', 'bonino', 'calenda', 'conte', 'dimaio',
                         'gentiloni', 'mattarella', 'meloni', 'renzi', 'salvini',
                         'zingaretti']
    
    from pathlib import Path
    import os
    dirs_at_dir = [Path(f.path).stem for f in os.scandir(r'C:\dev\python\nm4cs\eigenfaces\dataset\output100lbp-test') if f.is_dir() ]
    output100_all_classes = output100strict_classes.copy()
    output100_all_classes.extend(dirs_at_dir)
    output100lbp_classes = list(set(output100_all_classes))
    
    from sklearn.model_selection import train_test_split
    from numpy import zeros

    from FaceRecognition.DatasetModel import DatasetModel
    from FaceRecognition.EigenfaceModel import EigenfaceModel
    # 1. Use only known faces from datasets and plot min errors' histogram
    # 2. Use only unknown faces from test dataset and plot min errors' histogram
    # 3. Use only non-facees from test dataset and plot min error's histogram
    # ---> at 150th eigenvector
    
    #1.
    if 0:
        model = DatasetModel([output100lbp, output100lbp_test_strict])
        model = model.labeledByFolderOfFiles(output100strict_classes)
        X, y = model.exportedAsClassicDataset()
    
        #X_train, X_test, y_train, y_test = train_test_split(X, 
        #                                                    y, 
        #                                                    test_size=0.1,  # 0.2
        #                                                    random_state=0,
        #                                                    stratify=y) 
            
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
    
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=50)
        
        err = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_i, err[i] = eigenfaceModel._noFitTest(X_i)
        
        plotHist(err, 'Error', 'Errors Histogram for faces that should be KNOWN from the model')
        print(err)
        print(max(err))
    
    if 0:
        ###########################################################################
        #2.
        model = DatasetModel([output100lbp, output100lbp_test_complement])
        model = model.labeledByFolderOfFiles(output100lbp_classes)
        X, y = model.exportedAsClassicDataset()
    
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
        
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=50)
        
        err2 = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_i, err2[i] = eigenfaceModel._noFitTest(X_i)
        
        plotHist(err2, 'Error', 'Errors Histogram for faces that should NOT be KNOWN from the model',"#ab1749")
        print(err2)
        print(min(err2))
        
    
    if 1:
        ###########################################################################
        #3. PART 1
        model = DatasetModel([output100, output100_test_strict])
        model = model.labeledByFolderOfFiles(output100strict_classes)
        X, y = model.exportedAsClassicDataset()
    
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
        
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=50)
        
        err2 = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_i, err2[i] = eigenfaceModel._noFitTest(X_i)
        
        plotHist(err2, 'Error', 'Errors Histogram for faces that should NOT be KNOWN from the model',"#ab1749")
        print(err2)
        print(min(err2))
        
        ###########################################################################
        #3.
        model = DatasetModel([output100, output100_test_junk])
        model = model.labeledByFolderOfFiles(output100strict_classes + ['_baddetections'])
        X, y = model.exportedAsClassicDataset()
    
        pI = model.partitionIndexes
        X_train = X[:pI[1]]
        y_train = y[:pI[1]]
        X_test  = X[pI[1]:]
        y_test  = y[pI[1]:]
        
        eigenfaceModel = EigenfaceModel(X_train, y_train,nmin=2,nmax=50)
        
        err2 = zeros(len(y_test))
        for i, X_i in enumerate(X_test):
            y_i, err2[i] = eigenfaceModel._noFitTest(X_i)
        
        plotHist(err2, 'Error', 'Errors Histogram for faces that should NOT be KNOWN from the model',"#ab1749")
        print(err2)
        print(min(err2))
        
    pass