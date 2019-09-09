# -*- coding: utf-8 -*-

def plotConfusionMatrix(y_test,y_predict, labels):
    import matplotlib.pyplot as plt
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