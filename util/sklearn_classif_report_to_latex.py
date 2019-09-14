# -*- coding: utf-8 -*-

import pandas as pd

def classificationReportDictToDataFrame(reportDict):
    COLS = [ 'F1-score', 'Precision', 'Recall', 'Support']
    df = pd.DataFrame(reportDict)
    arr = df.values.T
    
    df = pd.DataFrame(data = arr[:,:],index=list(reportDict.keys()), columns = COLS)
    
    #keys = list(reportDict.keys())
    #df.set_index([pd.Index(keys), ''])
    
    #df.loc[:,[COLS]].values = arr
    
    df = df[['Precision', 'Recall', 'F1-score', 'Support']]
    return df

def toLatex(reportDf):
    f = lambda x: "%.f%%" % (x*100.0)
    
    df = reportDf.copy()
    df.loc[:, 'Precision'] = df.loc[:, 'Precision'].map(f)
    df.loc[:, 'Recall'] = df.loc[:, 'Recall'].map(f)
    df.loc[:, 'F1-score'] = df.loc[:, 'F1-score'].map(f)
    df.Support = df.Support.astype(int)
    return df.to_latex()