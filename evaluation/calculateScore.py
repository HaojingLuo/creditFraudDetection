__author__ = 'Haohan Wang'

import numpy as np

from sklearn.metrics import precision_recall_curve, auc

def calculateScore():
    data = np.load('../data/data.npy')
    y = data[:284805,-1]

    for method in ['nb', 'knn', 'svm']:
        for features in ['original', 'lmm', 'chi2']:
            pred = np.load('../result/pred_'+method+'_'+features + '.npy')
            p, r, t = precision_recall_curve(y, pred)
            s = auc(r, p)
            if features == 'lmm':
                lbtext = 'trmm'
            else:
                lbtext = features
            print method.upper() + '_' + lbtext, '\t', s

def calculateTime():
    for method in ['nb', 'knn', 'svm']:
        for features in ['original', 'lmm', 'chi2']:
            ss = np.load('../result/seconds_'+method+'_'+features + '.npy')
            if features == 'lmm':
                lbtext = 'trmm'
            else:
                lbtext = features
            print method.upper() + '_' + lbtext, '\t', ss

if __name__ == '__main__':
    calculateScore()