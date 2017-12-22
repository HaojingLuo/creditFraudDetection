__author__ = 'Haohan Wang'

import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import time

from model.helpingMethods import KFold
from model.lrLMM import LowRankLinearMixedModel as trLMM
from sklearn.metrics import precision_recall_curve, auc

def cross_validation(model, X, y, CVNum=5):
    predL = []
    for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
        # model.setLearningRate(learningRate)
        model.fit(Xtrain, ytrain)
        pred = model.predict_proba(Xtest)[:,1]
        predL.extend(pred.tolist())
    return predL

def evaluation_aucPR(pred, y):
    p, r, t = precision_recall_curve(y, pred)
    return auc(r, p)

def run():
    data = np.load('../data/data.npy')
    X = data[:284805,:-1]
    y = data[:284805,-1]

    for method in ['nb', 'knn', 'svm']:
        for features in ['original', 'lmm', 'chi2']:
            print method, '\t', features, '\t',
            startTime = time.time()
            if method == 'nb':
                model = GaussianNB()
            elif method == 'knn':
                model = KNC()
            else:
                model = SVC(probability=True, kernel='linear')
            if features == 'original':
                pred = cross_validation(model, X, y)
            elif features == 'lmm':
                clf = trLMM(helperModel=model)
                pred = cross_validation(clf, X, y)
            else:
                Xtmp = np.abs(X)
                Xtmp = SelectKBest(chi2, k=10).fit_transform(Xtmp, y)
                pred = cross_validation(model, Xtmp, y)
            np.save('../result/pred_'+method+'_'+features, pred)

            seconds = time.time() - startTime

            np.save('../result/seconds_'+method+'_'+features, seconds)

            score = evaluation_aucPR(pred, y)

            print score

if __name__ == '__main__':
    run()
