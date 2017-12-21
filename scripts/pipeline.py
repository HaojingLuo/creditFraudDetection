__author__ = 'Haohan Wang'

import numpy as np
from model.helpingMethods import KFold

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier as KNC


from model.lrLMM import LowRankLinearMixedModel as trLMM
from sklearn.metrics import precision_recall_curve, auc

def cross_validation(model, X, y, learningRate, CVNum=5):
    predL = []
    for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
        # model.setLearningRate(learningRate)
        model.fit(Xtrain, ytrain)
        pred = model.predict_proba(Xtest)[:,1]
        predL.extend(pred.tolist())
    return predL

def evaluation_aucPR(pred, y):
    p, r, t = precision_recall_curve(y, pred)
    print p
    return auc(r, p)

#scores:
# NB: 0.4484
# tr-NB: 0.4898
# Lasso: 0.05919
# tr-Lasso: 0.05919
# knc: 0.7017
# tr-knc: 0.7969


def run():
    data = np.load('../data/data.npy')
    X = data[:284805,:-1]
    y = data[:284805,-1]

    model = trLMM(helperModel=KNC())
    pred = cross_validation(model, X, y, learningRate=1e-5)

    score = evaluation_aucPR(pred, y)

    print score

if __name__ == '__main__':
    run()
