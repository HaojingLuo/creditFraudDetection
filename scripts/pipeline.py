__author__ = 'Haohan Wang'

import numpy as np
from model.helpingMethods import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso
from model.lrLMM import LowRankLinearMixedModel as trLMM
from sklearn.metrics import precision_recall_curve, auc

def cross_validation(model, X, y, learningRate, CVNum=5):
    predL = []
    for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
        # model.setLearningRate(learningRate)
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        predL.extend(pred.tolist())
    return predL

def cross_validation_sklearn(model, X, y, CVNum=5):
    predL = []
    for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
        predL.extend(pred.tolist())
    return predL

def evaluation_aucPR(pred, y):
    p, r, t = precision_recall_curve(y, pred)
    return auc(r, p)

#scores:
# NB: 0.4484
# tr-NB: 0.4898
# Lasso: 0.05919
# tr-Lasso: 0.05919


def run():
    data = np.load('../data/data.npy')
    X = data[:284805,:-1]
    y = data[:284805,-1]

    model = trLMM()
    pred = cross_validation(model, X, y, learningRate=1e-5)

    score = evaluation_aucPR(pred, y)

    print score

if __name__ == '__main__':
    run()
