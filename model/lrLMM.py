__author__ = 'Haohan Wang'

import scipy.optimize as opt
import time

import sys

sys.path.append('../')

from helpingMethods import *

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Lasso


class LowRankLinearMixedModel:
    def __init__(self, lowRankFlag=True, numintervals=100, ldeltamin=-5, ldeltamax=5, discoverNum=50, mode='lmm',
                 learningRate=1e-6, realDataFlag=False):
        self.lowRankFlag = lowRankFlag
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.discoverNum = discoverNum
        self.mode = mode
        self.learningRate = learningRate
        self.realDataFlag = realDataFlag

    def setFlag(self, flag):
        self.lowRankFlag = flag

    def rescale(self, a):
        return a / np.max(np.abs(a))

    def selectValues(self, Kva):
        r = np.zeros_like(Kva)
        n = r.shape[0]
        tmp = self.rescale(Kva)
        ind = 0
        for i in range(n/2, n - 1):
            if tmp[i + 1] - tmp[i] > 1.0 / n:
                ind = i + 1
                break
        r[ind:] = Kva[ind:]
        r[n - 1] = Kva[n - 1]
        return r

    def fit(self, X, y):
        self.X = X
        self.y = y
        U, S, V = linalg.svd(X, full_matrices=False)
        S = np.power(S, 2)
        self.train(X=X, Kva=S, Kve=U, y=y)

    def train(self, X, Kva, Kve, y):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        # assert K.shape[0] == K.shape[1], 'dimensions do not match'
        # assert K.shape[0] == X.shape[0], 'dimensions do not match'
        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        X0 = np.ones(len(y)).reshape(len(y), 1)

        S, U, ldelta0 = self.train_nullmodel(y, S=Kva, U=Kve, numintervals=self.numintervals,
                                             ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

        delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_f, 1))
        SUX0 = scipy.dot(U.T, X0)

        w = self.generateWeights(SUX, SUy)

        self.weights = w

    def getWeights(self):
        return self.weights

    def train_nullmodel(self, y, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm',
                        p=1):
        ldeltamin += scale
        ldeltamax += scale

        y = y - np.mean(y)

        # if S is None or U is None:
        #     S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)

        # grid search
        if not self.lowRankFlag:
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

            nllmin = nllgrid.min()
            ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

            for i in scipy.arange(numintervals - 1) + 1:
                if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                    ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                                  (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                                  full_output=True)
                    if nllopt < nllmin:
                        nllmin = nllopt
                        ldeltaopt_glob = ldeltaopt

        else:
            S = self.selectValues(S)
            nllgrid = scipy.ones(numintervals + 1) * scipy.inf
            ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
            for i in scipy.arange(numintervals + 1):
                nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

            nllmin = nllgrid.min()
            ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

            for i in scipy.arange(numintervals - 1) + 1:
                if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                    ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                                  (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                                  full_output=True)
                    if nllopt < nllmin:
                        nllmin = nllopt
                        ldeltaopt_glob = ldeltaopt

        return S, U, ldeltaopt_glob

    def generateWeights(self, X, y):
        [n, p] = X.shape
        btmp = np.zeros(p)
        for i in range(p):
            Xtmp = X[:,i]
            btmp[i] = np.dot((1/np.dot(Xtmp.T, Xtmp))*Xtmp.T, y)
        btmp = np.abs(btmp)
        mtmp = np.mean(btmp)
        btmp[btmp<mtmp] = 0
        return btmp

    def predict(self, X):
        print self.weights
        idx = np.where(self.weights!=0)[0]
        Xtrain = self.X[:, idx]
        clf = Lasso()
        clf.fit(Xtrain, self.y)
        Xtmp = X[:, idx]
        pred = clf.predict(Xtmp)
        return pred