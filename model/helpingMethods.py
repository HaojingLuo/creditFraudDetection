__author__ = 'Haohan Wang'

import scipy.linalg as linalg
import scipy
import numpy as np
from scipy import stats

def matrixMult(A, B):
    try:
        linalg.blas
    except AttributeError:
        return np.dot(A, B)

    if not A.flags['F_CONTIGUOUS']:
        AA = A.T
        transA = True
    else:
        AA = A
        transA = False

    if not B.flags['F_CONTIGUOUS']:
        BB = B.T
        transB = True
    else:
        BB = B
        transB = False

    return linalg.blas.dgemm(alpha=1., a=AA, b=BB, trans_a=transA, trans_b=transB)

def factor(X, rho):
    """
    computes cholesky factorization of the kernel K = 1/rho*XX^T + I
    Input:
    X design matrix: n_s x n_f (we assume n_s << n_f)
    rho: regularizaer
    Output:
    L  lower triangular matrix
    U  upper triangular matrix
    """
    n_s, n_f = X.shape
    K = 1 / rho * scipy.dot(X, X.T) + scipy.eye(n_s)
    U = linalg.cholesky(K)
    return U

def tstat(beta, var, sigma, q, N, log=False):

    """
       Calculates a t-statistic and associated p-value given the estimate of beta and its standard error.
       This is actually an F-test, but when only one hypothesis is being performed, it reduces to a t-test.
    """
    ts = beta / np.sqrt(var * sigma)
    # ts = beta / np.sqrt(sigma)
    # ps = 2.0*(1.0 - stats.t.cdf(np.abs(ts), self.N-q))
    # sf == survival function - this is more accurate -- could also use logsf if the precision is not good enough
    if log:
        ps = 2.0 + (stats.t.logsf(np.abs(ts), N - q))
    else:
        ps = 2.0 * (stats.t.sf(np.abs(ts), N - q))
    if not len(ts) == 1 or not len(ps) == 1:
        raise Exception("Something bad happened :(")
        # return ts, ps
    return ts.sum(), ps.sum()

def nLLeval(ldelta, Uy, S):
    """
    evaluate the negative log likelihood of a random effects model:
    nLL = 1/2(n_s*log(2pi) + logdet(K) + 1/ss * y^T(K + deltaI)^{-1}y,
    where K = USU^T.
    Uy: transformed outcome: n_s x 1
    S:  eigenvectors of K: n_s
    ldelta: log-transformed ratio sigma_gg/sigma_ee
    """
    n_s = Uy.shape[0]
    delta = scipy.exp(ldelta)

    # evaluate log determinant
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    # evaluate the variance
    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    # evalue the negative log likelihood
    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    return nLL

def nLLeval_singleValue(ldelta_singleValue, ldelta, ind, Uy, S):
    n_s = Uy.shape[0]
    ldelta[ind] = ldelta_singleValue
    delta = scipy.exp(ldelta)

    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    return nLL

def nLLeval_delta(delta, Uy, S):
    n_s = Uy.shape[0]
    Sd = S + delta
    ldet = scipy.sum(scipy.log(Sd))

    Sdi = 1.0 / Sd
    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    nLL = 0.5 * (n_s * scipy.log(2.0 * scipy.pi) + ldet + n_s + n_s * scipy.log(ss))

    return nLL

def nLLeval_delta_grad(delta, Uy, S):
    n_s = Uy.shape[0]
    Sd = S + delta
    Sdi = 1.0 / Sd

    ldet_grad = Sdi

    Uy = Uy.flatten()
    ss = 1. / n_s * (Uy * Uy * Sdi).sum()

    nll_grad = 0.5*(ldet_grad - np.exp(-ss)*(Uy*Uy/(Sdi*Sdi)))
    return nll_grad

def binarySearch(model, X, y, select_num, learningRate):
    betaM = np.zeros([X.shape[1]])
    min_lambda_default = 1e-15
    max_lambda_default = 1e15
    patience = 25
    minFactor = 0.9
    maxFactor = 1.1

    iteration = 0
    maxIteration = 25
    min_lambda = min_lambda_default
    max_lambda = max_lambda_default

    minDiffC = np.inf

    stuckCount = 1
    previousC = -1

    while min_lambda < max_lambda and iteration < maxIteration:
        iteration += 1
        lmbd = np.exp((np.log(min_lambda) + np.log(max_lambda)) / 2.0)

        # print "\t\tIter:{}\tlambda:{}".format(iteration, lmbd),
        model.setLambda(lmbd)
        model.setLearningRate(learningRate)  # learning rate must be set again every time we run it.
        model.fit(X, y)
        beta = model.getBeta()

        c = len(np.where(np.abs(beta) > 0)[
                    0])  # we choose regularizers based on the number of non-zeros it reports
        # print "# Chosen:{}".format(c)
        if c < select_num*minFactor:  # Regularizer too strong
            max_lambda = lmbd
            diffC = select_num*minFactor - c
            if diffC < minDiffC:
                betaM = beta
                minDiffC = diffC
        elif c > select_num*maxFactor:  # Regularizer too weak
            min_lambda = lmbd
            diffC = c - select_num*maxFactor
            if diffC < minDiffC:
                betaM = beta
                minDiffC = diffC
        else:
            betaM = beta
            break
        if c == previousC:
            stuckCount += 1
        else:
            previousC = c
            stuckCount = 1
        if stuckCount > patience:
            # print 'Run out of patience'
            break

    return betaM

def KFold(X,y,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain = np.delete(X,testlst,0)
        ytrain = np.delete(y,testlst,0)
        Xtest = X[testlst]
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest

def cross_validation(model, X, y, learningRate, CVNum=5):
    min_mse = np.inf
    min_lam = 0
    for i in range(5):
        lam = 10**(i-2)
        model.setLambda(lam)
        model.setLearningRate(learningRate)
        mse = 0
        for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
            model.fit(Xtrain, ytrain)
            pred = model.predict(Xtest)
            mse += np.linalg.norm(pred - ytest)
        if mse < min_mse:
            min_mse = mse
            min_lam = lam
    model.setLambda(min_lam)
    model.fit(X, y)
    betaM = model.getBeta()
    return betaM