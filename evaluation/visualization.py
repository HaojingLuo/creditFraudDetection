__author__ = 'Haohan Wang'

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

import matplotlib

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)

def visualization_prCurve():
    lss = {'nb':'-', 'knn':':', 'svm':'--'}
    colors = {'original':'g', 'lmm':'m', 'chi2':'b'}

    fig = plt.figure(dpi=300, figsize=(20, 8))

    # axs = [0 for i in range(1)]
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.8])

    data = np.load('../data/data.npy')
    y = data[:284805,-1]

    for method in ['nb', 'knn', 'svm']:
        for features in ['original', 'lmm', 'chi2']:
            pred = np.load('../result/pred_'+method+'_'+features + '.npy')
            p, r, t = precision_recall_curve(y, pred)
            if features == 'lmm':
                lbtext = 'trmm'
            else:
                lbtext = features
            ax.plot(r, p, ls=lss[method], color=colors[features], label=method.upper()+'_'+lbtext)
    ax.title.set_text('Precision Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    # ax.set_xlim(-0.01, 0.5)
    # ax.set_ylim(0, 0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(1.1, 1),
          ncol=1, fancybox=True, shadow=True)
    plt.savefig('fig.pdf')


if __name__ == '__main__':
    visualization_prCurve()