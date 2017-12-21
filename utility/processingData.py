__author__ = 'Haohan Wang'

import numpy as np

def preprocessedData():
    text = [line.strip() for line in open('../data/raw/creditcard.csv')][1:]
    data = []
    for line in text:
        l = []
        items = line.split(',')
        for i in range(1, len(items)-1):
                l.append(float(items[i]))
        if items[-1] == '"0"':
            l.append(0)
        else:
            l.append(1)

        data.append(l)

    data = np.array(data)
    np.save('../data/data', data)

    ind = np.where(data[:,-1]==1)[0]
    print ind

if __name__ == '__main__':
    preprocessedData()