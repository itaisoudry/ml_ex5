from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing

mnist = fetch_mldata('MNIST original', data_home='./')
data = mnist['data']
labels = mnist['target']

neg, pos = 0, 8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos) * 2 - 1

# validation_data_unscaled = data[train_idx[6000:], :].astype(float)
# validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000 + test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000 + test_idx[:test_data_size]] == pos) * 2 - 1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
# validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)


def erm_decision_stumps(train_data, train_labels, D):
    d = len(train_data[0])
    f_star = inf
    teta_star = 0
    j_star = 0

    for j in range(d):
        f = 0
        for i in range(len(train_data)):
            if train_labels[i] == 1:
                f += D[i]
        if f < f_star:
            f_star = f
            teta_star = train_data[1][j] - 1
            j_star = j
        for i in range(d):
            f = f - train_labels[i] * D[i]
            if f < f_star and train_data[i][j] != train_data[i + 1][j]:
                f_star = f
                teta_star = 1 / 2 * (train_data[i][j] + train_data[i + 1][j])
                j_star = j

    return j_star, teta_star


j, t = erm_decision_stumps(train_data, train_labels, numpy.zeros(len(train_data)) + (1 / len(train_data)))
print(j)
print(t)
