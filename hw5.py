from numpy import *
import numpy.random
from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import sklearn.preprocessing
import itertools

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


def weak_learner(train_data, train_labels, D):
    d = len(train_data[0])
    f_star = inf
    teta_star = 0
    j_star = 0
    b = 0
    for j in range(d):
        f1 = 0
        f2 = 0

        train_data[j] = numpy.sort(train_data[j])
        all_j_indexes = numpy.array([train_data[0][j] for sample in train_data])
        sorted_samplea = all_j_indexes.argsort()
        for i in range(len(train_data)):
            if train_labels[i] == 1:
                f1 += D[i]
            else:
                f2 += D[i]

        if f1 < f_star:
            f_star = f1
            teta_star = train_data[sorted_samplea[0]][j] - 1
            j_star = j
            b = 1

        if f2 < f_star:
            f_star = f2
            teta_star = train_data[sorted_samplea[0]][j] - 1
            j_star = j
            b = -1

        for i, samp_index in enumerate(sorted_samplea):
            f1 = f1 - train_labels[samp_index] * D[samp_index]
            f2 = f2 + train_labels[samp_index] * D[samp_index]

            if f1 < f_star:
                if i == len(D) - 1:
                    f_star = f1
                    teta_star = 1 + train_data[samp_index][j]
                    j_star = j
                    b = 1
                elif train_data[sorted_samplea[i]][j] != train_data[sorted_samplea[i + 1]][j]:
                    f_star = f1
                    teta_star = 1 / 2 * (train_data[sorted_samplea[i]][j] + train_data[sorted_samplea[i + 1]][j])
                    j_star = j
                    b = 1
            if f2 < f_star:
                if i == len(D) - 1:
                    f_star = f1
                    teta_star = 1 + train_data[samp_index][j]
                    j_star = j
                    b = -1
                elif train_data[sorted_samplea[i]][j] != train_data[sorted_samplea[i + 1]][j]:
                    f_star = f1
                    teta_star = 1 / 2 * (train_data[sorted_samplea[i]][j] + train_data[sorted_samplea[i + 1]][j])
                    j_star = j
                    b = -1
    return j_star, teta_star, b


def adaBoost(train_data, train_labels, T):
    m = len(train_data)
    # initialize
    D = numpy.add(numpy.zeros(m), 1 / m)
    train_errors = []
    test_errors = []
    test_samples_curr_sum = [0] * len(test_data)
    train_samples_curr_sum = [0] * len(train_data)

    for l in range(T + 1):
        j, t, b = weak_learner(train_data, train_labels, D)
        print("got j_star: " + str(j) + " teta_star: " + str(t) + " b: " + str(b))
        error = 0

        for i in range(len(train_data[j])):
            sample = train_data[j][i]
            label = train_labels[j]
            prediction = b if sample <= t else -b
            if prediction != label:
                error += D[i]

        w = 1 / 2 * log(float(1) / float(error) - 1)
        Z = 0
        for k in range(m):
            prediction = b if train_data[k][j] <= t else -b
            Z += D[k] * exp(-w * train_labels[k] * prediction)
        for i in range(m):
            prediction = b if train_data[i][j] <= t else -b
            D[i] = (D[i] * exp(-w * train_labels[i] * prediction)) / float(Z)

        train_error = 0
        print("T: ", l)
        for i in range(len(train_data)):
            h_predication = b if train_data[i][j] <= t else -b
            train_samples_curr_sum[i] += w * h_predication
            final_h_predication = 1 if train_samples_curr_sum[i] >= 0 else -1
            if train_labels[i] != final_h_predication:
                train_error += 1
        train_error = float(train_error) / float(len(train_data))
        print("train_error " + str(train_error))
        train_errors.append(train_error)

        test_error = 0
        for i in range(len(test_data)):
            h_predication = b if test_data[i][j] <= t else -b
            test_samples_curr_sum[i] += w * h_predication
            final_h_predication = 1 if test_samples_curr_sum[i] >= 0 else -1
            if test_labels[i] != final_h_predication:
                test_error += 1
        test_error = float(test_error) / float(len(test_data))
        print("test error " + str(test_error))
        test_errors.append(test_error)

    x_axis = [t for t in range(1, T + 1)]
    plt.plot(x_axis, test_errors, 'bo', x_axis, train_errors, 'ro')
    plt.savefig('1b.png')
    plt.clf()
    # return w, j, t


def calculate_error(train_data, train_labels, D, j, t, b):
    errors = 0
    for i in range(len(train_data)):
        if train_data[i][j] <= t:
            label = b
        else:
            label = -b

        if label != train_labels[i]:
            errors += D[i]

    return errors


# j, t = weak_learner(train_data, train_labels, numpy.zeros(len(train_data)) + (1 / len(train_data)))
# print(j)
adaBoost(train_data, train_labels, 100)
