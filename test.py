from __future__ import division
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

s_train = numpy.array([numpy.array([x, y]) for x, y in itertools.zip_longest(train_data, train_labels)])
s_test = numpy.array([[x, y] for x, y in itertools.zip_longest(test_data, test_labels)])


def weak_learner(S, D):
    F_star = float("inf")

    d = len(S[0][0])
    j_star = -1
    teta_star = None
    b = 0

    for j in range(d):
        all_j_indexes = numpy.array([sample[0][j] for sample in S])
        sorted_samples_by_j = all_j_indexes.argsort()

        F1 = 0  # of the form 1 if x_ij <= teta else -1
        F2 = 0  # of the form -1 if x_ij <= teta else 1
        for sample, D_i in itertools.zip_longest(S, D):
            if sample[1] == 1:
                F1 += D_i
            else:
                F2 += D_i

        if F1 < F_star:
            F_star = F1
            teta_star = S[sorted_samples_by_j[0]][0][j] - 1
            j_star = j
            b = 1

        if F2 < F_star:
            F_star = F2
            teta_star = S[sorted_samples_by_j[0]][0][j] - 1
            j_star = j
            b = -1

        for i, sample_index in enumerate(sorted_samples_by_j):
            F1 = F1 - S[sample_index][1] * D[sample_index]
            F2 = F2 + S[sample_index][1] * D[sample_index]
            if F1 < F_star:
                if i == len(D) - 1:
                    F_star = F1
                    teta_star = 1 + S[sample_index][0][j]
                    j_star = j
                    b = 1

                elif S[sorted_samples_by_j[i]][0][j] != S[sorted_samples_by_j[i + 1]][0][j]:
                    F_star = F1
                    teta_star = 0.5 * (S[sorted_samples_by_j[i]][0][j] + S[sorted_samples_by_j[i + 1]][0][j])
                    j_star = j
                    b = 1

            elif F2 < F_star:
                if i == len(D) - 1:
                    F_star = F2
                    teta_star = 1 + S[sample_index][0][j]
                    j_star = j
                    b = -1

                elif S[sorted_samples_by_j[i]][0][j] != S[sorted_samples_by_j[i + 1]][0][j]:
                    F_star = F2
                    teta_star = 0.5 * (S[sorted_samples_by_j[i]][0][j] + S[sorted_samples_by_j[i + 1]][0][j])
                    j_star = j
                    b = -1

    assert j_star > -1
    assert teta_star
    assert b != 0
    return j_star, teta_star, b


def ada_boost(S, T):
    D = numpy.array([float(1) / float(len(S))] * len(S))

    test_errors = []
    train_errors = []

    test_samples_curr_sum = [0] * len(test_data)
    train_samples_curr_sum = [0] * len(train_data)

    for t in range(1, T + 1):
        # print("t:" + str(t))
        j_star, teta_star, b = weak_learner(S, D)
        print("got j_star: " + str(j_star) + " teta_star: " + str(teta_star) + " b: " + str(b))

        epsilon_t = 0
        for d, sample in itertools.zip_longest(D, S):
            predication = b if sample[0][j_star] <= teta_star else -b
            if predication != sample[1]:
                epsilon_t += d

        w_t = 0.5 * log((float(1) / float(epsilon_t)) - 1)

        denominator = 0
        for d, sample in itertools.zip_longest(D, S):
            predication = b if sample[0][j_star] <= teta_star else -b
            denominator += d * exp(-w_t * sample[1] * predication)

        for i in range(len(D)):
            predication = b if S[i][0][j_star] <= teta_star else -b
            D[i] = float((D[i] * exp(-w_t * S[i][1] * predication))) / float(denominator)

        train_error = 0
        for i in range(len(train_data)):
            h_predication = b if train_data[i][j_star] <= teta_star else -b
            train_samples_curr_sum[i] += w_t * h_predication
            final_h_predication = 1 if train_samples_curr_sum[i] >= 0 else -1
            if train_labels[i] != final_h_predication:
                train_error += 1
        train_error = float(train_error) / float(len(train_data))
        print("train_error" + str(train_error))
        train_errors.append(train_error)

        test_error = 0
        for i in range(len(test_data)):
            h_predication = b if test_data[i][j_star] <= teta_star else -b
            test_samples_curr_sum[i] += w_t * h_predication
            final_h_predication = 1 if test_samples_curr_sum[i] >= 0 else -1
            if test_labels[i] != final_h_predication:
                test_error += 1
        test_error = float(test_error) / float(len(test_data))
        print("test error" + str(test_error))
        test_errors.append(test_error)

    x_axis = [t for t in range(1, T + 1)]
    plt.plot(x_axis, test_errors, 'bo', x_axis, train_errors, 'ro')
    plt.savefig('1b-before2.png')
    plt.clf()


ada_boost(s_train, 100)
