from __future__ import division
from numpy import *
import numpy as np
import numpy.random
from sklearn.datasets import fetch_mldata
import sklearn.preprocessing
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

mnist = fetch_mldata('MNIST original', data_home='./')
data = mnist['data']
labels = mnist['target']

neg, pos = 0,8
train_idx = numpy.random.RandomState(0).permutation(where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
test_idx = numpy.random.RandomState(0).permutation(where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

train_data_size = 2000
train_data_unscaled = data[train_idx[:train_data_size], :].astype(float)
train_labels = (labels[train_idx[:train_data_size]] == pos)*2-1

#validation_data_unscaled = data[train_idx[6000:], :].astype(float)
#validation_labels = (labels[train_idx[6000:]] == pos)*2-1

test_data_size = 2000
test_data_unscaled = data[60000+test_idx[:test_data_size], :].astype(float)
test_labels = (labels[60000+test_idx[:test_data_size]] == pos)*2-1

# Preprocessing
train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
#validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)

def weak_learner(training_data, training_labels, distribution, dimension):
    '''
    :param training_data: numpy array of [x1...xn]
    :param training_labels: numpy array of [y1...yn]
    :param distribution: numpy array of D = [0.1,...,0.5]
    :param dimension: int of dimension
    :return: j,theta
    '''
    # initialize F*
    f_star = np.inf
    theta = None
    j_star = None
    b = 1 # TODO: fix
    for j in range(0, dimension):
        sorted_training_data, sorted_training_labels, sorted_distribution = sort_all_arrays_zipped(training_data, training_labels, distribution, j)
        sorted_training_data_j = [x[j] for x in sorted_training_data]
        sorted_training_data_j = np.append(sorted_training_data_j, [sorted_training_data_j[-1]+1], axis=0)
        f_1 = np.sum([sorted_distribution[i] for i in range(0, len(sorted_training_labels)) if sorted_training_labels[i] == 1])
        f_2 = 1 - f_1
        f = min(f_1, f_2)
        b = 1 if f_1 < f_2 else -1
        if f < f_star:
            f_star = f
            theta = sorted_training_data_j[0]-1
            j_star = j
        for i in range(0, len(training_data)):
            f_1 = f_1 - sorted_training_labels[i] * sorted_distribution[i]
            f_2 = f_2 + sorted_training_labels[i] * sorted_distribution[i]
            f = min(f_1, f_2)
            b = 1 if f_1 < f_2 else -1
            if f < f_star and sorted_training_data_j[i] != sorted_training_data_j[i+1]:
                f_star = f
                theta = (sorted_training_data_j[i] + sorted_training_data_j[i+1])/2
                j_star = j
    return theta, j_star, b


def sort_all_arrays_zipped(training_set, training_labels, distribution, j):
    ''' see https://stackoverflow.com/questions/1903462/how-can-i-zip-sort-parallel-numpy-arrays'''
    sorted_index = training_set[:, j].argsort()
    sorted_training_set = training_set[sorted_index]
    sorted_distribution = distribution[sorted_index]
    sorted_training_labels = training_labels[sorted_index]
    return sorted_training_set, sorted_training_labels, sorted_distribution


def h_t(h_t_theta, h_t_j, h_t_b, x):
    if x[h_t_j] <= h_t_theta:
        return 1*h_t_b
    else:
        return -1*h_t_b


def calculate_h_t_neq_y(h_t_theta, h_t_j, h_t_b, x, y):
    ''' returns true iff h_t(x) != y '''
    y_hat = h_t(h_t_theta, h_t_j, h_t_b, x)
    return y_hat != y


def calc_new_distribution(x, y, h_t_theta, h_t_j, h_t_b, w_t, distribution):
    arr = np.array([np.exp(-w_t*y[i]*h_t(h_t_theta, h_t_j, h_t_b, x[i]))*distribution[i] for i in range(0, len(distribution))])
    denominator = np.sum(arr)
    new_distribution = np.divide(arr, denominator)
    return new_distribution


def ada_boost(training_data, training_labels, T = 100):
    distribution = np.array([1 / len(training_data) for elem in training_data])
    h_list = []
    w_list = []
    for t in range(0, T):
        h_t_theta, h_t_j, h_t_b = weak_learner(training_data, training_labels, distribution, dimension=784)
        epsilon_t = np.sum([distribution[i]
                            for i in range(0, len(training_data))
                            if calculate_h_t_neq_y(h_t_theta, h_t_j, h_t_b, training_data[i], training_labels[i])])
        w_t = (np.log((1-epsilon_t)/epsilon_t))/2
        distribution = calc_new_distribution(training_data, training_labels, h_t_theta, h_t_j, h_t_b, w_t, distribution)
        h_list.append((h_t_theta, h_t_j, h_t_b))
        w_list.append(w_t)
        print('AdaBoost: Finished iteration ' + str(t))
    return h_list, w_list


def calculate_classifier(x, h_t_list, w_t_list):
    sum = np.sum([h_t(h_t_theta=h_t_list[j][0], h_t_j=h_t_list[j][1], h_t_b=h_t_list[j][2], x=x)*w_t_list[j] for j in range(0, len(h_t_list))])
    if sum >= 0:
        return 1
    else:
        return -1


def calculate_error(data, labels, h_t_list, w_t_list, T = 100):
    error_list = []
    for t in range(0,T):
        predicted_results = [calculate_classifier(data[i], h_t_list[:t], w_t_list[:t]) for i in range(0, len(data))]
        total_errors = 0
        for i in range(0, len(data)):
            if predicted_results[i] != labels[i]:
                total_errors += 1
        error_list.append(total_errors/len(data))
    print(error_list)
    return error_list


def main():
    T=100
    h_list, w_list = ada_boost(train_data, train_labels, T=T)
    training_error_list = calculate_error(train_data, train_labels, h_list, w_list, T=T)
    test_error_list = calculate_error(test_data, test_labels, h_list, w_list, T=T)
    t_list = [i for i in range(1, T+1)]
    plt.xlabel("t")
    plt.ylabel("error")
    plt.scatter(t_list, test_error_list, color='r')
    plt.scatter(t_list, training_error_list, color='b')
    red_patch = mpatches.Patch(color='r', label='Test error')
    blue_patch = mpatches.Patch(color='b', label='Training error')
    plt.legend(handles=[red_patch, blue_patch], loc='best')
    plt.savefig('Q01.png')
    plt.cla()


main()
