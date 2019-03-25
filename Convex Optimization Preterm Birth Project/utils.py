import os
import csv
import time
import pickle
import numpy as np
from scipy.io import loadmat


def load_data():

    # load data
    x = loadmat(os.path.join(os.getcwd(), 'Data', 't4.data.mat'))
    x = x['D']

    # load labels
    y = loadmat(os.path.join(os.getcwd(), 'Data', 't4.lbls.mat'))
    y = y['Y'][:, 1]

    return x, y


def split_data(x, y, percent_test, test_mode=False):

    # split data into positive and negative groups
    x_pos = x[y == +1]
    x_neg = x[y == -1]

    # generate and randomly shuffle the two groups
    i_pos = np.arange(x_pos.shape[0])
    np.random.shuffle(i_pos)
    i_neg = np.arange(x_neg.shape[0])
    np.random.shuffle(i_neg)

    # test mode trims negative examples down
    if test_mode:
        i_neg = i_neg[:len(i_pos)]

    # determine cut-points
    i_cut_pos = int(len(i_pos) * (1 - percent_test))
    i_cut_neg = int(len(i_neg) * (1 - percent_test))

    # split positive group
    x_pos_train = x_pos[i_pos[:i_cut_pos]]
    x_pos_valid = x_pos[i_pos[i_cut_pos:]]

    # split negative group
    x_neg_train = x_neg[i_neg[:i_cut_neg]]
    x_neg_valid = x_neg[i_neg[i_cut_neg:]]

    # assemble the training group
    x_train = np.concatenate([x_pos_train, x_neg_train], axis=0)
    y_train = np.concatenate([np.ones(x_pos_train.shape[0]), -np.ones(x_neg_train.shape[0])], axis=0)

    # assemble the validation group
    x_valid = np.concatenate([x_pos_valid, x_neg_valid], axis=0)
    y_valid = np.concatenate([np.ones(x_pos_valid.shape[0]), -np.ones(x_neg_valid.shape[0])], axis=0)

    return x_train, y_train, x_valid, y_valid


def load_feature_names(feature_name_file):

    # open feature name file
    with open(feature_name_file) as f:

        # configure csv reader
        reader = csv.reader(f, delimiter=',')

        # load the names
        feature_names = [l for l in reader][0]

    return feature_names


def feature_time_mapping():

    # get all feature names
    all_feats = load_feature_names(os.path.join(os.getcwd(), 'Data', 't4.atb.csv'))

    # initialize time assignments
    time_assignment = 4 * np.ones(len(all_feats))

    # loop backward through time
    for t in np.arange(3, -1, -1):

        # load the next set of feature names
        feat_names = load_feature_names(os.path.join(os.getcwd(), 'Data', 't{:d}.atb.csv'.format(t)))

        # loop over the names
        for n in feat_names:

            # does the name exist in the master list
            if n in all_feats:

                # find its location
                i = all_feats.index(n)

                # give it a time assignment
                time_assignment[i] = t

    return time_assignment


def metrics(y_valid, y_predict):

    # compute number of true positives
    tp = np.sum((y_valid == +1) * (y_predict == +1))
    fp = np.sum((y_valid == -1) * (y_predict == +1))
    fn = np.sum((y_valid == +1) * (y_predict == -1))

    # compute recall
    recall = tp / (tp + fn)

    # compute precision
    precision = tp / (tp + fp)

    # compute f1 score
    f1 = 2 / (recall ** -1 + precision ** -1)

    return recall, precision, f1


def save_result(t, alg, kernel, recall, precision, f1, C=None, gamma=None):

    # load a dictionary
    result = {'time': t,
              'alg': alg,
              'kernel': kernel,
              'recall': recall,
              'precision': precision,
              'f1': f1}

    # save C if applicable
    if C is not None:
        result.update({'C': C})

    # save gamma if applicable
    if gamma is not None:
        result.update({'gamma': gamma})

    # save file
    file = os.path.join(os.getcwd(), 'Results', str(int(time.time() * 1e7)) + '.p')
    pickle.dump(result, open(file, 'wb'))


if __name__ == '__main__':

    # load feature time assignments
    time_assignment = feature_time_mapping()

    # load the data
    x, y = load_data()
