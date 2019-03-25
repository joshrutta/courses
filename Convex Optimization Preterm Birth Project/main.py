import time
import numpy as np
from sklearn.metrics import f1_score
from utils import load_data, split_data, feature_time_mapping, metrics, save_result
from models import LogisticRegression, SVM, SVMPlus, gaussian_kernel

TEST_MODE = True

# load the data
x, y = load_data()

# load time assignments
time_assignment = feature_time_mapping()

# get classification times
times = np.unique(time_assignment)

# loop over the folds
FOLDS = 50
for f in range(FOLDS):

    # print time
    print('\n****************** Fold {:d}/{:d} ******************'.format(f + 1, FOLDS))

    # split the data
    x_train, y_train, x_valid, y_valid = split_data(x, y, 0.10, TEST_MODE)

    # loop over the classification times
    for t in times:

        # print time
        print('\n****************** t{:d} ******************'.format(int(t)))

        # separate normal information from privileged information
        x_norm_train = x_train[:, time_assignment <= t]
        x_priv_train = x_train[:, time_assignment > t]
        x_norm_valid = x_valid[:, time_assignment <= t]
        x_priv_valid = x_valid[:, time_assignment > t]
        assert x_norm_train.shape[1] + x_priv_train.shape[1] == x.shape[1]
        assert x_norm_valid.shape[1] + x_priv_valid.shape[1] == x.shape[1]

        # take data to kernel space
        k_norm_train = gaussian_kernel(x_norm_train, x_norm_train)
        k_norm_valid = gaussian_kernel(x_norm_train, x_norm_valid)
        k_priv_train = gaussian_kernel(x_priv_train, x_priv_train)

        # loop over svm model parameter space
        mdl = LogisticRegression()
        params = mdl.hyper_parameters()
        for p in params:

            # train the model
            t_start = time.time()
            success = mdl.train(x_norm_train, y_train, gamma=p['gamma'])

            # did we succeed?
            if success:

                # test the model with linear features
                y_hat = mdl.predict(x_norm_valid)

                # get metrics
                recall, precision, f1 = metrics(y_valid, y_hat)
                save_result(t, 'lr', 'linear', recall, precision, f1, C=None, gamma=p['gamma'])

                # print result
                t_elapsed = time.time() - t_start
                print('Logistic Regression w/ gamma = {:.2e}'.format(p['gamma'],) +
                      ' | Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f},  '.format(precision, recall, f1) +
                      ' | Time = {:.2f} seconds'.format(t_elapsed))

        # loop over svm model parameter space
        mdl = SVM()
        params = mdl.hyper_parameters()
        for p in params:

            # train the model with linear features
            t_start = time.time()
            success = mdl.train(x_norm_train, y_train, C=p['C'], mode='primal')

            # did we succeed?
            if success:

                # test the model
                y_hat = mdl.predict(x_norm_valid)

                # get metrics
                recall, precision, f1 = metrics(y_valid, y_hat)
                save_result(t, 'svm', 'linear', recall, precision, f1, C=p['C'], gamma=None)

                # print result
                t_elapsed = time.time() - t_start
                print('Linear SVM w/ C = {:.2e}'.format(p['C']) +
                      ' | Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f},  '.format(precision, recall, f1) +
                      ' | Time = {:.2f} seconds'.format(t_elapsed))

        # loop over svm model parameter space
        mdl = SVM()
        params = mdl.hyper_parameters()
        for p in params:

            # train the model with kernel features
            t_start = time.time()
            success = mdl.train(k_norm_train, y_train, C=p['C'], mode='primal')

            # did we succeed?
            if success:

                # test the model
                y_hat = mdl.predict(k_norm_valid)

                # get metrics
                recall, precision, f1 = metrics(y_valid, y_hat)
                save_result(t, 'svm', 'gaussian', recall, precision, f1, C=p['C'], gamma=None)

                # print result
                t_elapsed = time.time() - t_start
                print('Guassian SVM w/ C = {:.2e}'.format(p['C']) +
                      ' | Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f},  '.format(precision, recall, f1) +
                      ' | Time = {:.2f} seconds'.format(t_elapsed))

        # loop over svm+ model parameter space
        mdl = SVMPlus()
        params = mdl.hyper_parameters()
        for p in params:

            # train the model with linear features
            t_start = time.time()
            if x_priv_train.size > 0:
                success = mdl.train(x_norm_train, y_train, x_star=x_priv_train, C=p['C'], gamma=p['gamma'])
            else:
                success = mdl.train(x_norm_train, y_train, C=p['C'], gamma=p['gamma'])

            # did we succeed?
            if success:

                # test the model
                y_hat = mdl.predict(x_norm_valid)

                # get metrics
                recall, precision, f1 = metrics(y_valid, y_hat)
                save_result(t, 'svm+', 'linear', recall, precision, f1, C=p['C'], gamma=p['gamma'])

                # print result
                t_elapsed = time.time() - t_start
                print('Linear SVM+ w/ C = {:.2e}, gamma = {:.2e}'.format(p['C'], p['gamma']) +
                      ' | Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f},  '.format(precision, recall, f1) +
                      ' | Time = {:.2f} seconds'.format(t_elapsed))

        # loop over svm+ model parameter space
        mdl = SVMPlus()
        params = mdl.hyper_parameters()
        for p in params:

            # train the model with kernel features
            t_start = time.time()
            if x_priv_train.size > 0:
                success = mdl.train(k_norm_train, y_train, x_star=k_priv_train, gamma=p['gamma'], C=p['C'])
            else:
                success = mdl.train(k_norm_train, y_train, gamma=p['gamma'], C=p['C'])

            # did we succeed?
            if success:

                # test the model
                y_hat = mdl.predict(k_norm_valid)

                # get metrics
                recall, precision, f1 = metrics(y_valid, y_hat)
                save_result(t, 'svm+', 'gaussian', recall, precision, f1, C=p['C'], gamma=p['gamma'])

                # print result
                t_elapsed = time.time() - t_start
                print('Gaussian  SVM+ w/ C = {:.2e}, gamma = {:.2e}'.format(p['C'], p['gamma']) +
                      ' | Precision = {:.4f}, Recall = {:.4f}, F1 = {:.4f},  '.format(precision, recall, f1) +
                      ' | Time = {:.2f} seconds'.format(t_elapsed))
