import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

# result directory
RESULT_DIR = 'Results'


def get_all_results():

    # get all result files
    result_files = [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR) if f.endswith('.p')]

    # initialize master result dictionary
    results = {'time': [],
               'alg': [],
               'kernel': [],
               'C': [],
               'gamma': [],
               'recall': [],
               'precision': [],
               'f1': []}

    # loop over the results to create master result dictionary
    for r in result_files:

        # load the pickle file
        result = pickle.load(open(r, 'rb'))

        # append results
        results['time'].append(result['time'])
        results['alg'].append(result['alg'])
        results['kernel'].append(result['kernel'])
        results['recall'].append(result['recall'])
        results['precision'].append(result['precision'])
        results['f1'].append(result['f1'])

        # handle parameters
        if 'gamma' in result.keys():
            results['gamma'].append(result['gamma'])
        else:
            results['gamma'].append(None)
        if 'C' in result.keys():
            results['C'].append(result['C'])
        else:
            results['C'].append(None)

    return results


def process_logistic_regression_at_t(results, t):

    # find all matching time indices
    i_t = [i for i, x in enumerate(results['time']) if x == t]

    # find all matching algorithm indices
    i_a = [i for i, x in enumerate(results['alg']) if x == 'lr']

    # fin all matching kernel indices
    i_k = [i for i, x in enumerate(results['kernel']) if x == 'linear']

    # take the intersection
    i_results = list(set(i_t) & set(i_a) & set(i_k))

    # find unique hyper-parameters
    gamma = np.array([results['gamma'][i] for i in i_results])
    unique_gamma = np.unique(gamma)

    # extract performance results
    recall = np.array([results['recall'][i] for i in i_results])
    precision = np.array([results['precision'][i] for i in i_results])
    f1 = np.array([results['f1'][i] for i in i_results])

    # size result tables
    recall_avg_table = np.zeros([1, unique_gamma.size])
    recall_std_table = np.zeros([1, unique_gamma.size])
    precision_avg_table = np.zeros([1, unique_gamma.size])
    precision_std_table = np.zeros([1, unique_gamma.size])
    f1_avg_table = np.zeros([1, unique_gamma.size])
    f1_std_table = np.zeros([1, unique_gamma.size])

    # loop over gamma
    for i in range(len(unique_gamma)):
        recall_avg_table[0, i] = np.nanmean(recall[unique_gamma[i] == gamma])
        recall_std_table[0, i] = np.nanstd(recall[unique_gamma[i] == gamma])
        precision_avg_table[0, i] = np.nanmean(precision[unique_gamma[i] == gamma])
        precision_std_table[0, i] = np.nanstd(precision[unique_gamma[i] == gamma])
        f1_avg_table[0, i] = np.nanmean(f1[unique_gamma[i] == gamma])
        f1_std_table[0, i] = np.nanstd(f1[unique_gamma[i] == gamma])

    # print latex tables
    caption = 'Logistic Regression at t{:d}: F1 Score (mean $\pm$ std.)'.format(t)
    print_latex_mat(unique_gamma, None, f1_avg_table, f1_std_table, a_label='$\\gamma$', caption=caption)
    caption = 'Logistic Regression at t{:d}: Recall (mean) / Precision (mean)'.format(t)
    print_latex_mat(unique_gamma, None, recall_avg_table, precision_avg_table, a_label='$\\gamma$', caption=caption, sep=' / ')

    # return maximum
    return np.max(f1_avg_table),np.max(precision_avg_table),np.max(recall_avg_table)


def process_svm_at_t(results, kernel, t):

    # find all matching time indices
    i_t = [i for i, x in enumerate(results['time']) if x == t]

    # find all matching algorithm indices
    i_a = [i for i, x in enumerate(results['alg']) if x == 'svm']

    # fin all matching kernel indices
    i_k = [i for i, x in enumerate(results['kernel']) if x == kernel]

    # take the intersection
    i_results = list(set(i_t) & set(i_a) & set(i_k))

    # find unique hyper-parameters
    C = np.array([results['C'][i] for i in i_results])
    unique_C = np.unique(C)

    # extract performance results
    recall = np.array([results['recall'][i] for i in i_results])
    precision = np.array([results['precision'][i] for i in i_results])
    f1 = np.array([results['f1'][i] for i in i_results])

    # size result tables
    recall_avg_table = np.zeros([1, unique_C.size])
    recall_std_table = np.zeros([1, unique_C.size])
    precision_avg_table = np.zeros([1, unique_C.size])
    precision_std_table = np.zeros([1, unique_C.size])
    f1_avg_table = np.zeros([1, unique_C.size])
    f1_std_table = np.zeros([1, unique_C.size])

    # loop over gamma
    for i in range(len(unique_C)):
        recall_avg_table[0, i] = np.nanmean(recall[unique_C[i] == C])
        recall_std_table[0, i] = np.nanstd(recall[unique_C[i] == C])
        precision_avg_table[0, i] = np.nanmean(precision[unique_C[i] == C])
        precision_std_table[0, i] = np.nanstd(precision[unique_C[i] == C])
        f1_avg_table[0, i] = np.nanmean(f1[unique_C[i] == C])
        f1_std_table[0, i] = np.nanstd(f1[unique_C[i] == C])

    # print latex tables
    caption = 'SVM with ' + kernel + ' kernel at t{:d}: F1 Score (mean $\pm$ std.)'.format(t)
    print_latex_mat(unique_C, None, f1_avg_table, f1_std_table, a_label='C', caption=caption)
    caption = 'SVM with ' + kernel + ' kernel at t{:d}: Recall (mean) / Precision (mean)'.format(t)
    print_latex_mat(unique_C, None, recall_avg_table, precision_avg_table, a_label='C', caption=caption, sep=' / ')

    # return maximum
    return np.max(f1_avg_table),np.max(precision_avg_table),np.max(recall_avg_table)


def process_svmp_at_t(results, kernel, t):

    # find all matching time indices
    i_t = [i for i, x in enumerate(results['time']) if x == t]

    # find all matching algorithm indices
    i_a = [i for i, x in enumerate(results['alg']) if x == 'svm+']

    # fin all matching kernel indices
    i_k = [i for i, x in enumerate(results['kernel']) if x == kernel]

    # take the intersection
    i_results = list(set(i_t) & set(i_a) & set(i_k))

    # find unique hyper-parameters
    gamma = np.array([results['gamma'][i] for i in i_results])
    unique_gamma = np.unique(gamma)
    C = np.array([results['C'][i] for i in i_results])
    unique_C = np.unique(C)

    # extract performance results
    recall = np.array([results['recall'][i] for i in i_results])
    precision = np.array([results['precision'][i] for i in i_results])
    f1 = np.array([results['f1'][i] for i in i_results])

    # size result tables
    recall_avg_table = np.zeros([unique_gamma.size, unique_C.size])
    recall_std_table = np.zeros([unique_gamma.size, unique_C.size])
    precision_avg_table = np.zeros([unique_gamma.size, unique_C.size])
    precision_std_table = np.zeros([unique_gamma.size, unique_C.size])
    f1_avg_table = np.zeros([unique_gamma.size, unique_C.size])
    f1_std_table = np.zeros([unique_gamma.size, unique_C.size])

    # loop over parameters
    for i in range(len(unique_gamma)):
        for j in range(len(unique_C)):
            recall_avg_table[i, j] = np.nanmean(recall[(unique_gamma[i] == gamma) * (unique_C[j] == C)])
            recall_std_table[i, j] = np.nanstd(recall[(unique_gamma[i] == gamma) * (unique_C[j] == C)])
            precision_avg_table[i, j] = np.nanmean(precision[(unique_gamma[i] == gamma) * (unique_C[j] == C)])
            precision_std_table[i, j] = np.nanstd(precision[(unique_gamma[i] == gamma) * (unique_C[j] == C)])
            f1_avg_table[i, j] = np.nanmean(f1[(unique_gamma[i] == gamma) * (unique_C[j] == C)])
            f1_std_table[i, j] = np.nanstd(f1[(unique_gamma[i] == gamma) * (unique_C[j] == C)])

    # print latex tables
    caption = 'SVM+ with ' + kernel + ' kernel at t{:d}: F1 Score (mean $\pm$ std.)'.format(t)
    print_latex_mat(unique_gamma, unique_C, f1_avg_table, f1_std_table, a_label='$\\gamma$', b_label='C', caption=caption)
    caption = 'SVM+ with ' + kernel + ' kernel at t{:d}: Recall (mean) / Precision (mean)'.format(t)
    print_latex_mat(unique_gamma, unique_C, recall_avg_table, precision_avg_table, a_label='$\\gamma$', b_label='C', caption=caption, sep=' / ')

    # return maximum
    return np.max(f1_avg_table),np.max(precision_avg_table),np.max(recall_avg_table)


def print_latex_mat(a, b, mu, std, f1='.3f', a_label='', b_label='', caption='', label='', sep='$\pm$'):
    """
    :param a: table columns
    :param b: table rows
    :param mu: mean
    :param std: stdev
    :param f1: ctr formatting instructions
    :param a_label: a parameter table descriptor
    :param b_label: b parameter table descriptor
    :param caption: table Latex caption
    :param label: table Latex label
    :return:
    """

    # find unique a in non-sorted order
    i_a = np.unique(a, return_index=True)[1]
    a = np.array([a[index] for index in sorted(i_a)])

    # is b provided
    if b is not None:

        # find unique b in non-sorted order
        i_b = np.unique(b, return_index=True)[1]
        b = np.array([b[index] for index in sorted(i_b)])

        # reshape ctr into a matrix
        mu = np.array(mu).reshape([a.size, b.size])

    else:

        # reshape ctr into a vector
        mu = np.array(mu).reshape([a.size, 1])

    # take transpose of ctr since we print by column
    mu = mu.T

    # declare table
    print('\n')
    # print('\\begin{table}[H]')
    print('\\centering')
    print('\\begin{tabular}')
    if b is not None:
        print('{|c|' + 'c|' * (mu.shape[1]) + '}')
    else:
        print('{' + '|c|' * (mu.shape[1]) + '}')
    print('\\hline')

    # print header
    if b is not None:
        s = '&' + a_label + ' = {:.0e} '.format(a[0])
    else:
        s = a_label + ' = {:.0e} '.format(a[0])
    for c in range(1, mu.shape[1]):
        s += '&' + a_label + ' = {:.0e} '.format(a[c])
    print(s + '\\\\')
    print('\\hline')

    # get max position if using p/m
    if sep == '$\pm$':
        i_max = np.argwhere(mu == np.max(mu))
        r_max = np.array([i[0] for i in i_max])
        c_max = np.array([i[1] for i in i_max])
    else:
        r_max = -1
        c_max = -1

    # print rows
    for r in range(mu.shape[0]):
        if b is not None:
            s = '$' + b_label + '$ = {:.0e} '.format(b[r])
            if np.sum((r_max == r) * (c_max == 0)):
                s += '&\\textbf{'
                s += ('{:' + f1 + '} ').format(mu[r, 0]) + '}' + sep + (' {:' + f1 + '} ').format(std[r, 0])
            else:
                s += ('&{:' + f1 + '} ').format(mu[r, 0]) + sep + (' {:' + f1 + '} ').format(std[r, 0])
        else:
            if np.sum((r_max == r) * (c_max == 0)):
                s = '\\textbf{' + ('{:' + f1 + '} ').format(mu[r, 0]) + '}' + sep + (' {:' + f1 + '} ').format(std[r, 0])
            else:
                s = ('{:' + f1 + '} ').format(mu[r, 0]) + sep + (' {:' + f1 + '} ').format(std[r, 0])
        for c in range(1, mu.shape[1]):
            if np.sum((r_max == r) * (c_max == c)):
                s += '&\\textbf{'
                s += ('{:' + f1 + '} ').format(mu[r, c]) + '}' + sep + (' {:' + f1 + '} ').format(std[r, c])
            else:
                s += ('&{:' + f1 + '} ').format(mu[r, c]) + sep + (' {:' + f1 + '} ').format(std[r, c])
        print(s + '\\\\')
    print('\\hline')

    # complete table
    print('\\end{tabular}')
    print('\\vspace{2pt}')
    print('\\caption{' + caption + '}')
    # print('\\label{tab:' + label + '}')
    # print('\\end{table}')


if __name__ == '__main__':

    # get result files
    results = get_all_results()

    # initialize results
    lr_f1,lr_pr,lr_r = np.zeros(5),np.zeros(5),np.zeros(5)
    svm_l_f1,svm_l_pr,svm_l_r = np.zeros(5),np.zeros(5),np.zeros(5)
    svm_g_f1,svm_g_pr,svm_g_r = np.zeros(5),np.zeros(5),np.zeros(5)
    svmp_l_f1,svmp_l_pr,svmp_l_r = np.zeros(5),np.zeros(5),np.zeros(5)
    svmp_g_f1,svmp_g_pr,svmp_g_r = np.zeros(5),np.zeros(5),np.zeros(5)

    # loop over time
    for t in [0, 1, 2, 3, 4]:

        # process results
        print('\\begin{table}[H]')
        if t == 0:
            print('\section{Appendix}')
        print('\subsection{{T{:d} Complete Results}}'.format(t))
        lr_f1[t],lr_pr[t],lr_r[t] = process_logistic_regression_at_t(results, t)
        svm_l_f1[t],svm_l_pr[t],svm_l_r[t] = process_svm_at_t(results, 'linear', t)
        svm_g_f1[t],svm_g_pr[t],svm_g_r[t] = process_svm_at_t(results, 'gaussian', t)
        svmp_l_f1[t],svmp_l_pr[t],svmp_l_r[t] = process_svmp_at_t(results, 'linear', t)
        svmp_g_f1[t],svmp_g_pr[t],svmp_g_r[t] = process_svmp_at_t(results, 'gaussian', t)
        print('\n\\end{table}')

    # make top performer plot
    cmap = plt.get_cmap("tab10")
    fig = plt.figure()
    plt.suptitle('Top Mean Performance Per Model Across Time')
    plt.plot(lr_f1, label='Log. Reg.', color=cmap(2), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_l_f1, label='SVM Lin.', color=cmap(1), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_g_f1, label='SVM Gauss.', color=cmap(1), linestyle='--', marker='o', markersize=5.0)
    plt.plot(svmp_l_f1, label='SVM+ Lin.', color=cmap(0), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svmp_g_f1, label='SVM+ Gauss.', color=cmap(0), linestyle='--', marker='o', markersize=5.0)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel('MFMU Temporal Data Position')
    plt.ylabel('F1 Score')
    plt.legend()

    cmap = plt.get_cmap("tab10")
    fig = plt.figure()
    plt.suptitle('Top Mean Performance Per Model Across Time')
    plt.plot(lr_pr, label='Log. Reg.', color=cmap(2), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_l_pr, label='SVM Lin.', color=cmap(1), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_g_pr, label='SVM Gauss.', color=cmap(1), linestyle='--', marker='o', markersize=5.0)
    plt.plot(svmp_l_pr, label='SVM+ Lin.', color=cmap(0), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svmp_g_pr, label='SVM+ Gauss.', color=cmap(0), linestyle='--', marker='o', markersize=5.0)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel('MFMU Temporal Data Position')
    plt.ylabel('Precision')
    plt.legend()

    cmap = plt.get_cmap("tab10")
    fig = plt.figure()
    plt.suptitle('Top Mean Performance Per Model Across Time')
    plt.plot(lr_r, label='Log. Reg.', color=cmap(2), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_l_r, label='SVM Lin.', color=cmap(1), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svm_g_r, label='SVM Gauss.', color=cmap(1), linestyle='--', marker='o', markersize=5.0)
    plt.plot(svmp_l_r, label='SVM+ Lin.', color=cmap(0), linestyle=':', marker='x', markersize=5.0)
    plt.plot(svmp_g_r, label='SVM+ Gauss.', color=cmap(0), linestyle='--', marker='o', markersize=5.0)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xlabel('MFMU Temporal Data Position')
    plt.ylabel('Recall')
    plt.legend()
    plt.show()



