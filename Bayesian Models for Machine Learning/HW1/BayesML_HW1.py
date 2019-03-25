import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
import math

x_test_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW1/EECS6720-hw1-data/X_test.csv',
                         header = None)
x_train_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW1/EECS6720-hw1-data/X_train.csv',
                        header = None)

y_train_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW1/EECS6720-hw1-data/label_train.csv',
                         header=None)
y_test_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW1/EECS6720-hw1-data/label_test.csv',
                        header=None)



def bayes_classifier(x_train_df,y_train_df,x_new):
    x_train = x_train_df.values

    N = x_train.shape[0]
 #   print ("y_train_df.iloc[:,0] = ",y_train_df.iloc[:,0])
    x_train_y_0_df = x_train_df.loc[y_train_df.iloc[:,0] == 0]
    y_train_y_0_df = y_train_df.loc[y_train_df.iloc[:,0] == 0]

    x_train_y_0 = x_train_y_0_df.values
    y_train_y_0 = y_train_y_0_df.values

    num_y0 = y_train_y_0.shape[0]
 #   print("type(num_y1) =",type(num_y0))
    x_train_y_1_df = x_train_df.loc[y_train_df.iloc[:,0] == 1]
    y_train_y_1_df = y_train_df.loc[y_train_df.iloc[:,0] == 1]

    x_train_y_1 = x_train_y_1_df.values
    y_train_y_1 = y_train_y_1_df.values
    num_y1 = y_train_y_1.shape[0]

    p_y1_given_y = (1 + num_y1) / (N + 2)
    p_y0_given_y = (1 + num_y0) / (N + 2)

    log_prob_y1 = np.log(p_y1_given_y)
    log_prob_y0 = np.log(p_y0_given_y)
    #create a d-dimensional vectors which contains the sums
    ## of training x across d-dimensions
    x_train_y_1_sums = np.sum(x_train_y_1,axis=0)
    x_train_y_0_sums = np.sum(x_train_y_0,axis=0)
    for i in range(len(x_new)):
        t1 = (x_train_y_1_sums[i]+1)*math.log(1+num_y1)
        t2 = special.gammaln(x_train_y_1_sums[i]+x_new[i]+1)
#        t2 = math.log(math.factorial(x_train_y_1_sums[i]+x_new[i]))
        t3 = special.gammaln(x_new[i]+1)
#        t3 = math.log(math.factorial(x_new[i]))
        t4 = special.gammaln(x_train_y_1_sums[i]+1)
#        t4 = math.log(math.factorial(x_train_y_1_sums[i]))
        t5 = (x_train_y_1_sums[i]+x_new[i]+1)*math.log(num_y1+2)

        log_prob_y1 += (t1 + t2 - (t3+t4+t5))

        u1 = (x_train_y_0_sums[i] + 1) * math.log(1 + num_y0)
        u2 = special.gammaln(x_train_y_0_sums[i]+x_new[i]+1)
#        u2 = math.log(math.factorial(x_train_y_0_sums[i] + x_new[i]))
        u3 = special.gammaln(x_new[i]+1)
#        u3 = math.log(math.factorial(x_new[i]))
        u4 = special.gammaln(x_train_y_0_sums[i]+1)
#        u4 = math.log(math.factorial(x_train_y_0_sums[i]))
        u5 = (x_train_y_0_sums[i] + x_new[i] + 1) * math.log(num_y0 + 2)

        log_prob_y0 += (u1 + u2 - (u3+u4+u5))

    init_prob_y1 = math.exp(log_prob_y1)
    init_prob_y0 = math.exp(log_prob_y0)

    prob_y1 = np.true_divide(init_prob_y1,(init_prob_y1+init_prob_y0))
    prob_y0 = np.true_divide(init_prob_y0, (init_prob_y1 + init_prob_y0))
    # print("prob_y1 = ",prob_y1)
    # print("prob_y0 = ", prob_y0)
    return prob_y0,prob_y1


x_test = x_test_df.values

y_test = y_test_df.values
conf_mat = np.zeros([2,2])
misclassified = []
misclassified_probs = []
probs = []
for i in range(x_test.shape[0]):
    x_new = x_test[i]
    p_y0,p_y1 = bayes_classifier(x_train_df,y_train_df,x_new)
    probs.append([p_y0,p_y1])
    pred = np.argmax([p_y0,p_y1])
    if pred != y_test[i]:
        misclassified_probs.append([p_y0,p_y1])
        misclassified.append(x_new)
    conf_mat[pred,y_test[i]] += 1
    # (0,0) not spam marked as not spam
    # (0,1) spam marked as not spam
    # (1,0) not spam marked as spam
    # (1,1) spam marked as spam
print("conf_mat:",conf_mat)

#%% Problem 4c

filepath = "/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW1/EECS6720-hw1-data/README.txt"
file = open(filepath,'r')
x_label = []
for line in file:
    x_label.append(str(line[:-1]))
sel = np.random.choice(np.arange(len(misclassified)),size=3)

colors = ['b','c','m']
for i in range(len(sel)):
    misclassified_ex = misclassified[sel[i]]
    plt.scatter(np.arange(54),misclassified_ex,c = colors[i],label = 'misclassified ex %d' % i)
    print("misclassified example %d" % i, ": p(y* = 1|x*,X,y) =",misclassified_probs[sel[i]][1],
          ",p(y* = 0|x*,X,y) =",misclassified_probs[sel[i]][0]) 
#calculating length-54 vector of expected lambda vals
# should be a/b for gamma(a,b) for posterior on lambda given data

x_train = x_train_df.values

N = x_train.shape[0]
#   print ("y_train_df.iloc[:,0] = ",y_train_df.iloc[:,0])
x_train_y_0_df = x_train_df.loc[y_train_df.iloc[:,0] == 0]
y_train_y_0_df = y_train_df.loc[y_train_df.iloc[:,0] == 0]

x_train_y_0 = x_train_y_0_df.values
y_train_y_0 = y_train_y_0_df.values

num_y0 = y_train_y_0.shape[0]
#   print("type(num_y1) =",type(num_y0))
x_train_y_1_df = x_train_df.loc[y_train_df.iloc[:,0] == 1]
y_train_y_1_df = y_train_df.loc[y_train_df.iloc[:,0] == 1]

x_train_y_1 = x_train_y_1_df.values
y_train_y_1 = y_train_y_1_df.values
num_y1 = y_train_y_1.shape[0]

#create a d-dimensional vectors which contains the sums
## of training x across d-dimensions
x_train_y_1_sums = np.sum(x_train_y_1,axis=0)
x_train_y_0_sums = np.sum(x_train_y_0,axis=0)

expec_lamba1 = (x_train_y_1_sums+1)/(1+num_y1)
expec_lamba0 = (x_train_y_1_sums+0)/(1+num_y0)

plt.scatter(np.arange(54),expec_lamba1,c='r',label = r'$\mathbb{E}[\lambda_1]$')
plt.scatter(np.arange(54),expec_lamba0,c='g', label = r'$\mathbb{E}[\lambda_0]$')
plt.legend(loc='best')

plt.xticks(np.arange(54),x_label, fontsize = 8,rotation=70)

plt.tight_layout()
plt.show()
#%% problem 4d
dists = np.linalg.norm(np.array(probs)-np.array([0.5,0.5]),axis=1)
closest_dist_idx = dists.argsort()[:3]

for i in range(len(closest_dist_idx)):
    ambig_pred = x_test[closest_dist_idx[i]]
    plt.scatter(np.arange(54),ambig_pred,c = colors[i],label = 'ambig pred %d' % i)
    print("ambiguous pred %d" % i, ": p(y* = 1|x*,X,y) =",probs[closest_dist_idx[i]][1],
          ",p(y* = 0|x*,X,y) =",probs[closest_dist_idx[i]][0]) 
    
plt.scatter(np.arange(54),expec_lamba1,c='r',label = r'$\mathbb{E}[\lambda_1]$')
plt.scatter(np.arange(54),expec_lamba0,c='g', label = r'$\mathbb{E}[\lambda_0]$')
plt.legend(loc='best')

plt.xticks(np.arange(54),x_label, fontsize = 8,rotation=70)

plt.tight_layout()
plt.show()
