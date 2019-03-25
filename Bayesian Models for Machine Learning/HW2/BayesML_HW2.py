#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 9 16:24:18 2018

@author: joshrutta
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

ratings_test_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW2/movies_csv/ratings_test.csv',header = None) 
ratings_train_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW2/movies_csv/ratings.csv',header = None)

ratings_test = ratings_test_df.values
ratings_train = ratings_train_df.values

N =int(max(max(ratings_train[:,0]),max(ratings_test[:,0])))
M = int(max(max(ratings_train[:,1]),max(ratings_test[:,1])))
R = np.zeros([N,M])
R_test = np.zeros([N,M])
#populating M from ratings_train
for rating in ratings_train:
    R[int(rating[0]-1),int(rating[1]-1)] = rating[2]
for rating in ratings_test:
    R_test[int(rating[0]-1),int(rating[1]-1)] = rating[2]       


#%%
def EM_for_matrix_completion(c,d,var):
    I = np.eye(d)
    U = np.random.multivariate_normal(np.zeros(d),np.identity(d)*0.1,size=N)
    V = np.random.multivariate_normal(np.zeros(d),np.identity(d)*0.1,size=M).T
    expec_matrix = np.zeros([N,M])
    logprobs = np.zeros(100)
    for k in range(100):
        # E step
        idx1 = (R==1)
        idx0 = (R==-1)
        sqrt_var = np.sqrt(var)
        UV = np.dot(U,V)
        pdf = norm.pdf(-UV/sqrt_var)
        cdf = norm.cdf(-UV/sqrt_var)
        expec_matrix[idx1] = UV[idx1] + sqrt_var*(pdf[idx1]/(1-cdf[idx1]))
        expec_matrix[idx0] = UV[idx0] + sqrt_var*(-pdf[idx0]/(cdf[idx0]))
        # M step
        t1 = np.linalg.inv(np.dot(V,V.T) + I*(1/c))
        t2 = np.dot(V,expec_matrix.T)
        U = np.dot(t1,t2).T
    #    print('U.shape =',U.shape)
        t3 = np.linalg.inv(np.dot(U.T,U) + I*(1/c))
        t4 = (np.dot(U.T,expec_matrix))
        V = np.dot(t3,t4)
    #    print('V.shape =',V.shape)
        cdf = norm.cdf(UV/sqrt_var)
        logprobs[k] += np.sum(np.log(cdf[idx1])) + np.sum(np.log(1-cdf[idx0]))
#        logprobs[k] += -d*N/2*np.log(2*np.pi*c)-(1/(2*c))*np.sum(U**2)
#        logprobs[k] += -d*M/2*np.log(2*np.pi*c)-(1/(2*c))*np.sum(V**2)
        logprobs[k] += -(1/(2*c))*np.sum(U**2)
        logprobs[k] += -(1/(2*c))*np.sum(V**2)
    return logprobs
c = 1
d = 5
var = 1
logprobs = EM_for_matrix_completion(c,d,var)
plt.title('EM for Matrix Completion')
plt.xlabel('number iterations')
plt.ylabel('ln(p(R,U,V))')
plt.plot(np.arange(100),logprobs)
#%%
#mult_logprobs = np.zeros([100,5])
colors = ['r','b','g','y','k']
iters = np.arange(100)
iters = iters[20:]
c = 1
d = 5
var = 1
for i in range(5):
    logprobs = EM_for_matrix_completion(c,d,var)
    plt.plot(iters,logprobs[20:],c=colors[i])
plt.xlabel('number iterations')
plt.ylabel('ln(p(R,U,V))')
plt.title('Multiple Iterations of EM')
plt.show()

#%%
#problem 2c
conf=np.zeros([2,2])
R_pred = norm.cdf(np.dot(U,V)/np.sqrt(var))
for i,j in list(zip(*R_test.nonzero())):
        if R_test[i,j] == 1:
            if R_pred[i,j] > 0.5:
                conf[1,1]+=1
            elif R_pred[i,j]<=0.5:
                conf[0,1]+=1
        elif R_test[i,j] == -1:
             if R_pred[i,j] > 0.5:
                conf[1,0]+=1
             if R_pred[i,j] <= 0.5:
                conf[0,0]+=1
    
    