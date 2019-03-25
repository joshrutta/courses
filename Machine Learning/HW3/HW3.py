#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Problem 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

x_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/gaussian_process/X_test.csv',header = None) 
x_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/gaussian_process/X_train.csv',header = None)

y_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/gaussian_process/y_test.csv',header = None) 
y_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/gaussian_process/y_train.csv',header = None)

x_test = x_test_df.as_matrix()
x_train = x_train_df.as_matrix()

y_test = y_test_df.as_matrix()
y_train = y_train_df.as_matrix()

def make_gaussian_kernel_matrix(b,x):
    K=np.zeros([x.shape[0],x.shape[0]])
    for i in range (0,x.shape[0]):
        for j in range (0,x.shape[0]):
            x_i = x[i]
            x_j = x[j]
            exp_arg = (-1/b)*((np.linalg.norm(x_i - x_j))**2)
            K[i,j] = np.exp(exp_arg)
    return K
def make_gaussian_mean(b,x,x_train,var,K,y):
    if len(x_train.shape) == 2:
        exp_arg = (-1/b)*(np.linalg.norm(x- x_train,axis = 1)**2)
    else:
        exp_arg = (-1/b)*((x- x_train)**2)
    K_x_D = np.exp(exp_arg)
    K_x_D = np.reshape(K_x_D, [1,350])
    t1 = np.linalg.inv(var*np.identity(K.shape[0])+K)
    mean = np.dot(np.dot(K_x_D,t1),y)
    return mean

gaussian_table = np.zeros([6,10])

b = [5,7,9,11,13,15]
var = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

for b_i in b:
    K = make_gaussian_kernel_matrix(b_i,x_train)
    for var_i in var:
        SSE = 0
        for i in range(0,x_test.shape[0]):
            y_pred = make_gaussian_mean(b_i,x_test[i],x_train,var_i,K,y_train)
            y_true = y_test[i]
            SSE += ((y_pred-y_true)**2)
        gaussian_table[b.index(b_i),var.index(var_i)] = ((1/42)*(SSE))**.5
#sns.heatmap(gaussian_table,annot=True,fmt='.3f')
print(gaussian_table)
#%% problem 1 d
b = 5
var = 2
y_pred = np.array([])
print('y_pred.shape = ',y_pred.shape)
x_4 = np.reshape(x_train[:,3],[350,1])
K = make_gaussian_kernel_matrix(b,x_4)
for i in range(0,x_test.shape[0]):
    y_pred_i = make_gaussian_mean(b,x_test[i,3],x_train[:,3],var,K,y_train)
    y_pred = np.append(y_pred,y_pred_i)
print('y_pred.shape = ',y_pred.shape)
plt.scatter(x_4,y_train, c = 'b',label = 'True Value')

x_test_4 = x_test[:,3]
x_test_4.shape = (x_test_4.shape[0],1)
A = np.insert(x_test_4,0,y_pred,axis = 1)
A = A[A[:,1].argsort()]
plt.plot(A[:,1],A[:,0],'r',label='Predicted Value')
plt.title('Gaussian Process with variance = 2 and b = 5')
plt.xlabel('Car Weight')
plt.ylabel('Mileage')
plt.legend()
plt.show()

#%% problem 2 

x_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/boosting/X_test.csv',header = None) 
x_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/boosting/X_train.csv',header = None)

y_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/boosting/y_test.csv',header = None) 
y_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW3/hw3-data/boosting/y_train.csv',header = None)

x_test = x_test_df.as_matrix()
x_test = np.insert(x_test,0,1,axis = 1)
x_train = x_train_df.as_matrix()
x_train = np.insert(x_train,0,1,axis = 1)

y_test = y_test_df.as_matrix()
y_train = y_train_df.as_matrix()

#
#def bootstrap_set(data_set):
#    #creates bootstrap set the size of the
#    #original set by randomly sampling with replacement
#    n = data_set.shape[0]
#    B = data_set[np.random.choice(n,size = n),:]
#    return B
n = x_train.shape[0]
f_boost_train_sum = np.zeros(y_train.shape)
error_array_train = np.array([]) 
f_boost_test_sum = np.zeros(y_test.shape)

error_array_test = np.array([]) 
#array to keep track of how many times data points were selected
rand_index_hist = np.zeros(n)

eps_array = np.array([])
alpha_array = np.array([])

ub = np.array([])
boost_weights = (1/n)*np.ones(n)
boost_weights.shape = (boost_weights.shape[0],1)
for i in range(1500):
    rand_indices = np.random.choice(n,size = n,p=boost_weights[:,0])
    for i in rand_indices:
       rand_index_hist[i] += 1
    B_x = x_train[rand_indices,:]
    B_y = y_train[rand_indices,:]
    # Least squares weight vector using bootstrap set B
    t1 = np.linalg.inv(np.dot(B_x.T,B_x))
    t2 = np.dot(B_x.T,B_y)
    w = np.dot(t1,t2)
    #Getting training and testing predictions
    y_pred_train = np.sign(np.dot(x_train,w))
    y_pred_test = np.sign(np.dot(x_test,w))
    ind_func = ~np.equal(y_pred_train,y_train)
    eps = np.dot(boost_weights.T,ind_func)
    # If eps > 0.5, set w = -w and recalculate error
    if eps > 0.5:
        w = -w
        y_pred_train = np.sign(np.dot(x_train,w))
        y_pred_test = np.sign(np.dot(x_test,w))
        ind_func = ~np.equal(y_pred_train,y_train)
        eps = np.dot(boost_weights.T,ind_func)
    alpha = 0.5*np.log((1-eps)/eps)
    alpha_array = np.append(alpha_array,alpha)
    eps_array = np.append(eps_array,eps)
    #performing update on distribution
    boost_weights = boost_weights*np.exp(-alpha*y_pred_train*y_train)
    boost_weights = boost_weights/(np.sum(boost_weights))
    #calculating f_boost_train on this particular step and the error
    f_boost_train_sum += alpha*y_pred_train
    f_boost_test_sum +=alpha*y_pred_test
    #taking sign of f_boost_sums
    f_boost_train = np.sign(f_boost_train_sum)
    f_boost_test = np.sign(f_boost_test_sum)
    #calculating f_boost_test on this particular step and the error

    #error is number incorrect over total number of examples
    error_train_i = np.sum(~np.equal(f_boost_train,y_train))/(y_train.shape[0])*100
    error_array_train = np.append(error_array_train,error_train_i)
    
    error_test_i = np.sum(~np.equal(f_boost_test,y_test))/(y_test.shape[0])*100
    error_array_test = np.append(error_array_test,error_test_i)
    #calculate upperbound of training error
    t1 = (0.5 - eps_array)**2
    ub = np.append(ub,np.exp(-2*np.sum(t1)))

#plotting testing and training error
plt.plot(range(1,1501),error_array_train,'b')
plt.plot(range(1,1501),error_array_test,'r')
plt.legend(['Training Error','Testing Error'])
plt.title('Training and Testing Error of Boosted LS Classifier vs. Boosting Rounds')
plt.ylabel('Error (%)')
plt.xlabel('Boosting Round')
plt.show()
#%%plotting upper bound of training error against t
plt.figure()
plt.plot(range(1,1501),ub)
plt.title('Training Error Upper Bound vs. Boosting Round')
plt.ylabel('Training Error Upper Bound')
plt.xlabel('Boosting Round')
plt.show()
#%%plotting histogram of data points accessed
plt.figure()
plt.stem(rand_index_hist,markerfmt='C0-')
plt.title('Histogram of Training Data Indices Accessed')
plt.show()
#%%plot eps and alpha as fn of t
plt.figure()
plt.plot(range(1,1501),eps_array)
plt.ylabel('Epsilon')
plt.xlabel('Boosting Round')
plt.title('Epsilon vs. Boosting Round')
plt.show()
plt.figure()
plt.plot(range(1,1501),alpha_array,)
plt.ylabel('Alpha')
plt.xlabel('Boosting Round')
plt.title('Alpha vs. Boosting Round')
plt.show()


    
        

           
            
        
