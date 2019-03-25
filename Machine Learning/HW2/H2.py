#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 23:15:29 2018

@author: joshrutta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special

x_test_df = pd.read_csv('/Users/joshrutta/Desktop/Columbia University/Spring 2018/Machine Learning/HW2/hw2-data/X_test.csv',header = None) 
x_train_df = pd.read_csv('/Users/joshrutta/Desktop/Columbia University/Spring 2018/Machine Learning/HW2/hw2-data/X_train.csv',header = None)

y_test_df = pd.read_csv('/Users/joshrutta/Desktop/Columbia University/Spring 2018/Machine Learning/HW2/hw2-data/y_test.csv',header = None) 
y_train_df = pd.read_csv('/Users/joshrutta/Desktop/Columbia University/Spring 2018/Machine Learning/HW2/hw2-data/y_train.csv',header = None)

x_test = x_test_df.values
x_train = x_train_df.values

y_test = y_test_df.values
y_train = y_train_df.values

#2a

#first step: compute maximum likelihood params - pi, theta1's, and theta2's
pi_hat = (sum(y_train)[0])/(len(y_train))

#obtain submatrices of x_train where y = 0 and y = 1 respectively

x_train_with_y_0 = x_train_df[y_train_df[0]==0].values
x_train_with_y_1 = x_train_df[y_train_df[0]==1].values

#since there are 54 cols to use bernoulli dist on, theta1's will be 54 d vectors

theta1_0 = np.sum(x_train_with_y_0[:,:-3],0)/(x_train_with_y_0.shape[0])
theta1_1 = np.sum(x_train_with_y_1[:,:-3],0)/(x_train_with_y_1.shape[0])

#since there are 3 cols to use pareto dist on, theta2's will be 3 d vectors

theta2_0 = x_train_with_y_0.shape[0]/(np.sum(np.log(x_train_with_y_0[:,-3:]),0))
theta2_1 = x_train_with_y_1.shape[0]/(np.sum(np.log(x_train_with_y_1[:,-3:]),0))

#bayes classifier - given x's return the maximum from prob yielded from y = 1 and y = 0

# defining function to calculate probs for y = 0 and y = 1
# In this function, y is a scalar, x is row vector
#function returns scalar
def bayes_probs(y,x,pi_hat,theta1,theta2):
    #prior term
    term1 = ((pi_hat**(y))*((1-pi_hat)**(1-y)))
    term2 = 1
    term3 = 1
    #term2 = product of bayes terms
    for i in range(0,len(x)-3):
        term2 = term2*((theta1[i]**(x[i]))*((1-theta1[i])**(1-x[i])))
    #term3 = product of pareto terms
    for i in range(len(x)-3,len(x)):
        term3 = term3*(theta2[i-54]*(x[i]**(-(theta2[i-54]+1))))
        
    return term1*term2*term3

y_pred = np.zeros(y_test.shape)
confusion_matrix = np.zeros([2,2])
#set y_pred equal to 0 or 1 based on whichever yields higher probability
for i in range(0,x_test.shape[0]):
    y_pred[i] = int(bayes_probs(1,x_test[i,:],pi_hat,theta1_1,theta2_1) > \
    bayes_probs(0,x_test[i,:],pi_hat,theta1_0,theta2_0))
    
    confusion_matrix[y_test[i][0],int(y_pred[i][0])]=confusion_matrix[y_test[i][0],int(y_pred[i][0])]+1
    
print("accuracy = ",(100*(confusion_matrix[0,0]+confusion_matrix[1,1])/93),"% ")
#2b 
plt.subplot(1,2,1)
plt.stem(np.arange(1,55),theta1_0)
plt.xlabel('Bernoulli Parameter #')
plt.ylabel('Bernoulli Parameter Value')
plt.title('Bernoulli Parameters for Class 0')
plt.subplot(1,2,2)
plt.stem(np.arange(1,55),theta1_1)
plt.xlabel('Bernoulli Parameter #')
plt.title('Bernoulli Parameters for Class 1')
plt.show()
#%%
#2c - KNN

#y_pred which contains y_pred cols for k = 1,...,20
y_pred = np.zeros([y_test.shape[0],20])
#computing l1 dists for input row x_i against x_train
def l1_dists(x_train,x_i):
    t1 = abs(x_train - x_i)
    l1_dists = np.sum(t1,1)
    l1_dists.shape = (x_train.shape[0],1)
    return l1_dists
#x_i is row from x_test
def KNN(k,x_train,x_i,y_train):
    y_pred_i = 0
    l1_distances = l1_dists(x_train,x_i)
    k_nearest_y = np.array([])
    for i in range (0,k):
    #find closest y, append to list, delete from l1_dists
        closest_y = y_train[np.argmin(l1_distances)]
        k_nearest_y = np.append(k_nearest_y,closest_y)
        l1_distances = np.delete(l1_distances,np.argmin(l1_distances))
    k_nearest_y = k_nearest_y.astype(int)
    y_pred_i = np.argmax(np.bincount(k_nearest_y))
    return y_pred_i

for k in range(1,21):
    for i in range(0,y_test.shape[0]):
        y_pred[i][k-1] = KNN(k,x_train,x_test[i,:],y_train)
        
#computing accuracy 
accuracy = np.zeros([20,1])
y_test.shape = 93
for i in range(0,20):
    accuracy[i] = np.sum(np.equal(y_pred[:,i],y_test))/(y_test.shape[0])
plt.plot(np.arange(1,21),100*accuracy)
plt.xlabel("k")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()
plt.title("K-Nearest-Neighbors")

#%%
#2d
#def sigmoid(x):
#    return (1/(1+np.exp(-x)))
w = np.zeros([58,1])
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
loss = np.zeros([10000,1])
x_train2 = np.insert(x_train,0,1,axis=1)
for t in range(0,10000):
    eta = (1e5*((t+1)**.5))**-1
    #iterating through x_train, calculating value
    #to update w and also calculating loss fn for 
    # current value of w
    sig_arg = y_train*np.dot(x_train2,w)
    k = np.sum(np.log(special.expit(sig_arg)))
    
    update_param = (1-special.expit(sig_arg))*y_train*x_train2
    update_param = np.sum(update_param.T,axis = 1)
    update_param.shape = (58,1)
    w = w + eta*update_param
    loss[t] = k
plt.figure()
plt.plot(loss[1:10000])
plt.grid()
plt.ylabel("Objective Function Value")
plt.xlabel("Iteration #")
plt.title("Logistic Regression (Steepest Ascent)")

y_pred=np.zeros(y_test.shape)
for i in range(0,x_test.shape[0]):
    x_i=x_test[i,:]
    x_i = np.insert(x_i,0,1)
    x_i.shape = (x_test.shape[1]+1,1)
    sig_arg = np.dot(x_i.T,w)
    if(special.expit(sig_arg)[0][0]>.5):
        y_pred[i] = 1
    else:
        y_pred[i] = -1
num_correct = np.sum(np.equal(y_pred,y_test))
print("pred_accuracy = ",num_correct/(y_test.shape[0]))
    
#%%
#2e
w = np.zeros([58,1])
y_train[y_train == 0] = -1
y_test[y_test == 0] = -1
loss2 = np.zeros([100,1])
for t in range(0,100):
    eta = (t+1)**-.5
    update_param1 = 0
    update_param2 = 0
    loss_i = 0
    for i in range(0,x_train.shape[0]):
        x_i = x_train[i,:]
        x_i = np.insert(x_i,0,1)
        x_i.shape = (x_train.shape[1]+1,1)
        #iterating through x_train, calculating value
        #to update w and also calculating loss fn for 
        # current value of w
        sig_arg = y_train[i]*np.dot(x_i.T,w)
        #to avoid -inf
        x=max(special.expit(sig_arg),1e-10)
        loss_i += np.log(x)
        
        t1 = (1-special.expit(sig_arg))*y_train[i]*(x_i)
        t2 = -special.expit(sig_arg)*(1-special.expit(sig_arg))*\
            np.dot(x_i,x_i.T)
        update_param1 += t1
        update_param2 += t2
    
#    print("w.shape = ",w.shape)
#    print("update_param2 = ",update_param2.shape)
#    print("update_param1 = ",update_param1.shape)
#    print("np.dot((np.linalg.inv(update_param2)),update_param1).shape = ",\
#          np.dot((np.linalg.inv(update_param2)),update_param1).shape)
    w = w - eta*np.dot((np.linalg.inv(update_param2)),update_param1)
#    print("w.shape = ",w.shape)

          
    loss2[t] = loss_i
plt.figure()
plt.plot(loss2)
plt.grid()
plt.ylabel("Objective Function Value")
plt.xlabel("Iteration #")  
plt.title("Logistic Regression (Newton's Method)")

#printing accuracy 
y_pred=np.zeros(y_test.shape)
for i in range(0,x_test.shape[0]):
    x_i=x_test[i,:]
    x_i = np.insert(x_i,0,1)
    x_i.shape = (x_test.shape[1]+1,1)
    sig_arg = np.dot(x_i.T,w)
    if(special.expit(sig_arg)[0][0]>.5):
        y_pred[i] = 1
    else:
        y_pred[i] = -1
num_correct = np.sum(np.equal(y_pred,y_test))
print("pred_accuracy = ",num_correct/(y_test.shape[0]))
    
    
        
    
    