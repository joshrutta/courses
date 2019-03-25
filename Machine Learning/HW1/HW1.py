#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 21:02:25 2018

@author: joshrutta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW1/hw1-data/X_test.csv',header = None) 
x_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW1/hw1-data/X_train.csv',header = None)

y_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW1/hw1-data/y_test.csv',header = None)
y_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW1/hw1-data/y_train.csv',header = None)

x_test = x_test_df.as_matrix()
x_train = x_train_df.as_matrix()

y_test = y_test_df.as_matrix()
y_train = y_train_df.as_matrix()

#2 (a)

Wrr_matrix = np.zeros((5001,7))

U,S,V = np.linalg.svd(x_train)

inv_X_t_X = np.linalg.inv(np.dot(x_train.T,x_train))
X_t_y = np.dot(x_train.T,y_train)

w_ls = np.dot(inv_X_t_X,X_t_y)
d_lambd = np.zeros((5001,1))
for lambd in range(0,5001):
    Wrr_matrix[lambd] = np.dot(np.linalg.inv(lambd*inv_X_t_X+np.identity(7)),w_ls).T
    d_lambd[lambd] = np.sum((S**2/(S**2+lambd)))
for i in range(0,7):
    plt.plot(d_lambd,Wrr_matrix[:,i],label = "dim "+str(i+1))
plt.legend(loc='best')
plt.xlabel('df(lambda)')
plt.grid()
plt.savefig('df_plot.pdf')
plt.gcf().clear()
#%%
#2 (b)
# Looking at the graph, we can see that dimension 4 and dimension 6 are outliers, 
# with dim 4 very negative, and dim 6 very positive. When variables
# are highly correlated, one large positive coefficient
# can be cancelled by a large negative correlated coefficient. Thus the 
# data suggests that the two coefficients are correlated

#%%
#2 (c)
RMSE = np.zeros((51,1))
for lambd in range(0,51):
    w_rr = np.array(Wrr_matrix[lambd])
    w_rr.shape = (7,1)
    yhat = np.dot(x_test,w_rr)
    RMSE[lambd] = (np.sum((y_test-yhat)**2))
RMSE = (1/42)*RMSE
plt.plot(list(range(0,51)),RMSE)
plt.xlabel('lambda')
plt.ylabel('RMSE^2')
plt.grid()
plt.savefig('RMSE^2.pdf')
plt.gcf().clear()
#%%
#2 (d)
RMSE = np.zeros((501,3))
# doing p = 2 polynomial regression
x_train2 = np.zeros((350,13))
x_train2[:,:6] = x_train[:,:-1]
x_train2[:,6:] = x_train**2

x_test2 = np.zeros((42,13))
x_test2[:,:6]=x_test[:,:-1]
x_test2[:,6:]=x_test**2

Wrr_matrix_2 = np.zeros((501,13))
U,S,V = np.linalg.svd(x_train2)
inv_X_t_X = np.linalg.inv(np.dot(x_train2.T,x_train2))
X_t_y = np.dot(x_train2.T,y_train)

w_ls = np.dot(inv_X_t_X,X_t_y)
for lambd in range(0,501):
    Wrr_matrix_2[lambd] = np.dot(np.linalg.inv(lambd*inv_X_t_X+np.identity(13)),w_ls).T
    
# doing p = 3 polynomial regression

x_train3 = np.zeros((350,19))
x_train3[:,:12]=x_train2[:,:-1]
x_train3[:,12:]=x_train**3

x_test3 = np.zeros((42,19))
x_test3[:,:12]=x_test2[:,:-1]
x_test3[:,12:]=x_test**3


Wrr_matrix_3 = np.zeros((501,19))
U,S,V = np.linalg.svd(x_train3)
inv_X_t_X = np.linalg.inv(np.dot(x_train3.T,x_train3))
X_t_y = np.dot(x_train3.T,y_train)

w_ls = np.dot(inv_X_t_X,X_t_y)
for lambd in range(0,501):
    Wrr_matrix_3[lambd] = np.dot(np.linalg.inv(lambd*inv_X_t_X+np.identity(19)),w_ls).T
    

for lambd in range(0,501):
    w_rr1 = np.array(Wrr_matrix[lambd])
    w_rr1.shape = (7,1)
    yhat1 = np.dot(x_test,w_rr1)
    
    w_rr2 = np.array(Wrr_matrix_2[lambd])
    w_rr2.shape = (13,1)
    yhat2 = np.dot(x_test2,w_rr2)
    
    w_rr3 = np.array(Wrr_matrix_3[lambd])
    w_rr3.shape = (19,1)
    yhat3 = np.dot(x_test3,w_rr3)
    
    RMSE1 = np.sum((y_test-yhat1)**2)**0.5
    RMSE2 = np.sum((y_test-yhat2)**2)**0.5
    RMSE3 = np.sum((y_test-yhat3)**2)**0.5
    RMSE[lambd] = [RMSE1,RMSE2,RMSE3]
RMSE = ((1/42)**0.5)*RMSE
plt.plot(list(range(0,501)),RMSE[:,0],label = "p = 1")
plt.plot(list(range(0,501)),RMSE[:,1],label = "p = 2")
plt.plot(list(range(0,501)),RMSE[:,2],label = "p = 3")
plt.legend()
plt.xlabel("lambda")
plt.grid()
plt.savefig('RMSE.pdf')

