#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 19:05:13 2018

@author: joshrutta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

#Problem 1a
fb_scores_filename = '/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW5/hw5-data/CFB2017_scores.csv'
team_names_filename = '/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW5/hw5-data/TeamNames.txt' 
team_names = np.loadtxt(team_names_filename,dtype=str)
M = np.zeros([763,763])
fb_scores_df = pd.read_csv(fb_scores_filename)
fb_scores = fb_scores_df.as_matrix()
for game in fb_scores:
    
    i = game[0]-1
    i_score = game[1]
    j = game[2]-1
    j_score = game[3]
    
    if i_score>j_score:
        teamA_win = 1
    else:
        teamA_win = 0
    teamB_win = 1-teamA_win
    
    M[i,i] += (teamA_win + i_score/(i_score + j_score))
    M[j,j] += (teamB_win + j_score/(i_score + j_score))
    M[i,j] += (teamB_win + j_score/(i_score + j_score))
    M[j,i] += (teamA_win + i_score/(i_score + j_score))

M_sum = 1/np.sum(M,axis=1)

M_norm = M*M_sum[:,np.newaxis]

w0 = np.random.uniform(size = [1,763])

w_t_dict = {}
for t in [10,100,1000,10000]:
    print('t = ',t,':')
    w_t_dict[t] = w0@(np.linalg.matrix_power(M_norm,t))
    
vList = []

    
for t in [10,100,1000,10000]:
    v = pd.DataFrame(team_names)
    v[1] = w_t_dict[t][0].round(6)
    v = v.sort([1, 0], ascending=[0,1]).rename(index=int, columns={0: "t = {}".format(t), 1: "Weight"}).head(25)
    v = v.reset_index(drop=True)
    vList.append(v)
full_df = pd.concat(vList, axis=1)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(full_df)
#%%
#1b
pbar = progressbar.ProgressBar()
lambd,u = np.linalg.eig(M_norm.T)
u_mags = np.linalg.norm(u,axis=0)

w_inf = u[:,lambd.argmax()]/(np.sum(u[:,lambd.argmax()]))

w_inf.shape=[1,len(w_inf)]

w0 = (1/763)*np.ones([1,763])
abs_diff = np.zeros(10000)

abs_diff[0] = np.linalg.norm(w0-w_inf,ord=1)
w_t = w0
for t in range(1,10000):
    w_t = w_t@M_norm
    abs_diff[t]=np.linalg.norm(w_t-w_inf,ord=1)
plt.plot(abs_diff)
plt.title(r'$\||w_{\infty}-w_t||_1$ vs. t')
plt.ylabel(r'$\||w_{\infty}-w_t||_1$')
plt.xlabel('t')
#%%
#Problem 2
n1 = 3012
n2 = 8447
X = np.zeros([n1,n2])
filename = '/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW5/hw5-data/nyt_data.txt'
word_file = '/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW5/hw5-data/nyt_vocab.dat'

#loading and preprocessing text data
with open(filename) as f:
    doc = f.readlines()
docs = [x.strip().strip('\n').strip("'") for x in doc]
for i in range(len(docs)):
    for j in docs[i].split(','):
        X[int(j.split(':')[0])-1,i]=int(j.split(':')[1])
        
#%%
pbar = progressbar.ProgressBar()
rank = 25
W = np.random.uniform(low=1,high=2,size=[n1,rank])
H = np.random.uniform(low=1,high=2,size=[rank,n2])
obj_array = np.zeros(100)
for i in pbar(range(100)):
    
    t1 = np.divide(X,(W@H)+1e-16)
    #update H
    W_norm_vect = 1/(np.sum(W,axis=0))
    W_norm = W*W_norm_vect
    H = np.multiply(H,W_norm.T@t1)
    #update W
    t1 = np.divide(X,(W@H)+1e-16)
    H_norm_vect = 1/(np.sum(H,axis=1))
    H_norm_T = H.T*H_norm_vect
    W = np.multiply(W,(t1@H_norm_T))
    #calculate divergence penalty
    obj_array[i] = -np.sum(np.multiply(X,np.log(W@H+1e-16))-(W@H))
plt.plot(np.arange(1,101),obj_array)
plt.title('Divergence Penalty vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Divergence Penalty')
plt.show()

#%%
W_norm_vect = 1/(np.sum(W,axis=0))
W_norm = W*W_norm_vect
word_file = '/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW5/hw5-data/nyt_vocab.dat'
words = np.loadtxt(word_file,dtype=str)
top_10_groups = np.zeros([5,5,10])

vList = []

    
for i in range(rank):
    v = pd.DataFrame(words)
    v[1] = W_norm[:,i].round(6)
    v = v.sort([1, 0], ascending=[0,1]).rename(index=int, columns={0: "Topic {}".format(i+1), 1: "Weight"}).head(10)
    v = v.reset_index(drop=True)
    vList.append(v)
    
for num in [5,10,15,20,25]:
    print('\n',(pd.concat(vList[num-5:num], axis=1)),'\n')
        
    
        
