#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:27:02 2018

@author: joshrutta
"""
import numpy as np
import pandas as pd
from scipy.stats import binom
from scipy.special import beta
from scipy.special import comb
from scipy.special import digamma
from scipy.special import gammaln
import matplotlib.pyplot as plt

x_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW4/x.csv',header = None)

x = x_df.values
n = len(x)
#%%
c = np.random.choice(range(21),n)

#1b and c
k_param = [3,9,15]
logprob = np.zeros([len(k_param),50])
most_prob_clusters = np.zeros([len(k_param),n])
def calc_bin(x,i,theta,k):
    return comb(20,x[i])*(theta[k]**x[i])*((1-theta[k])**(20-x[i]))

plt.figure()
for i in range(len(k_param)):
    k = k_param[i]
    pi = (1/k)*np.ones(k)
    theta = np.random.uniform(0,1,size = k)
    for t in range(50):
        #E-Step
        n_idx = np.arange(n)
        k_idx = np.arange(k)
        k_idx[:,np.newaxis]
        phi = calc_bin(x,n_idx,theta,k_idx)*pi[k_idx]
        phi_prob=(phi.T/np.sum(phi,axis=1)).T
        
        #M-Step 
        pi = np.sum(phi_prob,axis=0)/n
        theta = np.sum(phi_prob*x,axis=0)/(20*np.sum(phi_prob,axis=0))
        
        #Calculate log marginal likelihood
        logsum = 0
        phi = calc_bin(x,n_idx,theta,k_idx)*pi[k_idx]
        logsum = np.sum(np.log(np.sum(phi,axis=1)))
        logprob[i][t] = logsum
    plt.plot(logprob[i][1:],label = 'k = %d'%k)
    # getting stuff for 1c
    phi = calc_bin(x,n_idx,theta,k_idx)*pi[k_idx]
    phi_prob=(phi.T/np.sum(phi,axis=1)).T
    most_prob_clusters[i] = np.argmax(phi_prob,axis=1)
    
plt.title('log marginal likelihoods for k = 3,9,15') 
plt.xlabel('iteration')
plt.ylabel('log marginal likelihood')
plt.legend()
plt.show()
for i in range(len(k_param)):
    k = k_param[i]
    plt.figure()
    plt.scatter(x,most_prob_clusters[i])
    plt.title('EM - Indices of most probable cluster for k =%d' % k)
    plt.show()

#%% 2 VI
k_param = [3,15,50]
logprob = np.zeros([len(k_param),1000])
 

def compute_phi(x,a,b,alpha,i,k):
    t1 = x[i]*(digamma(a[k])-digamma(a[k]+b[k]))
    t2 = (20-x[i])*(digamma(b[k])-digamma(a[k]+b[k]))
    t3 = digamma(alpha[k])
    phi = np.exp(t1+t2+t3)
    return ((phi.T)/np.sum(phi,axis=1)).T

def update_a_b(x,phi,a_old,b_old):
    a = np.sum(x*phi,axis=0)+a_old
    b = np.sum((20-x)*phi,axis=0)+b_old
    return a, b

def update_alpha(alpha_old,phi):
    alpha_new = alpha_old+np.sum(phi,axis=0)
    return alpha_new

def calc_var_loss(alpha,phi):
    alpha_sum = np.sum(alpha)
    t1 = np.sum(gammaln(alpha))
    t2 = -gammaln(alpha_sum)
    
    dig_alph = digamma(alpha)
    dig_alph = np.reshape(dig_alph,[len(dig_alph),1])
    t3 = -np.sum(phi@dig_alph)
    return t1+t2+t3

def VI(x,k):
#    np.random.seed(7865)
    loss = np.zeros(1000)
    #Initialization - specific to problem
    alpha_old = np.full(k,0.1)
    alpha = np.random.uniform(0,1,size = k)
    a_old = np.full(k,0.5)
    a = np.random.uniform(0,1,size = k)
    b_old = np.full(k,0.5)
    b = np.random.uniform(0,1,size = k)
    #
    n_idx = np.arange(len(x))
    k_idx = np.arange(k)
    for t in range(1000):
        phi = compute_phi(x,a,b,alpha,n_idx,k_idx)
        a,b = update_a_b(x,phi,a_old,b_old)
        alpha = update_alpha(alpha_old,phi)
        loss[t] = calc_var_loss(alpha,phi)
    return phi,a,b,alpha,loss

most_prob_clusters = np.zeros([len(k_param),n])
losses = np.zeros([len(k_param),1000])

for i in range(len(k_param)): 
    k = k_param[i]
    phi,a,b,alpha,loss = VI(x,k)
    most_prob_clusters[i] = np.argmax(phi,axis=1)
    losses[i] = loss
    plt.figure()
    plt.scatter(x,most_prob_clusters[i])
    plt.title('VI - Indices of most probable cluster for k =%d' % k)
    plt.show()
plt.figure()
for i in range(len(k_param)):
    k = k_param[i]
    plt.plot(losses[i][1:],label='k = %d'%k)
plt.title('Variational Objective Function for k = 3,15, 50')
plt.legend()
plt.show()

#%% 3b
a = 0.5
b = 0.5
alpha = 0.75
c = np.zeros(n).astype(int)
k = 30
cluster_counts = np.zeros(k)
cluster_counts[0] = n
theta = np.random.beta(a,b,size = k)
clusters_per_iter = np.zeros(1000)
six_largest_per_iter = np.zeros([6,1000])
for t in range(1000):
    for i in range(n):
        phi_i = np.zeros(k+1)
        for j in range(k):
            #c[i] = cluster data point i belongs to
            cluster_count_j = cluster_counts[j]
            if c[i]==j:
                cluster_count_j -= 1
            if cluster_count_j > 0:
                phi_i[j] = binom.pmf(x[i],20,theta[j])*(cluster_count_j/(alpha+n-1))
        #calculate prob new cluster
        phi_i[j+1] = (alpha/(alpha+n-1))*beta(x[i]+a,20-x[i]+b)/beta(a,b)
        #normalize phi_i
        phi_i = phi_i/np.sum(phi_i)
        #decrease from old c[i] from cluster_counts bc we're about to change it
        cluster_counts[c[i]] -= 1
        #choose cluster for c[i] based on discrete dist from phi_i
        c[i] = int(np.random.choice(np.arange(k+1),1,p = phi_i))
#        print('c[i] =',c[i])
        #if c[i] is new cluster (=k bc zero based indexing), generate new theta,
        #increment k, increment cluster_counts
        if c[i]==k:
#            theta = np.append(theta,np.random.beta(a+np.sum(x[c==j]),b+20*len(x[c==j])-np.sum(x[c==j])))
            theta = np.append(theta,np.random.beta(a+x[i],b+20-x[i]))
            k += 1
            cluster_counts = np.append(cluster_counts,0)
        #update cluster counts
        cluster_counts[c[i]]+=1
        # for any clusters that have zero points in them, delete cluster,
        # update cluster counts,k, and theta and update all c[i] with value greater than
        # cluster that was deleted
        shift = 0
        for j in range(len(cluster_counts)):
            j -= shift
            if cluster_counts[j] == 0:
                cluster_counts = np.delete(cluster_counts,j)
                theta = np.delete(theta,j)
                shift += 1
                k -= 1
                c[c>j]-= 1
    # resample theta
    for j in range(k):
        theta[j] = np.random.beta(a+np.sum(x[c==j]),b+20*len(x[c==j])-np.sum(x[c==j]))
    # save number of observations for 6 clusters with most data points per iteration
    m = len(cluster_counts)
    if m >= 6:
        six_largest_per_iter[:,t] = np.sort(cluster_counts)[-6:]
    else:
        six_largest_per_iter[-m:,t] = np.sort(cluster_counts)
    # save total number of clusters per iteration
    clusters_per_iter[t] = k
    print('iter =',t)
#%%
plt.figure()
for j in range(6):
    plt.plot(six_largest_per_iter[j,:])
plt.title('6 most probable clusters per iteration')
plt.ylabel('Number of data points in cluster')
plt.xlabel('iteration')
plt.show()
plt.figure()
plt.plot(clusters_per_iter)
plt.ylabel('Number of filled clusters')
plt.xlabel('iteration')
plt.title('Number of filled clusters per iteration')
plt.show()

            
                
            
        
        


