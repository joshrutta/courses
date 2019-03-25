#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 22:29:30 2018

@author: joshrutta
"""

##Problem 1a

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mix_weights = [0.2,0.5,0.3]
obs = np.zeros([500,2])

means = np.array([[0,0],[3,0],[0,3]])


cov = np.array([[1,0],
                [0,1]])

for i in range(500):
    dist_choice = np.random.choice([0,1,2],p=mix_weights)
    mean = means[dist_choice]
    obs[i] = np.random.multivariate_normal(mean,cov)
    
plt.scatter(obs[:,0],obs[:,1])
#%%
point_clusters = np.zeros(500)
obj_func_val = np.zeros([20,6])
clusters_for_3 = np.zeros(500)
clusters_for_5 = np.zeros(500)

for k in range(2,6):
    centroids = np.random.randn(k,2)
    for i in range(20):
        #step 1: update each c, finding centroid x_i is closest to
        diffs = np.zeros([500,k])
        for j in range(k):
            diffs[:,j] = np.linalg.norm(obs-centroids[j],axis = 1)**2
        point_clusters = np.argmin(diffs,axis=1)
        #step 2: update each centroid
        #first count cluster freqs for each cluster
        val, counts = np.unique(point_clusters,return_counts=True)
        cluster_freqs = dict(zip(val,counts))
        #find each obs which belongs to cluster k, and avg them (for all k clusters)
        for j in range(k):
            obs_sum = np.sum(obs[point_clusters == j],axis=0)
            centroids[j] = (1/(cluster_freqs[j]))*obs_sum
        # in each iteration, calculate obj function
        obj_val = 0
        for j in range(k):
            #find squared distance of points from their respective clusters 
            sq_dists = np.linalg.norm(obs[point_clusters==j]-centroids[j],axis = 1)**2
            obj_val += np.sum(sq_dists)
        obj_func_val[i,k] = obj_val
    if k == 3:
        clusters_for_3 = point_clusters
    if k == 5:
        clusters_for_5 = point_clusters
plt.figure()       
for k in range(2,6):
    plt.plot(range(1,21),obj_func_val[:,k],label='k = %d'%k)
plt.legend()
plt.title("K-means Objective Function vs. Iterations")
plt.xlabel('# iterations')
plt.ylabel('K-means Objective Function')
plt.show()

#%% Problem 1b
plt.figure()
colors = ['b','r','g', 'y', 'm']
for k in range(3):
    plt.scatter(obs[clusters_for_3 == k][:,0],obs[clusters_for_3 == k][:,1],color = colors[k])
plt.title('K-means with k = 3')
plt.show()
plt.figure()
for k in range(5):
    plt.scatter(obs[clusters_for_5 == k][:,0],obs[clusters_for_5 == k][:,1],color = colors[k])
    plt.title('K-means with k = 5')
plt.show()

#%% Problem 2a
ratings_test_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW4/hw4-data/ratings_test.csv',header = None) 
ratings_train_df = pd.read_csv('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW4/hw4-data/ratings.csv',header = None)

ratings_test = ratings_test_df.as_matrix()
ratings_train = ratings_train_df.as_matrix()

n1 =int(max(max(ratings_train[:,0]),max(ratings_test[:,0])))
n2 = int(max(max(ratings_train[:,1]),max(ratings_test[:,1])))
M = np.zeros([n1,n2])
M_test = np.zeros([n1,n2])
#populating M from ratings_train
for rating in ratings_train:
    M[int(rating[0]-1),int(rating[1]-1)] = rating[2]
for rating in ratings_test:
    M_test[int(rating[0]-1),int(rating[1]-1)] = rating[2]        
#%%
import progressbar
var = 0.25
d = 10
lambd = 1
L = np.zeros([100,10])
pbar = progressbar.ProgressBar()
RMSE = np.zeros(10)
U = np.random.multivariate_normal(np.zeros(d),np.identity(d),size=[n1,10])
V = np.random.multivariate_normal(np.zeros(d),np.identity(d),size=[n2,10]).T

for i in pbar(range(10)): 
    for j in range(100):
        #first updating users
        #(k are the rows, l the columns)
        for k in range(n1):
            cols = np.nonzero(M[k,:])[0]
            v_mat_sum = np.dot(V[i,:,cols].T,V[i,:,cols])
            U[k,:,i]= np.dot(np.linalg.inv(var*lambd*np.identity(d)+v_mat_sum),np.dot(V[i,:,cols].T,M[k,cols]))
        #then updating objects
        #(k is now the columns, l the rows)
        for k in range(n2):
            rows = np.nonzero(M[:,k])[0]
            u_mat_sum = np.dot(U[rows,:,i].T,U[rows,:,i])
            V[i,:,k] = np.dot(np.linalg.inv(var*lambd*np.identity(d)+u_mat_sum),np.dot(U[rows,:,i].T,M[rows,k]))
        #calculate the log-joint likelihood   
        #find nonzero elements in M and subtract prediction based 
        #on u and v
        t1 = 0
        t2 = 0
        t3 = 0
        pred = np.dot(U[:,:,i],V[i,:,:])
        sq_diff = np.square(M[M!=0]-pred[M!=0])
        t1 = -(1/(2*var))*(np.sum(sq_diff))
        t2 = -(lambd/2)*np.sum(np.square(U[i,:,:]))
        t3 = -(lambd/2)*np.sum(np.square(V[:,:,i]))
        L[j,i] = t1+t2+t3
    # After 100 iterations, calculate RMSE
    pred = np.dot(U[:,:,i],V[i,:,:])
    t1 = np.square(M_test[M_test!=0]-pred[M_test!=0])
    t2 = np.mean(t1)
    RMSE[i] = t2**(0.5)
#%%
print('RMSE:')
for i in RMSE:
    print(np.round(i,4))
plt.figure()
colors = [str(.1*x) for x in range(10)]
for k in range(10):
    plt.plot(np.arange(2,101),L[1:,k],label = 'trial %d' % (k+1))
plt.title('Log Joint Likelihood vs. Iterations')
plt.xlabel('# iterations')
plt.ylabel('log joint likelihood')
plt.legend() 
plt.show()
#%%
#2b
movies = np.loadtxt('/Users/joshrutta/Desktop/Spring 2018/Machine Learning/HW4/hw4-data/movies.txt',delimiter='\n',dtype=str)
indices = {'Star Wars': -1, 'My Fair Lady':-1,'GoodFellas':-1}
for i in range(len(movies)):
    if movies[i][:9] == 'Star Wars':
        indices['Star Wars'] = i
    elif movies[i][:12] == 'My Fair Lady':
        indices['My Fair Lady'] = i
    elif movies[i][:10] == 'GoodFellas':
        indices['GoodFellas'] = i
        
V_best = V[np.argmax(L[99,:]),:,:]
for k,j in indices.items():
    print('Query Movie : ',k)
    v_j = V_best[:,j]
    v_j = np.reshape(v_j,[10,1])
    dists = np.linalg.norm(V_best-v_j,axis=0)
    closest_movies_indices = dists.argsort()[:11]
    print('Ten closest movies:')
    for i in range(1,11):
        index = closest_movies_indices[i]
        print('Movie :',movies[index],':','Distance :',np.round(dists[index],4))
    print('\n')
    

    
    
            
        
            
