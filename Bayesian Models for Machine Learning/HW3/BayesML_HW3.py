import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special



x_set1_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/X_set1.csv',
                         header = None)
x_set2_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/X_set2.csv',
                        header = None)

x_set3_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/X_set3.csv',
                         header=None)

y_set1_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/y_set1.csv',
                         header = None)
y_set2_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/y_set2.csv',
                        header = None)

y_set3_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/y_set3.csv',
                         header=None)

z_set1_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/z_set1.csv',
                         header = None)
z_set2_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/z_set2.csv',
                        header = None)

z_set3_df = pd.read_csv('/Users/joshrutta/Desktop/Fall 2018/Bayesian ML/HW/HW3/Homework 3/data_csv/z_set3.csv',
                         header=None)

x_set1 = (x_set1_df.values).T
x_set2 = (x_set2_df.values).T
x_set3 = (x_set3_df.values).T

y_set1 = y_set1_df.values
y_set2 = y_set2_df.values
y_set3 = y_set3_df.values

z1 = z_set1_df.values
z2 = z_set2_df.values
z3 = z_set3_df.values



def VI_for_Bayes_LR(y,X,num_iter):
    d = X.shape[0]
    mu = np.zeros(d)
    sigma = np.eye(d)
    
    a_0 = 1e-16*np.ones(d)
    b_0 = 1e-16*np.ones(d)
    b = b_0
    e_0 = f_0 = 1
    f = f_0
    n = y.shape[0]
    L = np.zeros(num_iter)
    
    #e and a never change after first update
    e = (n/2) +e_0
    a = a_0+0.5 
    
    for t in range(num_iter):
        b,f,sigma,mu = update(a,b,b_0,e,f,f_0,sigma,mu,y,X)
#        print('a.shape =',a.shape,'; b.shape =',b.shape,'; mu.shape =',mu.shape,'; sigma.shape =',sigma.shape)
        L[t] =calc_loss(a,a_0,b,b_0,e,e_0,f,f_0,sigma,mu,y,X)
    return L,a,b,e,f,mu
    
def update_q_lambda(e,f,f_0,mu,sigma,y,X):
    t1 = sigma+(mu@mu.T)
    y_T_X_T = y.T@X.T
    y_T_y = y.T@y
    f = float(0.5*(y_T_y-2*(y_T_X_T@mu)+np.trace(t1@(X@X.T))+2*f_0))
    return f

def update_q_alpha(a,b,b_0,mu,sigma):
    for i in range(len(b)):
        b[i] = b_0[i]+0.5*(sigma[i][i]+mu[i]**2)
    return b

def update_q_w(a,b,e,f,y,X):
    t1 = np.diag(a/b)
    t2 = (e/float(f))*(X@X.T)
    sigma = np.linalg.inv(t1+t2)
    mu = (e/float(f))*(sigma@(X@y))
    return mu,sigma
def update(a,b,b_0,e,f,f_0,sigma,mu,y,X):
    b_old = np.copy(b) #d-dimension vector
    f_old = np.copy(f) #scalar
    sigma_old = np.copy(sigma) #dxd matrix

    mu_old = np.copy(mu) #d-dimension vector
    
    f = update_q_lambda(e,f,f_0,mu_old,sigma_old,y,X)
    
    b = update_q_alpha(a,b_old,b_0,mu_old,sigma_old)

    #updating variational params for q(w)
    mu, sigma = update_q_w(a,b_old,e,f_old,y,X)
    
    #updating variational params for q(lambda)
#    print('X.shape =',X.shape,'; y.shape =',y.shape,';X@y.shape =',(X@y).shape)
#    print('mu_old.shape =',mu_old.shape,';sigma_old.shape =',sigma_old.shape)
#    print('(X@X.T).shape =',(X@X.T).shape)
#    print('f =',f)
    #updating variational params for q(alpha)
#    print('mu_old.shape',mu_old.shape,'; np.diag(sigma_old).shape',np.diag(sigma_old).shape)
    return b,f,sigma,mu


def calc_loss(a,a_0,b,b_0,e,e_0,f,f_0,sigma,mu,y,X):
    d = sigma.shape[0]
    n = y.shape[0]
    
    t1 = -(d/2)*np.log(2*np.pi)
    t2 = 0.5*np.sum(special.digamma(a)-np.log(b))
    t3 = -0.5*(np.trace((sigma+mu@mu.T)@(np.diag(a/b))))
    
    l1 = t1+t2+t3
    
    t4 = e_0*np.log(f_0)-special.gammaln(e_0)
    t5 = (e_0-1)*(special.digamma(e)-np.log(f))
    t6 = -f_0*(e/f)
    
    
    l2 = t4+t5+t6
    
#    print('t4 =',t4,'; t5 =',t5,'; t6=',t6,'; t7 =',t7,'; t8 =',t8)
    
    
    t7 = np.sum(a_0*np.log(b_0)-special.gammaln(a_0)-b_0*(a/b))
    t8 = np.sum((a_0-1)*(special.digamma(a)-np.log(b)))
    
    l3 = t7+t8
    
    sign,logdet = np.linalg.slogdet(sigma)
    
    l4 = -(d/2)*np.log(2*np.pi+1)-0.5*(logdet*sign)
    
    t9 = np.log(f)-special.gammaln(e)
    t10 = (e-1)*special.digamma(e)-e
    
    l5 = t9+t10
    
    l6 = np.sum(np.log(b)-special.gammaln(a)+(a-1)*special.digamma(a)-a)
    
    t1 = (special.digamma(e)-np.log(f))
    t2 = (e/f)
    t3 = mu@mu.T+sigma
#    print('y.shape =',y.shape,'X.T.shape =',X.T.shape)
    
    l7 = 0.5*n*(t1-np.log(2*np.pi))-0.5*t2*((y.T@y)-2*((y.T@X.T)@mu)+np.trace(t3@(X@X.T)))


#    print('l1 = ',l1)
#    print('l2 = ',l2)
#    print('l3 = ',l3)
#    print('l4 = ',l4)
#    print('l5 = ',l5)
#    print('l7 = ',l7)
#    print('L =',l1+l2+l3+l4+l5)
    return l1+l2+l3-l4-l5-l6+l7
    
L1,a1,b1,e1,f1,mu1 = VI_for_Bayes_LR(y_set1,x_set1,500)
L2,a2,b2,e2,f2,mu2 = VI_for_Bayes_LR(y_set2,x_set2,500)
L3,a3,b3,e3,f3,mu3 = VI_for_Bayes_LR(y_set3,x_set3,500)

plt.plot(L1) 
plt.ylabel('Variational Objective Function')
plt.xlabel('Iteration')
plt.title('Data Set 1')
plt.figure()
plt.plot(L2)
plt.ylabel('Variational Objective Function')
plt.xlabel('Iteration')
plt.title('Data Set 2')
plt.figure()
plt.plot(L3)
plt.ylabel('Variational Objective Function')
plt.xlabel('Iteration')
plt.title('Data Set 3')
plt.show()
#%%
d1 = x_set1.shape[0]
d2 = x_set2.shape[0]
d3 = x_set3.shape[0]
plt.figure()
plt.plot(np.arange(d1),(b1/a1))
plt.ylabel(r'$1/\mathbb{E}_{q(\alpha_k)}$')
plt.xlabel('Dimension')
plt.title('Dataset 1')
plt.figure()
plt.plot(np.arange(d2),(b2/a2))
plt.ylabel(r'$1/\mathbb{E}_{q(\alpha_k)}$')
plt.xlabel('Dimension')
plt.title('Dataset 2')
plt.figure()
plt.plot(np.arange(d3),(b3/a3))
plt.ylabel(r'$1/\mathbb{E}_{q(\alpha_k)}$')
plt.xlabel('Dimension')
plt.title('Dataset 3')

e_lambda1 = e1/f1
e_lambda2 = e2/f2
e_lambda3 = e3/f3
#%% 2d
plt.figure()
y1_hat = x_set1.T@mu1
plt.plot(z1,y1_hat,label='y1_hat vs. z1')
plt.scatter(z1,y_set1,c='r',label='ground truth')
plt.plot(z1,np.sinc(z1))
plt.legend()
plt.show()
plt.figure()
y2_hat = x_set2.T@mu2
plt.plot(z2,y2_hat,label='y2_hat vs. z2')
plt.scatter(z2,y_set2,c='r',label='ground truth')
plt.plot(z2,np.sinc(z2))
plt.legend()
plt.show()
plt.figure()
y3_hat = x_set3.T@mu3
plt.plot(z3,y3_hat,label='y3_hat vs. z3')
plt.scatter(z3,y_set3,c='r',label='ground truth')
plt.plot(z3,np.sinc(z3))
plt.legend()
plt.show()
        
        





