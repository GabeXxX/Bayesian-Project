#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy
import math
from sklearn.gaussian_process.kernels import RBF


# # Useful Methods
# ### What you see in this section:
# 1. function *integrate*, taking as input x-values and f(x)-values as lists, giving as output an estimate of the integral $$\int_{T}f(t)dt.$$ 
# >Note that the accuracy of the estimation is largely impacted by the discretization step used in *T*.
# 2. function *proj*, taking as input the kernel K, the interval of interest T, the interval T0 where we want a generic funtion to be null, and the function f. It outputs the projection of f into the space H0, where functions are null along T0
# >Note: it outputs also the error done in solving the linear system, usually large depending on K.
# >Note: the method is not used in the rest of the code.
# 3. function *get_K0*, taking as input the kernel K, the interval of interest T, the interval T0 where we want a generic funtion to be null. It outputs the kernel K0 of H0, i.e. the kernel of the subspace H0 of function null along T0.

# In[2]:


def integrate(T, f): 
    tot = 0;
    for i in range(len(T)-1):
        tot = tot + (1/2 * (f[i]+f[i+1]) * (T[1]-T[0]))
    return tot

def proj(K,T,T0,f): #returns error and projection
    up_idx = int(np.where(T==T0[-1])[0])
    dw_idx = int(np.where(T==T0[0])[0])
    K_tau_tau = K[dw_idx:up_idx,dw_idx:up_idx]
    K_tau = K[dw_idx:up_idx,:]
    beta_tau = f[dw_idx:up_idx] 
    
    a = np.linalg.solve(np.linalg.inv(K_tau_tau),beta_tau)
    error = np.dot(a,K_tau_tau)-beta_tau
    proj = f-np.dot(a,K_tau)
    
    return error, proj

def get_K0(K,T,T0):
    up_idx = int(np.where(T==T0[-1])[0])
    dw_idx = int(np.where(T==T0[0])[0])
    K_tau_tau = K[dw_idx:up_idx,dw_idx:up_idx]
    K_tau = K[dw_idx:up_idx,:]
    
    return K - np.dot(np.dot(K_tau.T,np.linalg.inv(K_tau_tau)),K_tau)




class data_generator:
    def __init__(self, T, beta, n, covariates_kernel, sigma):
        self.T = T
        self.g = len(self.T)
        self.K = np.fromfunction(np.vectorize(lambda s, t: covariates_kernel(self.T[s], self.T[t])), (self.g, self.g), dtype=int)
        self.n = n
        self.beta = beta
        self.sigma = sigma
    
    def i_o(self):
        x = np.random.multivariate_normal(np.zeros(self.g), self.K, self.n)
        y = np.fromfunction(np.vectorize(lambda i: integrate(self.T,  x[i,:]*self.beta)+np.random.normal(0,self.sigma,1)), (self.n,), dtype=int)
        return x,y



# # Posterior Mean and Covariance
# ### What you see in this section:
# The class *Posterior* returns the constrained model outcomes thanks to the functions
# 1. *posterior_mean_cov*, returning the posterior mean and covariance
# 2. *fitted_values*, returning the values $\hat{y}$ predicted by the constrained model

# In[8]:


class posterior:
    def __init__(self, T, T0, x, y, K,sigma):
        self.T = T
        self.x = x
        self.y = y
        self.n = y.shape[0]
        self.g = T.shape[0]
        self.T0 = T0
        self.sigma = sigma
        self.K = K
        self.K0 = get_K0(self.K,self.T,self.T0)
    
    def get_K(self):
        return self.K
    
    def getter(self):
        return self.K0
        
    def Lx0(self,t):   
        L = np.zeros(self.n)    
        for i in np.arange(0,self.n):
            L[i] = integrate(self.T,self.K0[t, :] * self.x[i,:])
        return L
    
    def R0(self,i, j):
        Ri = np.fromfunction(np.vectorize(lambda t: integrate(self.T,self.K0[:,t] * self.x[i,:])), (self.g,), dtype=int)
        Rij = integrate(self.T, Ri * self.x[j,:])
        return  Rij
    
    def M(self):
        return np.fromfunction(np.vectorize(lambda i, j: self.R0(i, j)), (self.n, self.n), dtype=int)+np.identity(self.n)
    
    def posterior_mean_cov(self):
        M_mat = self.M()
        inv = np.linalg.inv(M_mat)
        S11 = np.dot(np.dot(np.transpose(np.ones(M_mat.shape[0])),inv),np.ones(M_mat.shape[0]))
        SY1 = np.dot(np.dot(np.transpose(self.y),inv),np.ones(M_mat.shape[0]))
        
        def m(t):
            mean = np.dot(np.dot(np.transpose(self.Lx0(t)), inv), (self.y - (SY1 / S11) * np.ones(len(self.y))))
            return mean
        def Kstar(s,t):
            cov = (self.sigma**2)*(self.K0[s,t] - np.dot(np.dot(np.transpose(self.Lx0(s)),inv),self.Lx0(t)))
            return cov
        
        m = np.fromfunction(np.vectorize(lambda t: m(t)), (self.g,), dtype=int)
        C = np.fromfunction(np.vectorize(lambda t, s: Kstar(s,t)), (self.g,self.g), dtype=int)
        return m,C
    
    def only_mean(self):
        inv = np.linalg.inv(self.M())
        def m(t):
            mean = np.dot(np.dot(np.transpose(self.Lx0(t)),inv),self.y)
            return mean
        m = np.fromfunction(np.vectorize(lambda t: m(t)), (self.g,), dtype=int)
        return m
    
    def fitted_values(self):
        m = self.only_mean()
        return np.fromfunction(np.vectorize(lambda i: integrate(self.T, self.x[i,:]*m)), (self.n,), dtype=int)


