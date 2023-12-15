import numpy as np
import matplotlib.pyplot as plt
import scipy


# # Paper 1 - Uncostrained model 

# # Useful Methods
# ### What you see in this section:
# Function *integrate*, taking as input x-values and f(x)-values as lists, giving as output an estimate of the integral $$\int_{T}f(t)dt.$$ 
# >Note that the accuracy of the estimation is largely impacted by the discretization step used in *T*.
# 
# 
# Definition of the Class "data_generator", using the following input
# 1. *T1, T2*, the extremes of the interval of interest T
# 2. *g*, setting the discretization of T, we will store $$\{ T_{1},T_{1}+\frac{T_{2}-T_{1}}{g},T_{1}+2\frac{T_{2}-T_{1}}{g},...,T_{2}\}$$
# 3. *beta*, the actual functional parameter, as a list of values
# 4. *n*, the number of functional covariates
# 5. *covariates_kernel*, the kernel of the functional covariates
# 6. *sigma*, the variance of the noise term in the regression model,
# 
# containing the function *i_o*, which return as output
# 1. *x*, a $n\times g$ np.array storing the covariates values
# 2. *y*, the actual scalar response, estimated by perturbating the true value obtained by means of the regression model.

# In[2]:


def integrate(T, f): 
    tot = 0;
    for i in range(len(T)-1):
        tot = tot + (1/2 * (f[i]+f[i+1]) * (T[1]-T[0]))
    return tot


# In[3]:

# In[4]:


class data_generator:
    def __init__(self, T1, T2, g, beta, n, covariates_kernel, sigma):
        self.T = np.linspace(T1,T2,g)
        self.K = np.fromfunction(np.vectorize(lambda s, t: covariates_kernel(self.T[s], self.T[t])), (g, g), dtype=int)
        self.g = g
        self.n = n
        self.beta = beta
        self.sigma = sigma
    
    def grid(self):
        return self.T
    
    def i_o(self):
        x = np.random.multivariate_normal(np.zeros(self.g), self.K, self.n)
        y = np.fromfunction(np.vectorize(lambda i: integrate(self.T,  x[i,:]*self.beta)+np.random.normal(0,self.sigma,1)), (self.n,), dtype=int)
        return x,y
        


# # Posterior Mean and Covariance
# ### What you see in this section:
# The class *Posterior* returns the constrained model outcomes thanks to the functions
# 1. *posterior_mean_cov*, returning the posterior mean and covariance
# 2. *fitted_values*, returning the values $\hat{y}$ predicted by the constrained model

# In[6]:


class posterior:
    def __init__(self, T, x, y, prior_kernel,sigma):
        self.T = T
        self.x = x
        self.y = y
        self.prior_kernel = prior_kernel
        self.n = y.shape[0]
        self.g = T.shape[0]
        self.sigma = sigma
        self.K = np.fromfunction(np.vectorize(lambda s, t: self.prior_kernel(self.T[s], self.T[t])), (self.g, self.g), dtype=int)
    
    def Lx(self,t):   
        L = np.zeros(self.n)    
        for i in np.arange(0,self.n):
            L[i] = integrate(self.T,self.K[t, :] * self.x[i,:])
        return L
    
    def R(self,i, j):
        Ri = np.fromfunction(np.vectorize(lambda t: integrate(self.T,self.K[:,t] * self.x[i,:])), (self.g,), dtype=int)
        Rij = integrate(self.T, Ri * self.x[j,:])
        return  Rij
    
    def cov(self):
        return np.fromfunction(np.vectorize(lambda i, j: self.R(i, j)), (self.n, self.n), dtype=int)
    
    def posterior_mean_cov(self):
        inv = np.linalg.inv(self.cov() + self.sigma * np.identity(self.n))
        def m(t):
            mean = np.dot(np.dot(np.transpose(self.Lx(t)),inv),self.y)
            return mean
        def Kstar(s,t):
            cov = self.K[s,t] - np.dot(np.dot(np.transpose(self.Lx(s)),inv),self.Lx(t))
            return cov
        
        m = np.fromfunction(np.vectorize(lambda t: m(t)), (self.g,), dtype=int)
        C = np.fromfunction(np.vectorize(lambda t, s: Kstar(s,t)), (self.g,self.g), dtype=int)
        return m,C
    
    def only_mean(self):
        inv = np.linalg.inv(self.cov() + self.sigma * np.identity(self.n))
        def m(t):
            mean = np.dot(np.dot(np.transpose(self.Lx(t)),inv),self.y)
            return mean
        m = np.fromfunction(np.vectorize(lambda t: m(t)), (self.g,), dtype=int)
        return m
    
    def fitted_values(self):
        m = self.only_mean()
        return np.fromfunction(np.vectorize(lambda i: integrate(self.T, self.x[i,:]*m)), (self.n,), dtype=int)

