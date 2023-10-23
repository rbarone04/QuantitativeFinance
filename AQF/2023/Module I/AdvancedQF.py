# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:00:07 2022

@author: r.barone
"""


import numpy as np
import pandas as pd
import scipy.stats as si


class WienerProcess():
    """
    A Wiener Process class constructor
    """
    def __init__(self,maxTime = 1, steps = 1000):

        assert (type(maxTime)==float or type(maxTime)==int or maxTime > 0), "Expect a positive float number to define the time interval [0,maxTime]"
        
        self.__maxTime = maxTime
        self.__x0 = 0.
        self.__path = None
        self.__n_step = steps
        self.__time_vector = None
        self.__gen_path(self.__maxTime, self.__n_step)
        
    def steps(self):
        return self.__n_step
   
    def maxTime(self):
        return self.__maxTime
    
    def path(self):
        return self.__path
    
    def time_vector(self):
        return self.__time_vector
    
    def __set_time_vector(self, maxTime, n_step):
        self.__time_vector = np.linspace(0,maxTime, n_step)
    
    def new_path(self, T = 1, n_step = 1000):
        self.__gen_path(T, n_step)
        return self.__path
    
    def __gen_path(self, T = 1, n_step = 1000):
        self.__maxTime = T
        self.__n_step = n_step
        self.__set_time_vector(self.__maxTime, self.__n_step)
        
        w = np.ones(n_step)*self.__x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process construction
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        self.__path = w
        
class ArithmeticBM():
    """
    Class constructor for the Arithmetic Brownian Motion (a.k.a. Generalized Brownian Motion)
    """
    def __init__(self, wp = WienerProcess(), x_0=0., mu = 0., sigma = 1.):
        assert (type(x_0)==float or type(x_0)==int), "Expect a float for the initial condition x_0"
        assert (sigma >= 0.), "Expect a non negative value for sigma"
        
        self.__wp = wp
        self.__x_0 = x_0
        self.__mu = mu
        self.__sigma = sigma
        self.__path = None
        
        self.__gen_abm_path(self.__mu, self.__sigma)
    
    def x_0(self):
        return self.__x_0
    
    def mu(self):
        return self.__mu
    
    def sigma(self):
        return self.__sigma
    
    def wp(self):
        return self.__wp
    
    def path(self):
        return self.__path
    
    def new_path(self,x_0 = 0., mu = 0., sigma = 1.):
        self.__x_0 = x_0
        self.__mu = mu
        self.__sigma = sigma
        self.__gen_abm_path(mu, sigma)
        return self.__path
    
    def __gen_abm_path(self, mu = 0., sigma = 1.):
        n_step = self.__wp.steps()
        w_path = self.__wp.path()
        abm = np.ones(n_step)*self.__x_0
        for i in range(1,n_step):
            abm[i] += mu*i/n_step + sigma*w_path[i]
        self.__path = abm
        
class GeometricBM():
    """
    Class constructor for the Geometric Brownian Motion (used to describe the underlying dynamics in the Black-Scholes model)
    """
    def __init__(self, wp = WienerProcess(), x_0=1., mu = 0., sigma = 1.):
        assert (type(x_0)==float or type(x_0)==int or x_0 >0 ), "Expect a positive float number for the initial condition x_0"
        assert (sigma >= 0.), "Expect a non negative value for sigma"
        
        self.__wp = wp
        self.__x_0 = x_0
        self.__mu = mu
        self.__sigma = sigma
        self.__path = None
        
        self.__gen_gbm_path(self.__mu, self.__sigma)
    
    def x_0(self):
        return self.__x_0
    
    def mu(self):
        return self.__mu
    
    def sigma(self):
        return self.__sigma
    
    def wp(self):
        return self.__wp
    
    def path(self):
        return self.__path
    
    def new_path(self,x_0 = 0., mu = 0., sigma = 1.):
        self.__x_0 = x_0
        self.__mu = mu
        self.__sigma = sigma
        self.__gen_gbm_path(mu, sigma)
        return self.__path
    
    def __gen_gbm_path(self, mu = 0., sigma = 1.):
        time_vector = self.__wp.time_vector()
        deterministic_term = (mu - (sigma**2/2))*time_vector
        stochastic_term = sigma*self.__wp.path()
        self.__path = self.__x_0 *(np.exp(deterministic_term + stochastic_term))
        
class Vasicek():
    """
    Class constructor for the Gaussian mean-reverting process used in the Vasicek model
    """
    def __init__(self, wp = WienerProcess(), x_0=.005, theta = .03, k = 2., sigma = .04):
        assert (type(x_0)==float or type(x_0)==int or x_0 >0 ), "Expect a positive float number for the initial condition x_0"
        assert (sigma >= 0.), "Expect a non negative value for sigma"
        
        self.__wp = wp
        self.__x_0 = x_0
        self.__theta = theta
        self.__k = k
        self.__sigma = sigma
        self.__path = None
        
        self.__gen_vasicek_path(self.__theta, self.__k, self.__sigma)
    
    def x_0(self):
        return self.__x_0
    
    def theta(self):
        return self.__theta
    
    def k(self):
        return self.__k
    
    def sigma(self):
        return self.__sigma
    
    def wp(self):
        return self.__wp
    
    def path(self):
        return self.__path
    
    def new_path(self, x_0=.005, theta = .03, k = 2., sigma = 1.):
        self.__x_0 = x_0
        self.__theta = theta
        self.__k = k
        self.__sigma = sigma
        self.__gen_vasicek_path(theta,k,sigma)
        return self.__path
    
    def __gen_vasicek_path(self, theta = .03, k = 2., sigma = 1.):
        n_step = self.__wp.steps()
        dt = self.__wp.maxTime()/n_step
        w_path = self.__wp.path()
        vasicek = np.ones(n_step)*self.__x_0
        for i in range(1,n_step):
            determistic_term = k*(theta-vasicek[i-1])*dt
            stochastic_term = sigma*(w_path[i]-w_path[i-1])
            vasicek[i] = vasicek[i-1] + determistic_term +stochastic_term
        self.__path = vasicek

class CIR():
    """
    Class constructor for the Gaussian mean-reverting process used in the Cox-Ingersoll-Ross (CIR) model
    """
    def __init__(self, wp = WienerProcess(), x_0=.005, theta = .03, k = 2., sigma = .04):
        assert (type(x_0)==float or type(x_0)==int or x_0 >0 ), "Expect a positive float number for the initial condition x_0"
        assert (sigma >= 0.), "Expect a non negative value for sigma"
        if 2*k*theta <= sigma**2:
            print('WARNING! Feller condition does not hold.')
        self.__wp = wp
        self.__x_0 = x_0
        self.__theta = theta
        self.__k = k
        self.__sigma = sigma
        self.__path = None
        
        self.__gen_cir_path(self.__theta, self.__k, self.__sigma)
    
    def x_0(self):
        return self.__x_0
    
    def theta(self):
        return self.__theta
    
    def k(self):
        return self.__k
    
    def sigma(self):
        return self.__sigma
    
    def wp(self):
        return self.__wp
    
    def path(self):
        return self.__path
    
    def new_path(self, x_0=.005, theta = .03, k = 2., sigma = 1.):
        self.__x_0 = x_0
        self.__theta = theta
        self.__k = k
        self.__sigma = sigma
        self.__gen_cir_path(theta,k,sigma)
        return self.__path
    
    def __gen_cir_path(self, theta = .03, k = 2., sigma = 1.):
        n_step = self.__wp.steps()
        dt = self.__wp.maxTime()/n_step
        w_path = self.__wp.path()
        cir = np.ones(n_step)*self.__x_0
        for i in range(1,n_step):
            determistic_term = k*(theta-cir[i-1])*dt
            stochastic_term = sigma*np.sqrt(max(cir[i-1],0))*(w_path[i]-w_path[i-1])
            
            cir[i] = max(cir[i-1] + determistic_term + stochastic_term,0)
        self.__path = cir