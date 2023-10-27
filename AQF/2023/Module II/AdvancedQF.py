# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:00:07 2020
Last update on Fri Oct 27 14:34:23 2023

@author: r.barone
"""


import numpy as np
import pandas as pd
import scipy.stats as si


class WienerProcess():
    """
    A Wiener Process class constructor
    """
    
    def __init__(self,maxTime = 1, steps = 1000, mc = 1, fix_seed = False):

        assert (type(maxTime)==float or type(maxTime)==int or maxTime > 0), "Expect a positive float number to define the time interval [0,maxTime]"
        
        self._maxTime = maxTime
        self._x0 = 0.
        self._path = None
        self._n_step = steps
        self._time_vector = None
        self._fix_seed = fix_seed
        self._seed = 1234
        self._mc=mc
        self._gen_path(self._maxTime, self._n_step)
    
    
    def steps(self):
        return self._n_step
    
    def maxTime(self):
        return self._maxTime
    
    def path(self):
        return self._path
    
    def time_vector(self):
        return self._time_vector
    
    def _set_time_vector(self, maxTime, n_step):
        #Creates an equally-spaced time vector
        self._time_vector = np.linspace(0,maxTime, n_step)
    
    def new_path(self, T = 1, n_step = 1000):
        self._gen_path(T, n_step)
        return self._path
    
    def _gen_path(self, T = 1, n_step = 1000):
        self._maxTime = T
        self._n_step = n_step
        self._set_time_vector(self._maxTime, self._n_step)
        if self._fix_seed:
            # Set seed for reproducibility
            np.random.seed(self._seed)
        w = np.random.randn(self._mc,n_step)
        w[:,0]=w[:,0]*self._x0
        for i in range(1,n_step):
            w[:,i] = w[:,i-1]+(w[:,i]/np.sqrt(n_step))
            
        
        self._path = w
 

class GeometricBM():
    """
    Class constructor for the Geometric Brownian Motion
    (used to describe the underlying dynamics in the Black-Scholes model)
    """
    
    def __init__(self, wp = WienerProcess(), x_0=10., mu = 0.03, sigma = 0.25, simulate_dynamics=False):
        assert (type(x_0)==float or type(x_0)==int or x_0 >0 ), "Expect a positive float number for the initial condition x_0"
        assert (sigma >= 0.), "Expect a non negative value for sigma"
        
        self._wp = wp
        self._x_0 = x_0
        self._mu = mu
        self._dyn_simu = simulate_dynamics
        self._sigma = sigma
        self._path = None
        
        self._gen_gbm_path(self._mu, self._sigma)
    
    def x_0(self):
        return self._x_0
    
    def mu(self):
        return self._mu
    
    def sigma(self):
        return self._sigma
    
    def wp(self):
        return self._wp
    
    def path(self):
        return self._path
    
    def new_path(self,x_0 = 0., mu = 0., sigma = 1.):
        self._x_0 = x_0
        self._mu = mu
        self._sigma = sigma
        self._gen_gbm_path(mu, sigma)
        return self._path
    
    def _gen_gbm_path(self, mu = 0., sigma = 1.):
        if self._dyn_simu:
            dS = np.ones((self._wp._mc,self._wp._n_step))*self._x_0
            dt = self._wp._maxTime / self._wp._n_step
            for i in range(1,self._wp._n_step):
                dS[:, i] = dS[:, i - 1] * (1.0 + self._mu * dt + self._sigma * (self._wp._path[:, i]-self._wp._path[:, i - 1]))
            self._path = dS
        else:
            time_vector = self._wp.time_vector()
            deterministic_term = (mu - (sigma**2/2))*time_vector
            stochastic_term = sigma*self._wp.path()
            self._path = self._x_0 *(np.exp(deterministic_term + stochastic_term))


class AsianCallOption(GeometricBM):
    def __init__ (self, wp = WienerProcess(), x_0=10., strike=9.5, mu = 0.03, sigma = 0.25,simulate_dynamics=False):
        super().__init__(wp, x_0, mu, sigma,simulate_dynamics)
        self._average = None
        self._strike = strike
        self._sigma_A = self._sigma_A()
        self._b = self._b()
        self._d_1 = self._d_1()
        self._d_2 = self._d_2()
    
    def _new_wp(self,wp):
        self._wp = wp
        self._gen_gbm_path(self._mu,self._sigma)
             
    def _b(self):
        return 0.5 * (self._mu - ((self._sigma_A)**2)/2)
    
    def _sigma_A (self):
        return (self._sigma / np.sqrt(3))
    
    def _d_1(self):
        l = np.log(self._x_0 / self._strike)
        T = self._wp.maxTime()
        m = (self._b - (self._sigma_A**2 )/2)*T
        return (l + m)/(self._sigma_A * np.sqrt(T))
    
    def _d_2(self):
        return self._d_1 - (np.sqrt(self._wp.maxTime())*self._sigma_A)
    
    def closed_formula_premium(self,show_results=False):
        T = self._wp.maxTime()
        premium_1 = self._x_0 * np.exp((self._b - self._mu)*T) * si.norm.cdf(self._d_1,0.,1.)
        premium_2 = self._strike * np.exp(-self._mu * T) * si.norm.cdf(self._d_2,0.,1.)
        Premium = premium_1 - premium_2
        if show_results:
            print('T\tS_0\tK\tmu\tb\td_1\tN(d_1)\td_2\tN(d_2)\tpremium_1\tpremium_2')
            print('{:.1f}\t{:.2f}\t{:.2f}\t{:.4f}\t'.format(T,self._x_0,self._strike,self._mu),end='')
            print('{:.4f}\t{:.4f}\t{:.4f}\t'.format(self._b,self._d_1,si.norm.cdf(self._d_1,0.,1.)),end='')
            print('{:.4f}\t{:.4f}\t{:.4f}\t\t{:.4f}\n'.format(self._d_2,si.norm.cdf(self._d_2,0.,1.),premium_1,premium_2))
        return Premium
       
    def _compute_average(self):
        p = self._path
        sqr=1/self._wp._n_step
        self._average = np.prod(np.power(p,sqr),axis=1)
        return self._average
       
    def mc_simulation(self,step = 1000, mc = 100000):
        wp = WienerProcess(1,step,mc)
        self._new_wp(wp)
        T = wp.maxTime()
        avg = self._compute_average()
        payoff = np.maximum(avg-self._strike,0.)
        simulation = pd.DataFrame(avg,columns=['Und_Average'])
        simulation['Payoff'] = payoff
        simulation['MC_Premium'] = np.exp(-self._mu*T)*simulation.Payoff.expanding().mean()
        simulation['MC_Error'] = np.exp(-2 * self._mu * T) * simulation['Payoff']**2
        simulation['MC_Error'] = (simulation['MC_Error'].expanding().mean() - simulation['MC_Premium']**2)/np.sqrt(simulation.index + 1)
        simulation['Simulation_Error'] = simulation['MC_Premium']-self.closed_formula_premium(False)
        return simulation

class EuropeanCallOption(GeometricBM):
    def __init__ (self, wp = WienerProcess(), x_0=10., strike=9.5, mu = 0.03, sigma = 0.25,simulate_dynamics=False):
        super().__init__(wp, x_0, mu, sigma,simulate_dynamics)
        self._strike = strike
        self._sigma = sigma
        self._mu = mu
        self._d_1 = self._d_1()
        self._d_2 = self._d_2()
   
    def mc_simulation(self,step = 1000, mc = 100000):
        wp = WienerProcess(1,step,mc)
        self._new_wp(wp)
        T = wp.maxTime()
        S_T = self._path[:,-1]
        payoff = np.maximum(S_T-self._strike,0.)
        simulation = pd.DataFrame(S_T,columns=['Und_Value'])
        simulation['Payoff'] = payoff
        simulation['MC_Premium'] = np.exp(-self._mu*T)*simulation.Payoff.expanding().mean()
        simulation['MC_Error'] = np.exp(-2 * self._mu * T) * simulation['Payoff']**2
        simulation['MC_Error'] = (simulation['MC_Error'].expanding().mean() - simulation['MC_Premium']**2)/np.sqrt(simulation.index + 1)
        simulation['Simulation_Error'] = simulation['MC_Premium']-self.closed_formula_premium(False)
        return simulation
   
    def _new_wp(self,wp):
        self._wp = wp
        self._gen_gbm_path(self._mu,self._sigma)
    
    def _d_1(self):
        l = np.log(self._x_0 / self._strike)
        T = self._wp.maxTime()
        m = (self._mu + (self._sigma**2 )/2)*T
        return (l + m)/(self._sigma * np.sqrt(T))
    
    def _d_2(self):
        return self._d_1 - (np.sqrt(self._wp.maxTime())*self._sigma)
    
    def closed_formula_premium(self,show_results=False):
        T = self._wp.maxTime()
        premium_1 = self._x_0 * si.norm.cdf(self._d_1,0.,1.)
        premium_2 = self._strike * np.exp(-self._mu * T) * si.norm.cdf(self._d_2,0.,1.)
        Premium = premium_1 - premium_2
        if show_results:
            print('T\tS_0\tK\tmu\tsigma\td_1\tN(d_1)\td_2\tN(d_2)\tpremium_1\tpremium_2')
            print('{:.1f}\t{:.2f}\t{:.2f}\t{:.4f}\t'.format(T,self._x_0,self._strike,self._mu),end='')
            print('{:.4f}\t{:.4f}\t{:.4f}\t'.format(self._sigma,self._d_1,si.norm.cdf(self._d_1,0.,1.)),end='')
            print('{:.4f}\t{:.4f}\t{:.4f}\t\t{:.4f}\n'.format(self._d_2,si.norm.cdf(self._d_2,0.,1.),premium_1,premium_2))
        return Premium
