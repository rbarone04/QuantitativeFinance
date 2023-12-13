# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 17:21:13 2023

@author: r.barone
"""

import numpy as np
import scipy.stats as si

class Option():
    def __init__(self,Spot, Strike, Time, rate, vola, option_type = "call"):
        #S: underlying spot price
        #K: strike price
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        self.S = Spot
        self.K = Strike
        self.T = Time
        self.r = rate
        self.s = vola
        #omega: option sign (+1 for call, -1 for put)
        if option_type.lower() == "call":
            self.omega = 1.
        elif option_type.lower() == "put":
            self.omega = -1.
        else:
            print("Error in european_option: invalid option_type "+ str(option_type)+". Set as Call option by default")
            self.omega = 1.
            
    def price(self):
        if np.abs(self.T)<1e-5:
            return np.maximum(self.omega*(self.S-self.K),0.)

        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))
        d2 = (np.log(self.S / self.K) + (self.r - 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))

        premium =(self.omega * self.S * si.norm.cdf(self.omega * d1,0.0,1.0)) - (self.omega * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.omega * d2,0.0,1.0))

        return premium
    
    def price_complete(self,S, K, T, r, s):
        #S: underlying spot price
        #K: strike price
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        if np.abs(T)<1e-5:
            return np.maximum(self.omega*(S-K),0.)

        d1 = (np.log(S / K) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
        d2 = (np.log(S / K) + (r - 0.5 * s ** 2) * T) / (s * np.sqrt(T))

        premium =(self.omega * S * si.norm.cdf(self.omega * d1,0.0,1.0)) - (self.omega * K * np.exp(-r * T) * si.norm.cdf(self.omega * d2,0.0,1.0))

        return premium
    
    def price_S(self,spot):
        if np.abs(self.T)<1e-5:
            return np.maximum(self.omega*(spot-self.K),0.)

        d1 = (np.log(spot / self.K) + (self.r + 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))
        d2 = (np.log(spot / self.K) + (self.r - 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))

        premium =(self.omega * spot * si.norm.cdf(self.omega * d1,0.0,1.0)) - (self.omega * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.omega * d2,0.0,1.0))

        return premium
    
    def delta(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))
        if self.omega == 1.:
            return si.norm.cdf(d1,0.0,1.0)
        else:
            return (si.norm.cdf(d1,0.0,1.0)-1)
        
    def gamma(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.s ** 2) * self.T) / (self.s * np.sqrt(self.T))
        return (si.norm.pdf(d1,0.0,1.0) / (self.S*self.s*np.sqrt(self.T)))
        
class BullSpreadOptions():
    def __init__(self, Spot,Strike1,Strike2,Time,rate,vola):
        #S: underlying spot price
        #K1,K2: strike prices
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        self.S = Spot
        if Strike1 >= Strike2:
            self.K1 = Strike2
            self.K2 = Strike1
        else:
            self.K1 = Strike1
            self.K2 = Strike2
        self.T = Time
        self.r = rate
        self.s = vola
        self.opt1 = Option(self.S,self.K1,self.T,self.r,self.s,'call')
        self.opt2 = Option(self.S,self.K2,self.T,self.r,self.s,'call')
    
    def price(self):
        return self.opt1.price() - self.opt2.price()
    
    def price_S(self, spot):
        return self.opt1.price_S(spot) - self.opt2.price_S(spot)
    
    def price_complete(self,S, K1, K2, T, r, s):
        return self.opt1.price_complete(S, K1, T, r, s) - self.opt2.price_complete(S, K2, T, r, s)


class BearSpreadOptions():
    def __init__(self, Spot,Strike1,Strike2,Time,rate,vola):
        #S: underlying spot price
        #K1,K2: strike prices
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        self.S = Spot
        if Strike1 >= Strike2:
            self.K1 = Strike2
            self.K2 = Strike1
        else:
            self.K1 = Strike1
            self.K2 = Strike2
        self.T = Time
        self.r = rate
        self.s = vola
        self.opt1 = Option(self.S,self.K2,self.T,self.r,self.s,'put')
        self.opt2 = Option(self.S,self.K1,self.T,self.r,self.s,'put')
    
    def price(self):
        return self.opt1.price() - self.opt2.price()
    
    def price_S(self, spot):
        return self.opt1.price_S(spot) - self.opt2.price_S(spot)
    
    def price_complete(self,S, K1, K2, T, r, s):
        return self.opt1.price_complete(S, K2, T, r, s) - self.opt2.price_complete(S, K1, T, r, s)
        
class ButflySpreadOptions():
    def __init__(self, Spot,Strike1,Strike2,Strike3,Time,rate,vola):
        #S: underlying spot price
        #K1,K2,K3: strike prices
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        self.S = Spot
        self.K1 = Strike1
        self.K2 = Strike2
        self.K3 = Strike3
        self.T = Time
        self.r = rate
        self.s = vola
        self.opt1 = Option(self.S,self.K1,self.T,self.r,self.s,'call')
        self.opt2 = Option(self.S,self.K2,self.T,self.r,self.s,'call')
        self.opt3 = Option(self.S,self.K3,self.T,self.r,self.s,'call')
    
    def price(self):
        return self.opt1.price() - 2*self.opt2.price() + self.opt3.price()
    
    def price_S(self, spot):
        return self.opt1.price_S(spot) - 2*self.opt2.price_S(spot)+self.opt3.price_S(spot)
    
    def price_complete(self,S, K1, K2, K3, T, r, s):
        return self.opt1.price_complete(S, K1, T, r, s) - 2*self.opt2.price_complete(S, K2, T, r, s) + self.opt3.price_complete(S, K3, T, r, s)
    
class StrangleOptions():
    def __init__(self, Spot,Strike1, Strike2, Time,rate,vola):
        #S: underlying spot price
        #K1, K2: strike prices
        #T: time to maturity
        #r: risk-free interest rate
        #s: volatility of underlying asset

        self.S = Spot
        self.K1 = Strike1
        self.K2 = Strike2
        self.T = Time
        self.r = rate
        self.s = vola
        self.opt1 = Option(self.S,self.K1,self.T,self.r,self.s,'put')
        self.opt2 = Option(self.S,self.K2,self.T,self.r,self.s,'call')
    
    def price(self):
        return self.opt1.price() + self.opt2.price()
    
    def price_S(self, spot):
        return self.opt1.price_S(spot) +self.opt2.price_S(spot)
    
    def price_complete(self,S, K1, K2, T, r, s):
        return self.opt1.price_complete(S, K1, T, r, s) + self.opt2.price_complete(S, K2, T, r, s)
    
class StraddleOptions():
    def __init__(self, Spot,Strike, Time,rate,vola):
       
        self.opt = StrangleOptions(Spot, Strike, Strike, Time, rate, vola)
    
    def price(self):
        return self.opt.price()
    
    def price_S(self, spot):
        return self.opt.price_S(spot)
    
    def price_complete(self,S, K, T, r, s):
        return self.opt.price_complete(S, K, K, T, r, s) 

