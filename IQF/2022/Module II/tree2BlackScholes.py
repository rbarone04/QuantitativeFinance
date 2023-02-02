# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:15:40 2020

@author: u0i3016
"""

import numpy as np
import time
import scipy.stats as si
import csv

def european_option(S, K, T, r, s, option_type = "call"):
    #S: underlying spot price
    #K: strike price
    #T: time to maturity
    #r: risk-free interest rate
    #s: volatility of underlying asset

    #omega: option sign (+1 for call, -1 for put)
    omega = 0
    if option_type == "call":
        omega = 1.
    elif option_type == "put":
        omega = -1.
    else:
        print("Error in european_option: invalid option_type "+ str(option_type))
        return 0

    if np.abs(T)<1e-5:
        return np.maximum(omega*(S-K),0.)

    d1 = (np.log(S / K) + (r + 0.5 * s ** 2) * T) / (s * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * s ** 2) * T) / (s * np.sqrt(T))

    premium =(omega * S * si.norm.cdf(omega * d1,0.0,1.0)) - (omega * K * np.exp(-r * T) * si.norm.cdf(omega * d2,0.0,1.0))

    return premium

def binomial_tree(iteration, time, spot_price, strike, sigma, rate, option_type = 'call', array_out=False):
    dt=time/iteration
    uptick = np.exp(sigma*np.sqrt(dt))
    downtick = 1/uptick
    p = (np.exp(rate*dt)-downtick)/(uptick -downtick)
    price_tree = np.zeros([iteration+1, iteration+1])

    for i in range(iteration+1):
        for j in range (i+1):
            price_tree[j,i]= spot_price*(downtick**j)*(uptick**(i-j))

    option = np.zeros([iteration+1, iteration+1])
    opt_payoff = np.zeros
    if (option_type == 'call'):
           option[:,iteration] = np.maximum(np.zeros(iteration+1), price_tree[:,iteration]-strike)
    else:
        option[:,iteration] = np.maximum(np.zeros(iteration+1), strike - price_tree[:,iteration])

    for i in np.arange(iteration-1,-1,-1):
        for j in np.arange(0,i+1):
            option[j,i] = np.exp(-rate*dt)*(p*option[j,i+1]+(1-p)*option[j+1,i+1])
    if array_out:
        return [option[0,0],price_tree, option]
    else:
        return option[0,0]


with open('tree2BlackScholes.csv', mode='w',newline='') as csvFile:
    bs_record = csv.writer(csvFile, delimiter=';',quotechar='"',quoting=csv.QUOTE_MINIMAL)
    for i in np.arange(1,1000):
        bs_formula = european_option(95,80,1,0.02,0.09929,'call')
        start= time.time()
        tree_formula = binomial_tree(i,1,95,80,0.09929,0.02,'call')
        elapsed=time.time() - start
        bs_record.writerow([i,bs_formula, tree_formula, elapsed])
        print(i,end='\t')
        print(bs_formula, end='\t')
        print(tree_formula, end='\t')
        print(elapsed)