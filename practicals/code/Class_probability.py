# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:28:44 2020

@author: s152040
"""
import numpy as np
import scipy

def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])

def graphselect1(neg_gauss,pos_gauss,mean_pos,std_pos,mean_neg,std_neg):
    if neg_gauss[0] < pos_gauss[0]: 
        mean1 = mean_neg
        std1 = std_neg
        mean2 = mean_pos
        std2 = std_pos
            
    else:
        mean1 = mean_pos
        std1 = std_pos
        mean2 = mean_neg
        std2 = std_neg
        
    return(mean1,std1,mean2,std2)

def gauss_func_1(Xaxis):
    
    value = scipy.stats.norm.pdf(Xaxis,mean1,std1)
    return value

def gauss_func_2(Xaxis):
    value = scipy.stats.norm.pdf(Xaxis,mean2,std2)
    return value


