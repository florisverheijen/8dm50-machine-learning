# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 10:04:00 2020

@author: Verheyen
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from scipy.stats import norm
from scipy.integrate import quad
import scipy.stats


breast_cancer = load_breast_cancer()
nr_samples = breast_cancer.data.shape[0]
train_frac = 2/3
cutoff = round(train_frac*nr_samples)       
X_train = breast_cancer.data[:cutoff, :]
y_train = breast_cancer.target[:cutoff, np.newaxis]
X_test = breast_cancer.data[cutoff:, :]
y_test = breast_cancer.target[cutoff:, np.newaxis]


Data = np.append(X_train,y_train,axis=1)
Sorted_data = Data[np.argsort(Data[:,30])]
locatie = np.where(Sorted_data[:,30] == 1)

negData = Sorted_data[0:locatie[0][0],:]
posData = Sorted_data[locatie[0][0]:len(Sorted_data),:]

mean1=0
std1=0
mean2 = 0
std2=0


def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])





def gauss_func_side(Xaxis):
    value = scipy.stats.norm.pdf(Xaxis,mean1,std1)
    return value

def gauss_func_mid(Xaxis):
    value = scipy.stats.norm.pdf(Xaxis,mean2,std2)
    return value



fig,ax = plt.subplots(5,6 ,figsize = (50,40))

for j in range(5):
    for k in range(6):
        i = 1 
        mean_pos = np.mean(posData[:,i])
        std_pos = np.std(posData[:,i])
        mean_neg = np.mean(negData[:,i])
        std_neg = np.std(negData[:,i])
        Xmin = min(mean_pos - 3*std_pos,mean_neg - 3*std_neg)
        Xmax = max(mean_pos + 3*std_pos,mean_neg + 3*std_neg)
        Xaxis = np.linspace(Xmin,Xmax,200)
        
        pos_gauss = norm.pdf(Xaxis,mean_pos,std_pos)
        neg_gauss = norm.pdf(Xaxis,mean_neg,std_neg)
        
        #get intersect:
        r = solve(mean_pos, mean_neg, std_pos, std_neg)
        
        
       
        if r[0]>r[1]: # makes sure the smallest value of x is in r[0]
            a = r[0]
            r[0] = r[1]
            r[1] = a
            
            
        if r[0] < Xaxis[0]:
            
            if neg_gauss[0]>pos_gauss[0]: 
                mean1 = mean_pos
                std1 = std_pos
                mean2 = mean_neg
                std2 = std_neg
                
            else:
                mean1 = mean_neg
                std1 = std_neg
                mean2 = mean_pos
                std2 = std_pos
        
        Area1 = quad(gauss_func_side,-np.inf,r[0])[0]
        Area2 = quad(gauss_func_mid,r[0],r[1])[0]
        Area3 = quad(gauss_func_side,r[1],np.inf)[0]
        tot_Area = quad(gauss_func_side,-np.inf,r[0])[0] + quad(gauss_func_mid,r[0],r[1])[0] + quad(gauss_func_side,r[1],np.inf)[0] 
       
        
       
        print(tot_Area)
        
        
        ax[j,k].plot(Xaxis,pos_gauss,'r')
        ax[j,k].fill_between(Xaxis,pos_gauss,color = 'red',alpha = 0.2)
        ax[j,k].plot(Xaxis,neg_gauss,'b')
        ax[j,k].fill_between(Xaxis,neg_gauss,color = 'blue',alpha = 0.2)       
        ax[j,k].set_title(f'feature {i+1}',fontsize = 20)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        