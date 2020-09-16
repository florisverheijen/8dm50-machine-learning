# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 00:48:23 2020

@author: Verheyen
"""


from scipy.integrate import quad

import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


mean = 17.298214285714284
std = 3.1859698608244296
x = np.linspace(6.839407377701363,26.856123868187574,100)

def gaussfunc(x):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value


res = quad(gaussfunc,-10,40)[0]