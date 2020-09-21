"""
Functions to calculate the mean squared error (mse) and accuracy to determine the
performance of a classifier.

Group 2
"""
import numpy as np 

def mse(y_true, y_pred):
    mse = np.average((y_true - y_pred) ** 2,axis=0,weights=None)
    return mse

def accuracy(y_true, y_pred):
    nr_samples = y_true.shape[0]
    # when abs(y_test - y_pred) returns 1, the prediction is false (negative or positive)
    # the sum of this is the number of false negatives and positives:
    FNandFP = sum(abs(y_true - y_pred)) 
    TNandTP = nr_samples - FNandFP
    acc = TNandTP / nr_samples
    return acc