# -*- coding: utf-8 -*-
"""
Functions to make a kNN classifier. 

Group 2
"""
import numpy as np

# Normalize data
def norm(data):
    data_normalized = np.zeros(data.shape)
    for col in range(data.shape[1]):
        x = data[:,col]
        # formula to derive normalized values of each feature
        data_normalized[:,col] = (x - min(x)) / (max(x) - min(x))
    return data_normalized

# Calculate Euclidean Distance
def eucl_dist(row1, row2):
    # take each feature as one dimension so loop over all dimensions
	distance = sum(np.subtract(row1,row2)**2)  # use L2 norm
	return distance**(0.5)

def manh_dist(row1,row2):
    distance = sum(abs(np.subtract(row1,row2)))
    return distance

# Get the k nearest neighbors and the values of their features
def get_kNN(traindata, data_point, k):
    distances = list()
    for p1 in range(traindata.shape[0]):
        #dist = eucl_dist(traindata[p1],data_point)        
        dist = manh_dist(traindata[p1],data_point)
        distances.append((p1, dist))
    distances.sort(key=lambda tup:tup[1]) # sort distance (second column)
    neighbors = list()
    for n in range(k):
        neighbors.append(distances[n][0])
    return neighbors, traindata[neighbors] # returns the values of the features of the k closest neighbors

# Make predictions for classification
def pred(traindata, traintargets, testdata, k):
    NN, NN_data = get_kNN(traindata, testdata, k)
    y_NN = [traintargets.item(i) for i in NN]
    prediction = max(set(y_NN), key=y_NN.count)
    return prediction

# Make predictions for regression 
def pred_regression(traindata, traintargets, testdata, k):
    NN, NN_data = get_kNN(traindata, testdata, k)
    y_NN = [traintargets.item(i) for i in NN]
    prediction = np.average(y_NN)
    return prediction