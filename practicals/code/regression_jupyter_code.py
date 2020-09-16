# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 12:59:33 2020

@author: s165899
"""
from sklearn.datasets import load_diabetes

# Load data
diabetes = load_diabetes()

# split data into train and test and use all features
# _d stands for diabetes 
nr_samples_d = diabetes.data.shape[0]
train_frac_d = 2/3
cutoff_d = round(train_frac*nr_samples)       
X_train_d = diabetes.data[:cutoff_d, :]
y_train_d = diabetes.target[:cutoff_d, np.newaxis]
X_test_d = diabetes.data[cutoff_d:, :]
y_test_d = diabetes.target[cutoff_d:, np.newaxis]

# normalize data
X_train_n_d = norm(X_train_d)
X_test_n_d = norm(X_test_d)

# Calculate mean square error and accuracy for nr_k nearest neighbors
y_pred_k = np.zeros((X_test_d.shape[0],nr_k))
MSE_d = np.zeros((nr_k))
acc_d = np.zeros((nr_k))
y_pred_k_reg = np.zeros((X_test_d.shape[0],nr_k))
MSE_d_reg = np.zeros((nr_k))
acc_d_reg = np.zeros((nr_k))
for k in range(1,nr_k+1):
    for test_sample in range(X_test_d.shape[0]):
        # kNN classification:
        y_pred_k[test_sample,k-1] = pred(X_train_n_d, y_train_d, X_test_n_d[test_sample], k)
        # kNN regression:
        y_pred_k_reg[test_sample,k-1] = pred_regression(X_train_n_d, y_train_d, X_test_n_d[test_sample], k)        
    # classification:
    y_pred_this_k = y_pred_k[:,k-1]
    MSE_d[k-1] = mse(y_test_d,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))  
    acc_d[k-1] = accuracy(y_test_d,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))
    # regression
    y_pred_this_k_reg = y_pred_k_reg[:,k-1]
    MSE_d_reg[k-1] = mse(y_test_d,y_pred_this_k_reg.reshape((y_pred_this_k_reg.shape[0],1)))  
    acc_d_reg[k-1] = accuracy(y_test_d,y_pred_this_k_reg.reshape((y_pred_this_k_reg.shape[0],1)))

# Plot in figure
fig = plt.subplots()
plt.suptitle("Classification", fontsize=14)
plt.subplot(1,2,1)
plt.plot(np.arange(1,nr_k+1),MSE_d,marker='o')
plt.xlabel('k')
plt.ylabel('Mean squared error')
plt.subplot(1,2,2)
plt.plot(np.arange(1,nr_k+1),acc_d,marker='*')
plt.xlabel('k')
plt.ylabel('Accuracy')

# Plot in figure
fig = plt.subplots()
plt.suptitle("Regression", fontsize=14)
plt.subplot(1,2,1)
plt.plot(np.arange(1,nr_k+1),MSE_d_reg,marker='o')
plt.xlabel('k')
plt.ylabel('Mean squared error')
plt.subplot(1,2,2)
plt.plot(np.arange(1,nr_k+1),acc_d_reg,marker='*')
plt.xlabel('k')
plt.ylabel('Accuracy')
