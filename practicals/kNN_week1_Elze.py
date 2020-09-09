import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from math import sqrt
import sys
from sklearn.metrics import mean_squared_error, accuracy_score

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
	return sqrt(distance)

# Get the k nearest neighbors and the values of their features
def get_kNN(traindata, data_point, k):
    distances = list()
    for p1 in range(traindata.shape[0]):
        dist = eucl_dist(traindata[p1],data_point)
        distances.append((p1, dist))
    distances.sort(key=lambda tup:tup[1]) # sort distance (second column)
    neighbors = list()
    for n in range(k):
        neighbors.append(distances[n][0])
    return neighbors, traindata[neighbors] # returns the values of the features of the k closest neighbors

# Make predictions
def pred(traindata, traintargets, testdata, k):
    NN, NN_data = get_kNN(traindata, testdata, k)
    y_NN = [traintargets.item(i) for i in NN]
    prediction = max(set(y_NN), key=y_NN.count)
    return prediction

# Load data
breast_cancer = load_breast_cancer()

# split data into train and test and use all features
nr_samples = breast_cancer.data.shape[0]
train_frac = 2/3
cutoff = round(train_frac*nr_samples)       
X_train = breast_cancer.data[:cutoff, :]
y_train = breast_cancer.target[:cutoff, np.newaxis]
X_test = breast_cancer.data[cutoff:, :]
y_test = breast_cancer.target[cutoff:, np.newaxis]
# or let it select a particular amount of train samples at random?

k = 3 # number of neighbors you want to include

X_train_n = norm(X_train)
X_test_n = norm(X_test)

y_pred = np.zeros((X_test.shape[0]))    # create empty array
y_pred_n = np.zeros((X_test.shape[0]))  # create empty array
for test_sample in range(X_test.shape[0]):
    # fill array with predicted labels
    y_pred[test_sample] = pred(X_train, y_train, X_test[test_sample], k) 
    y_pred_n[test_sample] = pred(X_train_n, y_train, X_test_n[test_sample], k) 
    #y_pred = y_pred.reshape((y_pred.shape[0],1))   leads to the same results as without reshaping
    #y_pred_n = y_pred_n.reshape((y_pred_n.shape[0],1))
    
print("Mean squared error with normalization for k = %i : %.2f" % (k,mean_squared_error(y_test, y_pred_n)))
print("Mean squared error without normalization for k = %i : %.2f" % (k,mean_squared_error(y_test, y_pred)))
print("Accuracy with normalization for k = %i : %.2f" % (k,accuracy_score(y_test, y_pred_n)))
print("Accuracy without normalization for k = %i : %.2f" % (k,accuracy_score(y_test, y_pred)))

nr_k = 40
y_pred_k = np.zeros((X_test.shape[0],nr_k))
MSE = np.zeros((nr_k))
acc = np.zeros((nr_k))
for k in range(1,nr_k+1):
    for test_sample in range(X_test.shape[0]):
        y_pred_k[test_sample,k-1] = pred(X_train_n, y_train, X_test_n[test_sample], k)
    MSE[k-1] = mean_squared_error(y_test,y_pred_k[:,k-1])
    acc[k-1] = accuracy_score(y_test,y_pred_k[:,k-1])

# controle: is the same
MSE2 = np.zeros((nr_k))
acc2 = np.zeros((nr_k))
for k in range(0,nr_k):
    y_pred_this_k = y_pred_k[:,k]
    MSE2[k] = mean_squared_error(y_test,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))
    acc2[k] = accuracy_score(y_test,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))


# Plot in figure
fig = plt.subplots()
plt.subplot(1,2,1)
plt.plot(np.arange(1,nr_k+1),MSE,marker='o')
plt.xlabel('k')
plt.ylabel('Mean squared error')
plt.subplot(1,2,2)
plt.plot(np.arange(1,nr_k+1),acc,marker='*')
plt.xlabel('k')
plt.ylabel('Accuracy')