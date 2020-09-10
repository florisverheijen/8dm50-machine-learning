import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer

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
    return prediction #prediction.reshape((prediction.shape[0],1))

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
    
# Load data
breast_cancer = load_breast_cancer()

""" randomly selecting the train samples still has to be implemented !!""" 
# split data into train and test and use all features
nr_samples = breast_cancer.data.shape[0]
train_frac = 2/3
cutoff = round(train_frac*nr_samples)       
X_train = breast_cancer.data[:cutoff, :]
y_train = breast_cancer.target[:cutoff, np.newaxis]
X_test = breast_cancer.data[cutoff:, :]
y_test = breast_cancer.target[cutoff:, np.newaxis]

# normalize data
X_train_n = norm(X_train)
X_test_n = norm(X_test)

# Calculate mean square error and accuracy for nr_k nearest neighbors
nr_k = 40
y_pred_k = np.zeros((X_test.shape[0],nr_k))
MSE = np.zeros((nr_k))
acc = np.zeros((nr_k))
for k in range(1,nr_k+1):
    for test_sample in range(X_test.shape[0]):
        y_pred_k[test_sample,k-1] = pred(X_train_n, y_train, X_test_n[test_sample], k)
    y_pred_this_k = y_pred_k[:,k-1]
    # reshape y_pred_this_k because it has another shape than y_test
    MSE[k-1] = mse(y_test,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))  
    acc[k-1] = accuracy(y_test,y_pred_this_k.reshape((y_pred_this_k.shape[0],1)))


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