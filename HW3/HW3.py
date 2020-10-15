import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

# generate N = 50 samples in two equal-sized classes with p = 5000 predictors that are independent of the class labels 
data = np.hstack((np.random.rand(50, 5000), np.concatenate((np.ones((25, 1)), np.zeros((25, 1))), axis=0)))

# wrong way to do cross-validation
# step 1: screen predictors (choose a subset of "good" predictors)
corr = np.corrcoef(data.T)[-1,:-1]
abs_corr = np.abs(corr)
idx = abs_corr.argsort()[-100:]

plt.hist(corr[idx], bins=10)
plt.show()

# shuffle the data
X = data[:, idx]
y = data[:, -1]
X, y = shuffle(X, y)

# step 2/3: cross validate to estimate the unknown tuning parameter n (the optimal number of nearnest neighbors)
accuracies = []
for n in range(1, 10):
    # split data using k fold
    k = 5
    kf = KFold(n_splits=k)
    accuracy = 0
    for train_idx, test_idx in kf.split(X,y):
        KNN = KNeighborsClassifier(n_neighbors=n)
        KNN.fit(X[train_idx], y[train_idx])
        y_pred = KNN.predict(X[test_idx])
        accuracy = accuracy + np.sum(y_pred == y[test_idx])/len(y_pred)
    accuracies.append(accuracy/k)     
print(accuracies)


# right way to do cross-validation
# step 1: divide the samples into K cross-validation folds at random
k = 5
kf = KFold(n_splits=k)
accuracies = [0.0]*11
# step 2: for each fold k:
data = shuffle(data)
for train_idx, test_idx in kf.split(data):
    # a) screen predictors (choose a subset of "good" predictors)
    corr = np.corrcoef(data[train_idx,:].T)[-1,:-1]
    abs_corr = np.abs(corr)
    idx = abs_corr.argsort()[-100:]
    plt.hist(corr[idx], bins=10)
    plt.show()
    accuracy = 0
    # b)/c) build classifier and cross validate
    for n in range(1, 10):
        KNN = KNeighborsClassifier(n_neighbors=n)
        KNN.fit(data[:, idx][train_idx], data[train_idx, -1])
        y_pred = KNN.predict(data[:, idx][test_idx])
        accuracies[n] += np.sum(y_pred == data[test_idx, -1])/len(y_pred)
print(np.array(accuracies)/k)