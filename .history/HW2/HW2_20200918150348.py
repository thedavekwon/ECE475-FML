import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

def sigmoid(theta, x):
    return 1/(1 + np.exp(-theta.T@x)) 

    
class LogisticRegression:
    def __init__(self, step):
        self.theta = None
        self.step = step
        
    def fit(self, X, y):
        if self.theta is None:
            self.theta = np.random.rand(X.shape[1], 1)
        for i in range(len(X)):
            for j in range(len(self.theta)):
                self.theta[j] = self.theta[j] + self.step * (y[i] - sigmoid(self.theta, X[i, :]))*X[i][j]
    
        
        
df = pd.read_csv("SAheart.data")
df = df.drop(['row.names'], axis=1).replace('Present', 1).replace('Absent', 0)

# scatter matrix plot
X = df.drop(['chd', 'adiposity', 'typea'], axis=1)  
pd.plotting.scatter_matrix(X, c= df['chd'])
plt.show()

# dataset.describe()
dataset = df.to_numpy()

# Regularize the prostate data to be 0 mean and unit variance
dataset = scale(dataset)

# Adding our Intercept column (column of ones) to the data
dataset = np.hstack((np.ones((len(dataset), 1)), dataset))

# Splitting our data into train (80%), val (10%), and test (10%)
train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)


