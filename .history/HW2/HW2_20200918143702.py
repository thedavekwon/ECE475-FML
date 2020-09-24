import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pd.read_csv("SAheart.data")
df = df.replace('Present', 1).replace('Absent', 0)
# dataset.describe()
df = df.to_numpy()
pd.plotting.scatter_matrix(dataset, alpha=0.2)

# Regularize the prostate data to be 0 mean and unit variance
dataset = scale(dataset)

# Adding our Intercept column (column of ones) to the data
dataset = np.hstack((np.ones((len(dataset), 1)), dataset))

# Splitting our data into train (80%), val (10%), and test (10%)
train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)
