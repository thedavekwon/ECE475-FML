import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('SAheart.data')
# dataset.describe()
dataset = dataset.to_numpy()

# Regularize the prostate data to be 0 mean and unit variance
dataset = scale(dataset)

# Adding our Intercept column (column of ones) to the data
prostate_data = np.hstack((np.ones((len(prostate_data), 1)), prostate_data))

# Splitting our data into train (80%), val (10%), and test (10%)
train, temp = train_test_split(prostate_data, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)



