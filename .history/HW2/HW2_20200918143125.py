import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('SAheart.data')
# dataset.describe()
dataset = dataset.to_numpy()



