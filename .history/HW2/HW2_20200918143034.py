import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('SAheart.data')
# dataset.describe()
X = dataset.drop(['chd'], axis=1)
y = dataset['chd']

