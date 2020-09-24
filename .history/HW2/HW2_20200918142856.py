import pandas as pd

dataset = pd.read_csv('SAheart.data')
# dataset.describe()
X = dataset.drop(['chd'], axis=1)
y = dataset[:,-1]
print(y)
