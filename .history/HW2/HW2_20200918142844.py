import pandas as pd

dataset = pd.read_csv('SAheart.data')
# dataset.describe()
X = dataset.drop(['chd'])
y = dataset[:,-1]
print(y)
