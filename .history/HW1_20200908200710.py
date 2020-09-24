import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv

from sklearn.model_selection import train_test_split
from sklearn import linear_model

class linear_regression():
    def __init__(self):
        self.beta = None
        
    def fit(self, X, y):
        if self.beta:
           self.beta = inv(X.transpose() @ X) @ X.transpose() @ y
        else:
            raise Exception
            
    def __call__(self, X):
        return X @ self.b
    

prostate_lr = new linear_regression()


prostate_data = pd.read_csv("prostate.data", sep="\t", index_col=0)

X = prostate_data.drop(["train", "lpsa"], axis=1)
prostate_data.drop(["train"], axis=1, inplace=True)

# corr = prostate_data.corr()
column_names = prostate_data.columns

prostate_data = prostate_data.to_numpy()
prostate_data = np.hstack((np.ones((len(prostate_data), 1)), prostate_data))

train, temp = train_test_split(prostate_data, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

# remove the label (y)
X = train[:, :-1]
y = train[:, -1]

beta_hat = inv(X.transpose() @ X) @ X.transpose() @ y

y_pred = test[:, :-1] @ beta_hat
mse = ((y_pred - test[:, -1]) ** 2).mean()
mpe = abs(y_pred - test[:,-1]).mean()

sigma_hat_squared = np.sum((y - y_pred ) ** 2) / (X.shape[0]-X.shape[1]-1)
std_error = np.sqrt(sigma_hat_squared * np.diag(inv(X.transpose() @ X)))
z_score = np.divide(beta_hat, std_error)
print(std_error)
print(z_score)


find_beta_ridge = lambda l : inv(X.transpose() @ X + l * np.identity(X.shape[1]))@X.transpose()@y
mses = []
beta_ridges = []
ls = np.linspace(0.001, 10, 1000)
for l in ls:
    beta_ridge = find_beta_ridge(l)
    beta_ridges.append(beta_ridge)
    y_pred = val[:, :-1] @ beta_ridge
    mses.append(((y_pred - val[:, -1]) ** 2).mean())
plt.plot(ls, np.array(beta_ridges))
plt.xlabel("Lambda")
plt.ylabel("Coefficient")
plt.legend(column_names)
plt.show()

idx = np.argmin(mses)
print(f"Lambda: {ls[idx]}")
print(f"Smallest MSE: {mses[idx]}")

    
# part c
mses = []
ts = []
beta_lasso = []
for l in ls:
    clf = linear_model.Lasso(alpha=l)
    clf.fit(X[:, 1:], y)
    beta_lasso.append(clf.coef_)
    y_pred = clf.predict(val[:, 1:-1]) 
    ts.append(l/np.sum(np.abs(clf.coef_)))
    mses.append(((y_pred - val[:, -1]) ** 2).mean())
plt.plot(ts, np.array(beta_ridges))
plt.xlabel("Shrinkage Factor")
plt.ylabel("Coefficient")
plt.legend(column_names)
plt.show()

# CV
idx = np.argmin(mses)
print(f"Shrinkage Factor: {ts[idx]}")
print(f"Smallest MSE: {mses[idx]}")