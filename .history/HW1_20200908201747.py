import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv

from sklearn.model_selection import train_test_split
from sklearn import linear_model


class LinearRegression:
    def __init__(self):
        self.beta = None

    def __call__(self, X):
        if self.beta is not None:
            return X @ self.beta
        else:
            raise Exception
            
    def coef_(self):
        return self.beta

    def fit(self, X, y):
        self.beta = inv(X.transpose() @ X) @ X.transpose() @ y


class RidgeRegression:
    def __init__(self, l):
        self.beta = None
        self.l = l

    def __call__(self, X):
        if self.beta is not None:
            return X @ self.beta
        else:
            raise Exception

    def coef_(self):
        return self.beta
        
    def fit(self, X, y):
        self.beta = (
            inv(X.transpose() @ X + self.l * np.identity(X.shape[1])) @ X.transpose() @ y
        )

# Parsing input data
prostate_data = pd.read_csv("prostate.data", sep="\t", index_col=0)
X = prostate_data.drop(["train", "lpsa"], axis=1)
prostate_data.drop(["train"], axis=1, inplace=True)
column_names = prostate_data.columns
corr = prostate_data.corr() # Find correlation between features

prostate_data = prostate_data.to_numpy()
prostate_data = np.hstack((np.ones((len(prostate_data), 1)), prostate_data))

train, temp = train_test_split(prostate_data, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

# Remove the label (y)
X_test = train[:, :-1]
y_test = train[:, -1]

prostate_LR = LinearRegression()
prostate_LR.fit(X, y)

y_pred = prostate_LR(test[:, :-1])
mse = ((y_pred - test[:, -1]) ** 2).mean()
mpe = abs(y_pred - test[:, -1]).mean()

sigma_hat_squared = np.sum((y - y_pred) ** 2) / (X.shape[0] - X.shape[1] - 1)
std_error = np.sqrt(sigma_hat_squared * np.diag(inv(X.transpose() @ X)))
z_score = np.divide(prostate_LR.coef_(), std_error)
print(std_error)
print(z_score)


# find_beta_ridge = (
#     lambda l: inv(X.transpose() @ X + l * np.identity(X.shape[1])) @ X.transpose() @ y
# )
# mses = []
# beta_ridges = []
# ls = np.linspace(0.001, 10, 1000)
# for l in ls:
#     beta_ridge = find_beta_ridge(l)
#     beta_ridges.append(beta_ridge)
#     y_pred = val[:, :-1] @ beta_ridge
#     mses.append(((y_pred - val[:, -1]) ** 2).mean())
# plt.plot(ls, np.array(beta_ridges))
# plt.xlabel("Lambda")
# plt.ylabel("Coefficient")
# plt.legend(column_names)
# plt.show()

# idx = np.argmin(mses)
# print(f"Lambda: {ls[idx]}")
# print(f"Smallest MSE: {mses[idx]}")


# # Part C
# mses = []
# ts = []
# beta_lasso = []
# for l in ls:
#     clf = linear_model.Lasso(alpha=l)
#     clf.fit(X[:, 1:], y)
#     beta_lasso.append(clf.coef_)
#     y_pred = clf.predict(val[:, 1:-1])
#     ts.append(l / np.sum(np.abs(clf.coef_)))
#     mses.append(((y_pred - val[:, -1]) ** 2).mean())
# plt.plot(ts, np.array(beta_ridges))
# plt.xlabel("Shrinkage Factor")
# plt.ylabel("Coefficient")
# plt.legend(column_names)
# plt.show()

# # CV
# idx = np.argmin(mses)
# print(f"Shrinkage Factor: {ts[idx]}")
# print(f"Smallest MSE: {mses[idx]}")