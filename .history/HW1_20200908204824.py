import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv, pinv

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
        self.beta = inv(X.T @ X) @ X.T @ y


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
        self.beta = inv(X.T @ X + self.l * np.eye(X.shape[1])) @ X.T @ y


# Parsing input data
prostate_data = pd.read_csv("prostate.data", sep="\t", index_col=0)
prostate_data.drop(["train"], axis=1, inplace=True)
column_names = prostate_data.columns
corr = prostate_data.corr() # Find correlation between features

prostate_data = prostate_data.to_numpy()
prostate_data = np.hstack((np.ones((len(prostate_data), 1)), prostate_data))
train, temp = train_test_split(prostate_data, test_size=0.2)
test, val = train_test_split(temp, test_size=0.5)

# train = prostate_data[prostate_data["train"] == "T"].drop(["train"], axis=1).to_numpy()
# train = np.hstack((np.ones((len(train), 1)), train))
# test = prostate_data[prostate_data["train"] == "F"].drop(["train"], axis=1).to_numpy()
# test = np.hstack((np.ones((len(test), 1)), test))

# split X and y
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]

prostate_LR = LinearRegression()
prostate_LR.fit(X_train, y_train)

y_pred = prostate_LR(X_test)
mse = ((y_pred - y_test) ** 2).mean()
mpe = abs(y_pred - y_test).mean()

y_pred = prostate_LR(X_train)
sigma_hat_squared = np.sum((y_train - y_pred) ** 2) / (
    X_train.shape[0] - X_train.shape[1] - 2
)
std_error = np.sqrt(sigma_hat_squared * np.diag(inv(X_train.T @ X_train)))
z_score = np.divide(prostate_LR.coef_(), std_error)
print(prostate_LR.coef_())
print(std_error)
print(z_score)

####################################################3
# Q2

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
