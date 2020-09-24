import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.linalg import inv, pinv

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import scale


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
corr = prostate_data.corr()  # Find correlation between features

prostate_data = prostate_data.to_numpy()
prostate_data = scale(prostate_data)
prostate_data = np.hstack((np.ones((len(prostate_data), 1)), prostate_data))
train, temp = train_test_split(prostate_data, test_size=0.2)
test, val = train_test_split(temp, test_size=0.5)

# Use dataset split
# train = prostate_data[prostate_data["train"] == "T"].drop(["train"], axis=1).to_numpy()
# train = np.hstack((np.ones((len(train), 1)), train))
# test = prostate_data[prostate_data["train"] == "F"].drop(["train"], axis=1).to_numpy()
# test = np.hstack((np.ones((len(test), 1)), test))

# split X and y
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]
ls = np.linspace(0.001, 100, 10000)

##################################################################################
# Q1                                                                             #
##################################################################################
# prostate_LR = LinearRegression()
# prostate_LR.fit(X_train, y_train)

# y_pred = prostate_LR(X_test)
# mse = ((y_pred - y_test) ** 2).mean()
# mpe = abs(y_pred - y_test).mean()
# baseline_mse = ((y_test - y_train.mean()) ** 2).mean()
# print(f"baseline_mse: {baseline_mse}")
# print(f"mse: {mse}")

# y_pred = prostate_LR(X_train)
# sigma_hat_squared = np.sum((y_train - y_pred) ** 2) / (
#     X_train.shape[0] - X_train.shape[1] - 2
# )
# std_error = np.sqrt(sigma_hat_squared * np.diag(inv(X_train.T @ X_train)))
# z_score = np.divide(prostate_LR.coef_(), std_error)
# print(prostate_LR.coef_())
# print(std_error)
# print(z_score)
# pd.DataFrame(
#     {"Coefficient": prostate_LR.coef_(), "std error": std_error, "Z Score": z_score},
#     index=["intercept"] + column_names,
# )

##################################################################################
# Q2                                                                             #
##################################################################################
# mses = []
# beta_ridges = []
# for l in ls:
#     prostate_R = RidgeRegression(l)
#     prostate_R.fit(X_train, y_train)
#     beta_ridge = prostate_R.coef_()[1:]
#     beta_ridges.append(beta_ridge)
#     y_pred = prostate_R(val[:, :-1])
#     mses.append(((y_pred - val[:, -1]) ** 2).mean())
# plt.plot(ls, np.array(beta_ridges))
# plt.xlabel("Lambda")
# plt.ylabel("Coefficient")
# plt.legend(column_names)
# plt.show()
# idx = np.argmin(mses)
# print(f"Lambda: {ls[idx]}")
# print(f"Smallest MSE: {mses[idx]}")

##################################################################################
# Q3                                                                             #
##################################################################################
mses = []
ss = []
beta_lasso = []
ls = np.linspace(0, 30, 1000)
for l in ls:
    clf = linear_model.Lasso(alpha=l)
    clf.fit(X_train[:, 1:], y_train)
    beta_lasso.append(clf.coef_)
    y_pred = clf.predict(val[:, 1:-1])
    ss.append(l / np.sum(np.abs(clf.coef_)))
    mses.append(((y_pred - val[:, -1]) ** 2).mean())
beta_lasso = np.array(beta_lasso)
ss = np.array(ss)
plt.plot(ss[ss[0 < ss && ss < 1]], beta_lasso[ss[0 < ss < 1]])
plt.xlabel("Shrinkage Factor")
plt.ylabel("Coefficient")
plt.xscale("log")
plt.legend(column_names)
plt.show()
idx = np.argmin(mses)
print(f"Shrinkage Factor: {ss[idx]}")
print(f"Smallest MSE: {mses[idx]}")

##################################################################################
# Q4                                                                             #
##################################################################################

wine_data = pd.read_csv("winequality-red.csv")