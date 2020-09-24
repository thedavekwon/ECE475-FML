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
def linear_regression(X_train, y_train):
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    
    y_pred = LR(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    mpe = abs(y_pred - y_test).mean()
    baseline_mse = ((y_test - y_train.mean()) ** 2).mean()
    print(f"baseline_mse: {baseline_mse}")
    print(f"mse: {mse}")
    
    y_pred = LR(X_train)
    sigma_hat_squared = np.sum((y_train - y_pred) ** 2) / (
        X_train.shape[0] - X_train.shape[1] - 2
    )
    std_error = np.sqrt(sigma_hat_squared * np.diag(inv(X_train.T @ X_train)))
    z_score = np.divide(LR.coef_(), std_error)
    print(LR.coef_())
    print(std_error)
    print(z_score)
    pd.DataFrame(
        {"Coefficient": LR.coef_(), "std error": std_error, "Z Score": z_score},
        index=["intercept"] + column_names,
    )

# linear_regression(X_train, y_train)

##################################################################################
# Q2                                                                             #
##################################################################################
def ridge_regression(X_train, y_train):
    mses = []
    beta_ridges = []
    for l in ls:
        ridge = RidgeRegression(l)
        ridge.fit(X_train, y_train)
        beta_ridge = ridge.coef_()[1:]
        beta_ridges.append(beta_ridge)
        y_pred = ridge(val[:, :-1])
        mses.append(((y_pred - val[:, -1]) ** 2).mean())
    plt.plot(ls, np.array(beta_ridges))
    plt.xlabel("Lambda")
    plt.ylabel("Coefficient")
    plt.legend(column_names)
    plt.show()
    idx = np.argmin(mses)
    print(f"Lambda: {ls[idx]}")
    print(f"Smallest MSE: {mses[idx]}")

# ridge_regression(X_train, y_train)
##################################################################################
# Q3                                                                             #
##################################################################################
# mses = []
# ss = []
# beta_lasso = []
# s = np.linspace(0, 1, 1000)
# # lambda multiplies L1
# # t enforces the constraint of sum of abs of Beta < t

# for t in ts:
#     clf = linear_model.Lasso(alpha=t)
#     clf.fit(X_train[:, 1:], y_train)
#     beta_lasso.append(clf.coef_)
#     y_pred = clf.predict(val[:, 1:-1])
#     ss.append(t / np.sum(np.abs(clf.coef_)))
#     mses.append(((y_pred - val[:, -1]) ** 2).mean())
# beta_lasso = np.array(beta_lasso)
# ss = np.array(ss)
# plt.plot(ss, beta_lasso)
# plt.xlabel("Shrinkage Factor")
# plt.ylabel("Coefficient")
# plt.xscale("log")
# plt.legend(column_names)
# plt.show()
# idx = np.argmin(mses)
# print(f"Shrinkage Factor: {ts[idx]}")
# print(f"Smallest MSE: {mses[idx]}")


# Taken from Tiffany Yu's advice from : https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
def lasso_regression(X_train, y_train):
    _, _, coefs = linear_model.lars_path(X_train, y_train, method='lasso', verbose=True)
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('LASSO Path')
    plt.axis('tight')
    plt.show()
##################################################################################
# Q4                                                                             #
##################################################################################

wine_data = pd.read_csv("winequality-red.csv")
print(wine_data)
