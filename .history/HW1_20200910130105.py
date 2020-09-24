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
        self.beta = pinv(X.T @ X) @ X.T @ y


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
        self.beta = pinv(X.T @ X + self.l * np.eye(X.shape[1])) @ X.T @ y


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

# Use provided dataset split
# train = prostate_data[prostate_data["train"] == "T"].drop(["train"], axis=1).to_numpy()
# train = np.hstack((np.ones((len(train), 1)), train))
# test = prostate_data[prostate_data["train"] == "F"].drop(["train"], axis=1).to_numpy()
# test = np.hstack((np.ones((len(test), 1)), test))

# split X and y
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]
X_val = val[:, :-1]
y_val = val[:, -1]
ls = np.linspace(0.001, 100, 10000)

##################################################################################
# Q1                                                                             #
##################################################################################
def linear_regression(X_train, y_train, X_test, y_test, column_names, z_score=True):
    LR = LinearRegression()
    LR.fit(X_train, y_train)

    y_pred = LR(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    mpe = abs(y_pred - y_test).mean()
    baseline_mse = ((y_test - y_train.mean()) ** 2).mean()
    print(f"baseline_mse: {baseline_mse}")
    print(f"mse: {mse}")

    if z_score:
        y_pred = LR(X_train)
        sigma_hat_squared = np.sum((y_train - y_pred) ** 2) / (
            X_train.shape[0] - X_train.shape[1] - 2
        )
        std_error = np.sqrt(sigma_hat_squared * np.diag(pinv(X_train.T @ X_train)))
        z_score = np.divide(LR.coef_(), std_error)
        print(LR.coef_())
        print(std_error)
        print(z_score)
        pd.DataFrame(
            {"Coefficient": LR.coef_(), "std error": std_error, "Z Score": z_score},
            index=["intercept"] + column_names,
        )


# linear_regression(X_train, y_train, column_names)

##################################################################################
# Q2                                                                             #
##################################################################################
def ridge_regression(X_train, y_train, X_test, y_test, val, column_names):
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

    ridge = RidgeRegression(ls[idx])
    ridge.fit(X_train[:, 1:], y_train)
    y_pred = ridge(X_test)
    mse = ((y_pred - y_test) ** 2).mean()
    print(f"Lambda: {ls[idx]}")
    print(f"MSE: {mse}")


# ridge_regression(X_train, y_train, X_test, y_test, test, val, column_names)

##################################################################################
# Q3                                                                             #
##################################################################################
def lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names):
    mses = []
    beta_lasso = []
    ls = np.linspace(0, 1, 1000)
    # lambda multiplies L1
    # t enforces the constraint of sum of abs of Beta < t
    for l in ls:
        clf = linear_model.Lasso(alpha=l)
        clf.fit(X_train[:, 1:], y_train)
        beta_lasso.append(clf.coef_)
        y_pred = clf.predict(X_val)
        mses.append(((y_pred - y_val) ** 2).mean())
    beta_lasso = np.array(beta_lasso)

    idx = np.argmin(mses)
    plt.plot(ls, beta_lasso)
    plt.xlabel("Lambdas")
    plt.ylabel("Coefficient")
    plt.xscale("log")
    plt.legend(column_names)
    plt.show()

    clf = linear_model.Lasso(alpha=ls[idx])
    clf.fit(X_train[:, 1:], y_train)
    y_pred = clf.predict(test[:, 1:-1])
    mse = ((y_pred - test[:, -1]) ** 2).mean()
    print(f"Lambda: {ls[idx]}")
    print(f"MSE: {mse}")

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.lasso_path.html
    _, coefs, _ = linear_model.lasso_path(X_train, y_train)

    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]

    plt.plot(xx, coefs.T)
    plt.xlabel("shrinkage factor")
    plt.ylabel("Coefficients")
    plt.title("LASSO Path")
    plt.axis("tight")
    plt.legend(column_names)
    plt.show()


# lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)

##################################################################################
# Q4                                                                             #
##################################################################################
wine_data = pd.read_csv("winequality-red.csv", sep=";")
corr = wine_data.corr()
column_names = wine_data.columns
wine_data = scale(wine_data)
wine_data = np.hstack((np.ones((len(wine_data), 1)), wine_data))
train, temp = train_test_split(wine_data, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]
X_val = val[:, :-1]
y_val = val[:, -1]

linear_regression(X_train, y_train, X_test, y_test, column_names, False)
# ridge_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)
# lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)

##### Nonlinear
wine_data = scale(pd.read_csv("winequality-red.csv", sep=";"))
nonlin_wine_data = np.hstack(
    (
        np.ones((len(wine_data), 1)),
        np.log(wine_data[:, 0] + min(wine_data[:, 0])).reshape(len(wine_data), 1),
        wine_data,
    )
)
new_columns = ["log(fixed acidity)"]
new_columns.extend(column_names)
train, temp = train_test_split(nonlin_wine_data, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]

linear_regression(X_train, y_train, X_test, y_test, column_names, False)
# ridge_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)
# lasso_regression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)
