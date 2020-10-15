import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
from scipy.special import expit

eps = 1e-6


def sigmoid(x):
    # return 1 / (1 + np.exp(-x))
    return expit(x)


class LogisticRegression:
    def __init__(self, step, batch_size, lamb=None, regularization=None):
        self.theta = None
        self.step = step
        self.batch_size = batch_size
        self.lamb = lamb
        self.regularization = regularization
        self.q = None
        self.u = 0

    def applyL1Penalty(self):
        for i in range(len(self.theta)):
            z = self.theta[i]
            if self.theta[i] > 0:
                self.theta[i] = max(0, self.theta[i] - (self.u + self.q[i]))
            else:
                self.theta[i] = min(0, self.theta[i] + (self.u - self.q[i]))
            self.q[i] = self.q[i] + (self.theta[i] - z)

    def fit(self, X, y):
        X, y = shuffle(X, y)
        X = X[: self.batch_size, :]
        y = y[: self.batch_size]

        if self.theta is None:
            # self.theta = np.random.rand(X.shape[1], 1)
            self.theta = np.zeros((X.shape[1], 1))
            self.q = np.zeros_like(self.theta)
        self.u = self.u + self.step * self.lamb

        # print(sigmoid(X @ self.theta).shape)
        y = y.reshape((len(y), 1))
        if self.lamb and self.regularization == "L1":
            self.theta = self.theta + self.step * (
                X.T
                @ (y - sigmoid(X @ self.theta))
                # - self.lamb / self.batch_size * np.sign(self.theta)
            )
            self.applyL1Penalty()
        elif self.lamb and self.regularization == "L2":
            # Kinda from here not really: https://towardsdatascience.com/implement-logistic-regression-with-l2-regularization-from-scratch-in-python-20bd4ee88a59#4077
            self.theta = self.theta + self.step * (
                X.T @ (y - sigmoid(X @ self.theta)) + self.lamb * 2 * np.sum(self.theta)
            )
        else:
            # Unregularized
            self.theta = self.theta + self.step * X.T @ (y - sigmoid(X @ self.theta))

    def predict(self, X):
        return sigmoid(X @ self.theta) > 0.5


df = pd.read_csv("SAheart.data")
df = df.drop(["row.names"], axis=1).replace("Present", 1).replace("Absent", 0)
column_names = df.columns
# scatter matrix plot
# X = df.drop(["chd", "adiposity", "typea"], axis=1)
# pd.plotting.scatter_matrix(X, c=df["chd"])
# plt.show()

# dataset.describe()
dataset = df.to_numpy()

temp_y = dataset[:, -1]
# Regularize the prostate data to be 0 mean and unit variance
dataset = scale(dataset)
dataset[:, -1] = temp_y
# Adding our Intercept column (column of ones) to the data
dataset = np.hstack((np.ones((len(dataset), 1)), dataset))

# Splitting our data into train (80%), val (10%), and test (10%)
train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]
X_val = val[:, :-1]
y_val = val[:, -1]


def logisticRegression(
    X_train, y_train, X_test, y_test, lamb=None, regularization=None
):
    LR = LogisticRegression(0.01, 32, lamb=lamb, regularization=regularization)

    # Training our logistic regression model on the training set
    train_log_likelihoods = []
    test_log_likelihoods = []
    for _ in range(1, 101):
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_train)
        train_accuracy = np.sum(y_pred.flatten() == y_train) / len(y_train)
        # print(f"{i}th Train Accuracy: {train_accuracy}")
        print(LR.theta)
        train_log_likelihoods.append(
            np.sum(
                (X_train @ LR.theta) * y_train - np.log(1 + np.exp(X_train @ LR.theta))
            )
        )
        test_log_likelihoods.append(
            np.sum((X_test @ LR.theta) * y_test - np.log(1 + np.exp(X_test @ LR.theta)))
        )
    plt.plot(list(range(1, 101)), test_log_likelihoods)
    plt.legend(["test"])
    print(test_log_likelihoods)
    plt.show()
    # Applying the trained model to our test set
    y_pred = LR.predict(X_test)
    # Accuracy (percent correct)
    test_accuracy = np.sum(y_pred.flatten() == y_test) / len(y_test)
    print(f"Test Accuracy: {test_accuracy}")


def plot_lambda_weight(X_train, y_train, regularization):
    thetas = []
    lambdas = np.linspace(0, 1, 100)
    for l in lambdas:
        LR = LogisticRegression(0.01, 32, lamb=l, regularization=regularization)
        for _ in range(1, 101):
            LR.fit(X_train, y_train)
        thetas.append(LR.theta[1:])
    plt.plot(lambdas, np.squeeze(np.array(thetas)))
    plt.show()


def stepForwardLogisticRegression(
    X_train, y_train, X_test, y_test, X_val, y_val, column_names
):
    selected_features = []
    feature_accuracies = []
    idxs = list(range(X_train.shape[1]))

    while len(idxs) != len(selected_features):
        validation_accuracies = []
        left_features = list(set(idxs) - set(selected_features))
        for idx in range(len(left_features)):
            LR = LogisticRegression(0.01, 16)
            X_train_temp = X_train[:, selected_features + [left_features[idx]]].reshape(
                X_train.shape[0], len(selected_features) + 1
            )
            for _ in range(1, 101):
                LR.fit(X_train_temp, y_train)
            X_val_temp = X_val[:, selected_features + [left_features[idx]]].reshape(
                X_val.shape[0], len(selected_features) + 1
            )
            y_pred = LR.predict(X_val_temp)
            validation_accuracies.append(np.sum(y_pred.flatten() == y_val) / len(y_val))

        validation_accuracies = np.array(validation_accuracies)
        if not feature_accuracies or any(
            validation_accuracies > feature_accuracies[-1]
        ):
            selected_idx = np.argmax(validation_accuracies)
            selected_features.append(left_features[selected_idx])
            feature_accuracies.append(validation_accuracies[selected_idx])
        else:
            break
    print(
        f"Selected feature in order: {list(map(lambda x: column_names[x], selected_features))}"
    )
    print(f"Accurarcies adding featres: {feature_accuracies}")

    LR = LogisticRegression(0.01, 16)
    for _ in range(1, 101):
        LR.fit(X_train[:, selected_features], y_train)
    # Applying the trained model to our test set
    y_pred = LR.predict(X_test[:, selected_features])
    # Accuracy (percent correct)
    test_accuracy = np.sum(y_pred.flatten() == y_test) / len(y_test)
    print(f"Test Accuracy: {test_accuracy}")


# logisticRegression(X_train, y_train, X_test, y_test)
# logisticRegression(X_train, y_train, X_test, y_test, 0.0001, "L1")
# logisticRegression(X_train, y_train, X_test, y_test, 0.01, "L2")
# stepForwardLogisticRegression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)

# for x in range(0,100):
#     print("penalty:   "(x/100)
# logisticRegression(X_train, y_train, X_test, y_test, x/100, "L2")
#plot_lambda_weight(X_train, y_train, "L1")

# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()
# dataset = data['data']
# column_names = data['feature_names']

# # Regularize the prostate data to be 0 mean and unit variance
# dataset = scale(dataset)
# # Adding our Intercept column (column of ones) to the data
# dataset = np.hstack((np.ones((len(dataset), 1)), dataset))
# dataset = np.hstack((dataset, data['target'].reshape(len(dataset), 1)))
# # Splitting our data into train (80%), val (10%), and test (10%)
# train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
# test, val = train_test_split(temp, test_size=0.5, random_state=42)

# X_train = train[:, :-1]
# y_train = train[:, -1]
# X_test = test[:, :-1]
# y_test = test[:, -1]
# X_val = val[:, :-1]
# y_val = val[:, -1]

# logisticRegression(X_train, y_train, X_test, y_test)
# logisticRegression(X_train, y_train, X_test, y_test, .001, "L1")
# logisticRegression(X_train, y_train, X_test, y_test, 0.01, "L2")
# stepForwardLogisticRegression(X_train, y_train, X_test, y_test, X_val, y_val, column_names)

# Multi-Nominal
from sklearn import datasets
from sklearn.preprocessing import scale, OneHotEncoder

iris = datasets.load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)
y = OneHotEncoder(sparse=False).fit_transform(X=y)

dataset = np.hstack((np.ones((len(X), 1)), X))
dataset = np.hstack((dataset, y))
# # Splitting our data into train (80%), val (10%), and test (10%)
train, temp = train_test_split(dataset, test_size=0.2, random_state=42)
test, val = train_test_split(temp, test_size=0.5, random_state=42)

X_train = train[:, :-3]
y_train = train[:, -3:]
X_test = test[:, :-3]
y_test = test[:, -3:]
X_val = val[:, :-3]
y_val = val[:, -3:]

print(X_train.shape)
print(y_train.shape)