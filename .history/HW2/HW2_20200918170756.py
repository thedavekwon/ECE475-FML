import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.utils import shuffle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lambda, step, batch_size):
        self.theta = None
        self.step = step
        self.batch_size = batch_size

    def fit(self, X, y, regularization="None"):
        X, y = shuffle(X, y)
        X = X[: self.batch_size, :]
        y = y[: self.batch_size]

        if self.theta is None:
            self.theta = np.random.rand(X.shape[1], 1)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if(regularization = "l2"):
                    self.theta[j] = self.theta[j] + self.step * (self.c*(y[i] - sigmoid(self.theta.T @ X[i, :])) * X[i][j]
                self.theta[j] = (
                    self.theta[j]
                    + self.step * (y[i] - sigmoid(self.theta.T @ X[i, :])) * X[i][j]
                )

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


def logisticRegression(X_train, y_train, X_test, y_test):
    LR = LogisticRegression(0.01, 16)

    # Training our logistic regression model on the training set
    for i in range(1, 101):
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_train)
        train_accuracy = np.sum(y_pred.flatten() == y_train) / len(y_train)
        print(f"{i}th Train Accuracy: {train_accuracy}")

    # Applying the trained model to our test set
    y_pred = LR.predict(X_test)
    # Accuracy (percent correct)
    test_accuracy = np.sum(y_pred.flatten() == y_test) / len(y_test)
    print(f"Test Accuracy: {test_accuracy}")


def stepwiseLogisticRegression(X_train, y_train, X_test, y_test):
    selected_features = []
    feature_train_accuracies = []
    idxs = list(range(X_train.shape[1]))

    while len(idxs) != len(selected_features):
        train_accuracies = []
        left_features = list(set(idxs) - set(selected_features))
        for idx in range(len(left_features)):
            LR = LogisticRegression(0.01, 16)
            X_train_temp = X_train[:, selected_features + [left_features[idx]]].reshape(
                X_train.shape[0], len(selected_features) + 1
            )
            for _ in range(1, 101):
                LR.fit(X_train_temp, y_train)
            y_pred = LR.predict(X_train_temp)
            train_accuracies.append(np.sum(y_pred.flatten() == y_train) / len(y_train))

        train_accuracies = np.array(train_accuracies)
        if not feature_train_accuracies or any(
            train_accuracies > feature_train_accuracies[-1]
        ):
            selected_idx = np
            selected_features.append(left_features[selected_idx])
            feature_train_accuracies.append(train_accuracies[selected_idx])
        else:
            break
    print(list(map(lambda x: column_names[x], selected_features)))
    print(feature_train_accuracies)

    # # Applying the trained model to our test set
    # y_pred = LR.predict(X_test)
    # # Accuracy (percent correct)
    # test_accuracy = np.sum(y_pred.flatten() == y_test)/len(y_test)
    # print(f"Test Accuracy: {test_accuracy}")


def regularizedLogisticRegression(X_train, y_train, X_test, y_test):
    LR = LogisticRegression(0.01, 16)

    # Training our logistic regression model on the training set
    for i in range(1, 101):
        LR.fit(X_train, y_train)
        y_pred = LR.predict(X_train)
        train_accuracy = np.sum(y_pred.flatten() == y_train) / len(y_train)
        print(f"{i}th Train Accuracy: {train_accuracy}")

    # Applying the trained model to our test set
    y_pred = LR.predict(X_test)
    # Accuracy (percent correct)
    test_accuracy = np.sum(y_pred.flatten() == y_test) / len(y_test)
    print(f"Test Accuracy: {test_accuracy}")


# logisticRegression(X_train, y_train, X_test, y_test)
stepwiseLogisticRegression(X_train, y_train, X_test, y_test)
# regularizedLogisticRegression(X_train, y_train, X_test, y_test)

