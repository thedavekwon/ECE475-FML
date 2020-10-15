from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.inspection import plot_partial_dependence


data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

# Convert to DMatrix
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dtest = xgb.DMatrix(X_test, label=y_test)

# J = 6, v = 0.1, Huber loss
param = {"eta": 0.1, "tree_method": "hist", "grow_policy": "lossguide", "max_leaves": 6}

ms = range(0, 41)
train_maes = []
test_maes = []
for m in ms:
    bst = xgb.XGBRegressor(
        objective="reg:pseudohubererror",
        n_estimators=m,
        **param
    )
    bst.fit(X_train, y_train)
    y_test_pred = bst.predict(X_test)
    y_train_pred = bst.predict(X_train)
    train_maes.append(mean_absolute_error(y_train, y_train_pred))
    test_maes.append(mean_absolute_error(y_test, y_test_pred))

plt.plot(ms, train_maes)
plt.plot(ms, test_maes)
plt.title("Training and Test Mean Absolute Error")
plt.xlabel("Number of Boosting Iterations")
plt.ylabel("Mean Absolute Error")
plt.legend(["train mean absolute error", "test mean absolute error"])
plt.show()

# Plot Partial Dependence
features = ["MedInc", "AveOccup", "HouseAge", "AveRooms"]
plot_partial_dependence(bst, X_train, [0, 5, 1, 2, (1,2)], feature_names=data.feature_names[:6])
plt.show()

# F-1 Score in x axis shows the relative importance
# bst.get_booster().feature_names = data.feature_names
# xgb.plot_importance(bst)
# plt.show()

# Plot tree for fun
# xgb.plot_tree(bst)
# plt.show()