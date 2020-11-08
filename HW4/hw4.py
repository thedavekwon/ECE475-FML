from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, SCORERS
from sklearn.utils.validation import check_is_fitted

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

# We use the same parameters to generate the same model as the textbook, which limits the number of leaves and specifies the learning rate (eta) and type of loss (Huber loss), we also search for the optimal number of estimators
# J = 6, v = 0.1, Huber loss
ms = range(0, 41)
param = {
    "eta": [0.1],
    "tree_method": ["hist"],
    "grow_policy": ["lossguide"],
    "max_leaves": [6],
    "n_estimators": ms,
}

clf = GridSearchCV(
    xgb.XGBRegressor(objective="reg:pseudohubererror"),
    param,
    scoring="neg_mean_absolute_error",
)
clf.fit(X_train, y_train)

plt.plot(ms, -clf.cv_results_["mean_test_score"])
plt.title("Training Mean Absolute Error")
plt.xlabel("Number of Boosting Iterations")
plt.ylabel("Mean Absolute Error")
plt.legend(["train mean absolute error"])
plt.show()

# The partial dependence plots tell us that:
# 1) the median house price shows a linear relationship with the median income
# 2) the median house price drops when the average occupants per household increases
# 3) the median house price does not have a strong relationship with the house age
# 4) the median house price also does not have a strong relationship with the average number of rooms
# 5) house age is independent of number of rooms
# Plot Partial Dependence
plot_partial_dependence(
    clf.best_estimator_,
    X_train,
    [0, 5, 1, 2, (1, 2)],
    feature_names=data.feature_names[:6],
)
plt.show()

# Plotting the feature importance tells us that the most relative features are median income, location (longitude and latitude), and occupants
# If we think about what features should correlate with median house value in a district, these results make sense; income and location are obviously strong predictors for house value!
# F-1 Score in x axis shows the relative importance
clf.best_estimator_.get_booster().feature_names = data.feature_names
xgb.plot_importance(clf.best_estimator_)
plt.show()

# Plot tree for fun
xgb.plot_tree(clf.best_estimator_)
plt.show()

print(clf.best_params_)


#####
#####
#####


# This dataset has data concerning housing in Boston. It includes variables such as per capita crime rate by town, proportion of residential land zoned for lots over 25,000 sq. ft, and average number of rooms per dwelling.
# A more detailed explanation of the features can be found here: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# Get Data from the boston housine
# Formatting our data
data = load_boston()
X = pd.DataFrame(data.data)
y = data.target

# Splitting into train and test sets
X_train, y_train, X_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

print(data.feature_names)

# Grid Search Parameters
# eta: learning rate
# tree method:  hist: faster histogram optimized approximate greedy algorithm
#               exact: exact greedy algorithm. Enumerates all split candidates
# grow policy:  lossguide: split at nodes with highest loss change
#               depthwise: split at nodes closest to the root
# max_leaves:   maximum number of nodes to be added. Only relevant for lossguide
# n_estimators: the number of gradient boosted trees or the number of boosting rounds.
# min_child_weight: Minimum sum of instance weight (hessian) needed in a child.
# max_depth: we just use default 6

param = {
    "eta": [0.1, 0.01],
    "tree_method": ["hist", "exact"],
    "grow_policy": ["lossguide", "depthwise"],
    "max_leaves": range(4, 7),
    "n_estimators": range(10, 101),
    "colsample_bylevel": [0.5, 0.75],
}

clf = GridSearchCV(
    xgb.XGBRegressor(objective="reg:pseudohubererror"),
    param,
    scoring="neg_mean_absolute_error",
)
clf.fit(X_train, y_train)

# # F-1 Score in x axis shows the relative importance
clf.best_estimator_.get_booster().feature_names = list(
    map(lambda x: str(x), data.feature_names)
)
xgb.plot_importance(clf.best_estimator_.get_booster())
plt.show()

# best params
# {'eta': 0.1, 'grow_policy': 'lossguide', 'max_leaves': 2, 'n_estimators': 36, 'tree_method': 'exact'}

# Top 3 highest importance features
# RM: Average number of rooms per dwelling
# DIS: Weighted distances to five Boston employment centers
# LSTAT: % lower status of the population

# Analysis of feature importance
# The most important feature by far is RM, average number number of rooms per dwelling. The fact that this feature is thema most important makes a lot of sense, as most apartments/houses are marketed based on number of rooms and square footage (which are highly correlated with one another). The next most important features are DIS, the weighted distances to five Boston employment centers, and LSTAT, the percentage of lower status people in the population. The fact that DIS is an important feature means that houses get a lot of value from being close to popular, high traffic employment centers. The importance of LSTAT signifies that same income classes are generally grouped together in Boston.
# All-in-all the fact that these features were the most important makes a lot of sense in predicting a house's value.

# For the partial dependence plots, a horizontal line indicates that the label is independent of the feature which are all features except for LSTAT and RM;
# on that other hand, we can see in the 6th plot that if the number of rooms is greater than 7, the housing price increases with the number of rooms
# we also see in the last plot, that as the percentage of the lower status of the population increases in an area, that the housing price increases

plot_partial_dependence(
    clf.best_estimator_,
    X_train,
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    feature_names=data.feature_names,
)
plt.show()
