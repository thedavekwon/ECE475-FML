from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, SCORERS
from sklearn.utils.validation import check_is_fitted

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.inspection import plot_partial_dependence


def compare_rf_to_gbm(X_train, y_train, X_test, y_test, feature_names):
    p = len(feature_names)
    trees = range(1, 200, 10)
    test_absolute_mean_errors = []
    
    for t in trees:
        print(f"On tree: {t}")
        # Random Forest with m = 2
        rf2 = xgb.XGBRFRegressor(eta=0.05, n_estimators=t, colsample_bynode=2.1/p)

        # # Random Forest with m = 6
        rf6 = xgb.XGBRFRegressor(eta=0.05, n_estimators=t, colsample_bynode=6.1/p)

        # # Gradient Boosting with v = 0.05 and depth = 4
        gb4 = xgb.XGBRegressor(eta=0.05, max_depth=4, n_estimators=t)

        # # Gradient Boosting with v = 0.05 and depth = 6
        gb6 = xgb.XGBRegressor(eta=0.05, max_depth=6, n_estimators=t)

        models = [rf2, rf6, gb4, gb6]
        test_absolute_mean_error = []
        for model in models:
            model.fit(X_train, y_train)
            test_absolute_mean_error.append(
                np.sum(np.abs(model.predict(X_test) - y_test)) / len(y_test)
            )

        test_absolute_mean_errors.append(test_absolute_mean_error)

    test_absolute_mean_errors = np.array(test_absolute_mean_errors)

    plt.plot(trees, test_absolute_mean_errors)
    plt.title("California Housing Data")
    plt.xlabel("Number of Trees")
    plt.ylabel("Test Average Absolute Error")
    plt.legend(["RF m = 2", "RF m = 6", "GBM depth = 4", "GBM depth = 6"])
    plt.show()
    
    feature_names = list(map(lambda x: str(x), feature_names))
    rf2.get_booster().feature_names = feature_names
    xgb.plot_importance(rf2)
    plt.show()

    rf6.get_booster().feature_names = feature_names
    xgb.plot_importance(rf6)
    plt.show()

    gb4.get_booster().feature_names = feature_names
    xgb.plot_importance(gb4)
    plt.show()

    gb6.get_booster().feature_names = feature_names
    xgb.plot_importance(gb6)
    plt.show()


data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

compare_rf_to_gbm(X_train, y_train, X_test, y_test, data.feature_names)

data = pd.read_csv("Life-Expectancy-Data.csv")
data = data.drop(["Country", "Status"], axis=1)
data = data.dropna()

y = data["Life expectancy"]
X = data.drop(["Life expectancy"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y, test_size=0.2, random_state=42
)

compare_rf_to_gbm(X_train, y_train, X_test, y_test, X.columns)

