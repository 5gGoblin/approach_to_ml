import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
x = data["data"]
cols = data["feature_names"]
y = data["target"]

model = LinearRegression()

rfe = RFE(
    estimator = model,
    n_features_to_select = 3
)

rfe.fit(x, y)
x_transformed = rfe.transform(x)

print(cols)
print(rfe.support_)
print(np.array(cols)[rfe.support_])
print(cols)
print(rfe.ranking_)