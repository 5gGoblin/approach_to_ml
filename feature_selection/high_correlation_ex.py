import pandas as pd 
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import VarianceThreshold

housing = fetch_california_housing()
x = housing["data"]
cols = housing["feature_names"]
target = housing["target"]

data = pd.DataFrame(x, columns = cols)
data["MedInc_sqrt"] = data.MedInc.apply(np.sqrt)

print(data.corr())