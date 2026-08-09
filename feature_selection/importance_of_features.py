import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

data = load_diabetes()
x = data["data"]
cols = ["feature_names"]
y = data["target"]
model = RandomForestRegressor()

model.fit(x, y)

importances = model.feature_importances_
index_sort = np.argsort(importances)

plt.title("Feature Importances")
plt.barh(range(len(index_sort)), importances[index_sort], align = 'center')
plt.yticks(range(len(index_sort)), [cols[i] for i in index_sort])
plt.xlabel("Random Forest Feature Importance")
plt.show()