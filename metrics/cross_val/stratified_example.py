import pandas as pd
from sklearn import model_selection

data = pd.read_csv("../datasets/wineQualityReds.csv")
data["kfold"] = -1

data = data.sample(frac = 1).reset_index(drop = True)
y = data.quality.values

kf = model_selection.StratifiedKFold(n_splits = 5)

for f, (t_, v_) in enumerate(kf.split(X = data, y = y)):
    data.loc[v_, 'kfold'] = f

data.to_csv("train_folds_example.csv", index = False)
