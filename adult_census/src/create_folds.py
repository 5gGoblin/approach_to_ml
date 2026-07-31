import pandas as pd
from sklearn import model_selection
import config

all_features = ["age", "workclass", "fnlwgt", "education","education_num","marital_status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week]","native_country","income"]
data = pd.read_csv(config.TRAIN_FILE, names = all_features)

data["kfold"] = -1
data = data.sample(frac=1).reset_index(drop = True)
y = data.income.values

skf = model_selection.StratifiedKFold(n_splits = config.FOLDS)

for f, (t_, v_) in enumerate(skf.split(X = data, y = y)):
    data.loc[v_, "kfold"] = f

data.to_csv("../input/adult_folds_train.csv", index = False)
print(f"Folds(k={config.FOLDS}) file saved!")






