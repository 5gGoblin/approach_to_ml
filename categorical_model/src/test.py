import pandas as pd
import config

data = pd.read_csv(config.TRAIN_FOLDS_FILE)

print(data.shape)

print(data.kfold.value_counts())

for i in range(len(data.kfold.unique())):
    print(data[data.kfold == i].target.value_counts())