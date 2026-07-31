import pandas as pd
from sklearn.model_selection import KFold


data = pd.read_csv("../input/mnist_train.csv")

k = 5
kf = KFold(n_splits = k, shuffle= True)

data['fold'] = -1

for fold_num, (train_idx, val_idx) in enumerate(kf.split(data), start = 1):
    data.loc[val_idx, 'fold'] = fold_num

data.to_csv("../input/mnist_train_folds.csv", index = False)
print("CSV file saved!")