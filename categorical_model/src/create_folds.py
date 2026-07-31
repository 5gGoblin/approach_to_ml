import pandas as pd
from sklearn import model_selection
import config

if __name__ == "__main__":

    data = pd.read_csv(config.TRAIN_FILE)
    data["kfold"] = -1

    data = data.sample(frac = 1).reset_index(drop = True)
    y = data.target.values

    skf = model_selection.StratifiedKFold(n_splits = 5)

    for f, (t_, v_) in enumerate(skf.split(X = data, y = y)):
        data.loc[v_, "kfold"] = f

    data.to_csv("../input/cat_train_folds.csv", index = False)
    print("Folds file saved.")

    