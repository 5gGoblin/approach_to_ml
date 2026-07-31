#STRATIFIED K FOLD FOR REGRESSION

import pandas as pd
import numpy as np
from sklearn import datasets, model_selection

def cretate_folds(data):
    data['kfold'] = -1
    data = data.sample(frac = 1).reset_index(drop = True) #randomize data

    #Sturge's rule for bins 1 + log_2(N)
    num_bins = int(np.floor(1 + np.log2(len(data))))
    data['bins'] = pd.cut(data['target'], bins = num_bins, labels = False)
    
    kf = model_selection.StratifiedKFold(n_splits = 5)

    for f, (t_, v_) in enumerate(kf.split(X = data, y = data.bins.values)):
        data.loc[v_, 'kfold'] = f

    #dop bins column
    data = data.drop("bins", axis = 1)

    return data

if __name__ == "__main__":

    #create sample dataset 15000 samples, 100 features, 1 target
    X, y = datasets.make_regression(
        n_samples = 15000, 
        n_features= 100,
        n_targets= 1
    )

    df = pd.DataFrame(
        X,
        columns = (f"f_{i}" for i in range(X.shape[1]))
    )
    df["target"] = y

    df = cretate_folds(df)
    print(df)



