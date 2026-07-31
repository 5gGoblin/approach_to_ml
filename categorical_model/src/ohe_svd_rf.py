#ONE HOW ENCODE - SINGULAR VALUE DECOMPOSITION

import pandas as pd
import config
from sklearn import preprocessing, decomposition, ensemble, metrics

def run(fold):

    data = pd.read_csv(config.TRAIN_FOLDS_FILE)

    features = [f for f in data.columns if f not in ("id", "target", "kfold")]

    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")

    data_train = data[data["kfold"] != fold].reset_index(drop = True)
    data_valid = data[data["kfold"] == fold].reset_index(drop = True)

    ohe = preprocessing.OneHotEncoder(handle_unknown="ignore")

    ohe.fit(data_train[features])
    x_train = ohe.transform(data_train[features])
    x_valid = ohe.transform(data_valid[features])

    svd = decomposition.TruncatedSVD(n_components = 120)

    svd.fit(x_train)
    x_train = svd.transform(x_train)
    x_valid = svd.transform(x_valid)

    model = ensemble.RandomForestClassifier(n_jobs = -1)
    model.fit(x_train, data_train.target.values)
    valid_preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(data_valid.target.values, valid_preds)

    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == '__main__':
    run(0)