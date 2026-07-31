import pandas as pd
import config
from sklearn import ensemble, preprocessing, metrics

def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)

    features = [
        f for f in data.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")

    lbl = preprocessing.LabelEncoder()

    for col in features:
        lbl.fit(data[col])
        data[col] = lbl.transform(data[col])
    
    data_train = data[data.kfold != fold].reset_index(drop = True)
    data_valid = data[data.kfold == fold].reset_index(drop = True)

    x_train = data_train[features].values
    x_valid = data_valid[features].values

    model = ensemble.RandomForestClassifier(n_jobs = -1)

    model.fit(x_train, data_train.target.values)

    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(data_valid.target.values, valid_preds)
    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == '__main__':

    for fold_ in range(config.FOLDS):
        run(fold_)
