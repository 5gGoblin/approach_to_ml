import pandas as pd
import config
from sklearn import linear_model, preprocessing, metrics

def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)

    features = [
        f for f in data.columns if f not in ("id", "target", "kfold")
    ]

    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")

    data_train = data[data.kfold != fold].reset_index(drop = True)
    data_valid = data[data.kfold == fold].reset_index(drop = True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat(
        [data_train[features], data_valid[features]],
        axis = 0
    )

    ohe.fit(full_data[features])

    x_train = ohe.transform(data_train[features])
    x_valid = ohe.transform(data_valid[features])

    model = linear_model.LogisticRegression(max_iter=1000)

    model.fit(x_train, data_train.target.values)
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(data_valid.target.values, valid_preds)
    print(auc)

if __name__ == '__main__':

    for fold_ in range(config.FOLDS):
        run(fold_)
