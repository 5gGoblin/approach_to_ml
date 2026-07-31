import config
import pandas as pd
from sklearn import linear_model, metrics, preprocessing


def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)
    print(f"Original Data Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    num_cols = [col for col in data.columns if (data[col].dtype == int) and (col != "kfold")]

    data = data.drop(num_cols, axis = 1).copy()
    print(f"Categorical Data Shape: {data.shape[0]} rows, {data.shape[1]} columns")

    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }

    data["income"] = data.income.map(target_mapping)

    features = [f for f in data.columns if f not in ("kfold", "income")]
    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")

    ohe = preprocessing.OneHotEncoder()
    ohe.fit(data[features])
    
    data_train = data[data.kfold != fold].reset_index(drop = True)
    data_valid = data[data.kfold == fold].reset_index(drop = True)

    x_train = ohe.transform(data_train[features])
    y_train = data_train.income.values
    x_valid = ohe.transform(data_valid[features])
    y_valid = data_valid.income.values

    model = linear_model.LogisticRegression()

    model.fit(x_train, y_train)
    predictions = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, predictions)

    print(f"Fold: {fold}, AUC: {auc}")


run(2)