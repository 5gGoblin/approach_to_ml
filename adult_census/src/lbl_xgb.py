import pandas as pd
import xgboost as xgb
import config
from sklearn import metrics, preprocessing

def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)
    num_cols = [col for col in data.columns if data[col].dtype == int and col != "kfold"]
    data = data.drop(num_cols, axis = 1).copy()

    target_mapping = {
        " <=50K": 0,
        " >50K": 1
    }

    data["income"] = data.income.map(target_mapping)

    features = [f for f in data.columns if f not in ("kfold", "income")]

    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")
    label_encoders = {}
    
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(data[col])
        data[col] = lbl.transform(data[col])
        label_encoders[col] = lbl

    data_train = data[data.kfold != fold].reset_index(drop = True)
    data_valid = data[data.kfold == fold].reset_index(drop = True)

    x_train = data_train[features].values
    y_train = data_train.income.values
    x_valid = data_valid[features].values
    y_valid = data_valid.income.values

    model = xgb.XGBClassifier(
        n_jobs = -1
    )
    model.fit(x_train, y_train)
    predictions = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(y_valid, predictions)

    print(f"Fold: {fold}, AUC: {auc}")

run(4)

