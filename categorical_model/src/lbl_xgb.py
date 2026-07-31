import pandas as pd
import config
import xgboost as xgb
from sklearn import metrics, preprocessing

def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)
    features = [f for f in data.columns if f not in ("id", "target", "kfold")]
    #fill na 
    for col in features:
        data[col] = data[col].astype(str).fillna("NONE")
    
    #label encode the features
    label_encoders = {}
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(data[col])
        data[col] = lbl.transform(data[col])
        label_encoders[col] = lbl

    data_train = data[data["kfold"] != fold].reset_index(drop = True)
    data_valid = data[data["kfold"] == fold].reset_index(drop = True)

    x_train = data_train[features].values
    x_valid = data_valid[features].values

    model = xgb.XGBClassifier(
        n_jobs = -1,
        max_depth = 7,
        n_estimators = 200
    )

    model.fit(x_train, data_train.target.values)

    pred = model.predict_proba(x_valid)[:, 1]
    auc = metrics.roc_auc_score(data_valid.target.values, pred)

    print(f"Fold: {fold}, AUC: {auc}")

if __name__ == "__main__":

    run(1)