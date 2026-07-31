import pandas as pd
import xgboost as xgb
import itertools
import config
from sklearn import metrics, preprocessing


def feature_eng(df, cat_cols):
    combinations = list(itertools.combinations(cat_cols, 2))
    for col1, col2 in combinations:
        df[col1+"_"+col2] = df[col1].astype(str) + "_" + df[col2].astype(str)
    
    return df
    

def run(fold):
    data = pd.read_csv(config.TRAIN_FOLDS_FILE)

    num_cols = [col for col in data.columns if data[col].dtype == int and col != "kfold"]
    cat_columns = [col for col in data.columns if col not in num_cols and col not in ("kfold", "income")]
    
    data = feature_eng(data.copy(), cat_columns)

    
    target_mappings = {
        " <=50K": 0,
        " >50K": 1
    }

    data["income"] = data.income.map(target_mappings)
    features = [f for f in data.columns if f not in ("income", "kfolds")]

    for col in features:
        if col not in num_cols:
            data[col] = data[col].astype(str).fillna("NONE")

    encoders = {}

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()

            lbl.fit(data[col])
            data[col] = lbl.transform(data[col])
            #if we want to use that column encoder later on
            encoders[col] = lbl

    data_train = data[data["kfold"] != fold].reset_index(drop = True)
    data_valid = data[data["kfold"] == fold].reset_index(drop = True)

    x_train = data_train[features].values
    y_train = data_train["income"].values

    x_valid = data_valid[features].values
    y_valid = data_valid["income"].values

    model = xgb.XGBClassifier(n_jobs = -1)

    model.fit(x_train, y_train)
    preds = model.predict_proba(x_valid)[:,1]

    auc = metrics.roc_auc_score(y_valid, preds)

    print(f"Fold: {fold}, AUC: {auc:.3f}")

    
run(0)