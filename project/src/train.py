import os
import config
import argparse
import joblib
import pandas as pd
from sklearn import metrics
import model_dispatcher

def run(fold, model):
    data = pd.read_csv(config.TRAINING_FILE)

    data_train = data[data.fold != fold].reset_index(drop = True)
    data_valid = data[data.fold == fold].reset_index(drop = True)

    x_train = data_train.drop("label", axis = 1).values
    y_train = data_train.label.values

    x_val = data_valid.drop("label", axis = 1).values
    y_val = data_valid.label.values

    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_val)

    accuracy = metrics.accuracy_score(y_val, preds)
    print(f"Model: {model}, Fold = {fold}, Accuracy = {accuracy:.3f}")

    joblib.dump(clf,
                os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold", 
        type = int
    )
    parser.add_argument(
        "--model",
        type = str
    )
    args = parser.parse_args()

    run(fold = args.fold,
        model = args.model
        )