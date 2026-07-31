import numpy as np
import confusion_matrix as cm
from collections import Counter
from sklearn import metrics


def weighted_f1(true, pred):
    num_classes = len(np.unique(true))
    class_counts = Counter(true)
    f1 = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in true]
        temp_pred = [1 if p == class_ else 0 for p in pred]

        tp = cm.true_positive(temp_true, temp_pred)
        fp = cm.false_positive(temp_true, temp_pred)
        fn = cm.false_negative(temp_true, temp_pred)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        if precision + recall != 0:
            temp_f1 = 2 * precision * recall / (precision + recall)
        else:
            temp_f1 = 0

        weighted_f1 = class_counts[class_] * temp_f1

        f1 += weighted_f1

    overall_f1 = f1 / len(true)

    return overall_f1

if __name__ == '__main__':

    y_true = [0, 1, 2, 0, 1, 2, 0, 2, 2]
    y_pred = [0, 2, 1, 0, 2, 1, 0, 0, 2]

    w_f1 = weighted_f1(y_true, y_pred)
    w_f1_sk = metrics.f1_score(y_true, y_pred, average = "weighted")

    print(f"Weighted F1 Score: {w_f1:.3f}")
    print(f"Weighted F1 Score (Sci-Kit): {w_f1_sk:.3f}")
