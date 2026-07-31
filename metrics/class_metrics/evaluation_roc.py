import confusion_matrix as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
EPS = 1e-5

def evaluation_metric(true, pred, eval):
    tp = cm.true_positive(true, pred)
    tn = cm.true_negative(true, pred)
    fp = cm.false_positive(true, pred)
    fn = cm.false_negative(true, pred)

    if eval == "accuracy":
        acc  = tp + tn / (tp + tn + fp + fn)
        return acc
    elif eval == "precision":
        prec = tp / (tp + fp)
        return prec
    elif eval == "recall":
        rec = tp / (tp + fn)
        return rec
    
def false_positive_rate(true, pred):
    tn = cm.true_negative(true, pred)
    fp = cm.false_positive(true, pred)
    return fp / (tn + fp)

def true_positive_rate(true, pred): #sensitivity
    return evaluation_metric(true, pred, "recall")

if __name__ == "__main__":
    true_vals = np.random.choice([0, 1], size = 50, p = [.8, .2])
    predicted = np.random.choice([0, 1], size = 50, p = [.8, .2])

    accuracy = evaluation_metric(true_vals, predicted, "accuracy")
    precision = evaluation_metric(true_vals, predicted, "precision")
    recall = evaluation_metric(true_vals, predicted, "recall")

    f1 = (2*precision*recall) / (precision + recall + EPS)

    print(f"Accuracy: {accuracy:.2f} \nPrecision: {precision:.2f} \nRecall: {recall:.2f} \nF1: {f1:.2f}")
    
    sk_metric = metrics.f1_score(true_vals, predicted)
    print(f"Sci-Kit Learn F1 Score: {sk_metric:.2f}")

    #ROC curve

    tpr_list = []
    fpr_list = []

    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_pred = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05,
            0.9, 0.5, 0.3, 0.66, 0.3, 0.2,
            0.85, 0.15, 0.99]
    
    thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.85, 0.9, 0.99, 1.0]
    
    for thresh in thresholds:
        temp_pred = [1 if y >= thresh else 0 for y in y_pred]
        temp_tpr = true_positive_rate(y_true, temp_pred)
        temp_fpr = false_positive_rate(y_true, temp_pred)

        tpr_list.append(temp_tpr)
        fpr_list.append(temp_fpr)

    
    plt.figure(figsize = (7,7))
    plt.fill_between(fpr_list, tpr_list, alpha = 0.4)
    plt.plot(fpr_list, tpr_list, lw = 3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("graphs/ROC.png")