import numpy as np
import confusion_matrix as cm
from collections import Counter
from sklearn import metrics



def macro_precision(true, pred):
    '''
    macro-average precision: calculate precision for all classes individually
    and then average them
    '''

    num_classes = len(np.unique(true))
    precision = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in true]
        temp_pred = [1 if p == class_ else 0 for p in pred]

        fp = cm.false_positive(temp_true, temp_pred)
        tp = cm.true_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)

        precision += temp_precision
    precision /= num_classes
    return precision

def micro_precision(true, pred):
    '''
    micro-average precision: calculate class wise true positive and false positive
    and then use that to calculate overall precision
    '''

    tp = 0
    fp = 0
    num_classes = len(np.unique(true))

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in true]
        temp_pred = [1 if p == class_ else 0 for p in pred]

        tp += cm.true_positive(temp_true, temp_pred)
        fp += cm.false_positive(temp_true, temp_pred)

    precision = tp / (tp + fp)
    return precision

def weighted_precision(true, pred):
    '''
    Weighted precision: same as macro but in this case it is weighted average
    depending on the number of items in each class
    '''

    num_classes = len(np.unique(true))
    class_count = Counter(true)
    precision = 0

    for class_ in range(num_classes):
        temp_true = [1 if p == class_ else 0 for p in true]
        temp_pred = [1 if p == class_ else 0 for p in pred]

        tp = cm.true_positive(temp_true, temp_pred)
        fp = cm.false_positive(temp_true, temp_pred)

        temp_precision = tp / (tp + fp)

        weighted_precision = class_count[class_] * temp_precision
        precision += weighted_precision
    
    overall_precision = precision / len(true)

    return overall_precision

if __name__ == "__main__":
    y_true = [0,1,2,0,1,2,0,2,2]
    y_pred = [0,2,1,0,2,1,0,0,2]

    macro = macro_precision(y_true, y_pred)
    sklearn_macro = metrics.precision_score(y_true, y_pred, average = 'macro')

    micro = micro_precision(y_true, y_pred)
    sklearn_micro = metrics.precision_score(y_true, y_pred, average = 'micro')

    weighted = weighted_precision(y_true, y_pred)
    sklearn_weighted = metrics.precision_score(y_true, y_pred, average = 'weighted')

    print(f"Macro: {macro} \nSklearn Macro: {sklearn_macro}")
    print(f"\nMicro: {micro} \nSklearn Micro: {sklearn_micro}")
    print(f"\nWeighted: {weighted} \nSklearn Weighted: {sklearn_weighted}")