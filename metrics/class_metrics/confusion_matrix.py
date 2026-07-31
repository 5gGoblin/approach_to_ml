

def true_positive(true, pred):
    tp = 0
    for t, p in zip(true, pred):
        if t == 1 and p == 1:
            tp += 1
    return tp

def true_negative(true, pred):
    tn = 0
    for t, p in zip(true, pred):
        if t == 0 and p == 0:
            tn += 1
    return tn

def false_positive(true, pred):
    fp = 0
    for t, p in zip(true, pred):
        if t == 0 and p == 1:
            fp += 1
    return fp

def false_negative(true, pred):
    fn = 0
    for t, p in zip(true, pred):
        if t == 1 and p == 0:
            fn += 1
    return fn
