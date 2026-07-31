def pk(true, pred, k):
    if k < 1:
        return 0
    pred = pred[:k]

    pred_set = set(pred)
    true_set = set(true)

    common_values = pred_set.intersection(true_set)

    return len(common_values) / len(pred[:k])

def apk(true, pred, k):
    pk_values = []

    for i in range(1, k+1):
        pk_values.append(pk(true, pred, i))
    
    if len(pk_values) == 0:
        return 0
    return sum(pk_values) / len(pk_values)

y_true = [[1, 2, 3],
          [0, 2],
          [1], 
          [2, 3], 
          [1, 0], 
          []]
y_pred = [[0, 1, 2],
          [1],
          [0, 2, 3],
          [2, 3, 4, 0],
          [0, 1, 2],
          [0]]

if __name__ == '__main__':

    k = 4
    for i in range(len(y_true)):
        for j in range(1, k):
            print(f"y_true = {y_true[i]} \ny_pred = {y_pred[i]} \nAP@{j} = {apk(y_true[i], y_pred[i], j)}\n")