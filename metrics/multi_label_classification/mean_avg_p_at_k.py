import precision_at_k

def mapk(true, pred, k):
    apk_values = []

    for i in range(len(true)):
        apk_values.append(precision_at_k.apk(true[i], pred[i], k))
    return sum(apk_values) / len(apk_values)

if __name__ == '__main__':

    y_true = [
         [1, 2, 3],
         [0, 2],
         [1],
         [2, 3],
         [1, 0],
         []
         ]
    y_pred = [
         [0, 1, 2],
         [1],
         [0, 2, 3],
         [2, 3, 4, 0],
         [0, 1, 2],
         [0]
         ]
    k = 4
    print(mapk(y_true, y_pred, k))