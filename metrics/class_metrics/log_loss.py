import numpy as np
from sklearn import metrics
#log loss = -1.0 * (target * log(prediction) + (1-target) * log(1-prediction))

def log_loss(true, prob):
    eps = 1e-10
    loss = []

    for t, p in zip(true, prob):
        p = np.clip(p, eps, 1-eps)
        temp_loss = -1.0 * (t * np.log(p) + (1 - t) * np.log(1 - p))
        loss.append(temp_loss)

    return np.mean(loss)

if __name__ == "__main__":
    y_true = [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1]
    y_prob = [0.1, 0.3, 0.2, 0.6, 0.8, 0.05, 0.9, 0.5, 0.3, 0.66, 0.3, 0.2, 0.85, 0.15, 0.99]

    loss = log_loss(y_true, y_prob)
    sk_loss = metrics.log_loss(y_true, y_prob)
    print(f"Implementation: {loss:.5f}")
    print(f"Sklearn: {sk_loss:.5f}")