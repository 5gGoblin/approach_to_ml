import numpy as np

def mean_abs_error(true, pred):
    error = 0
    for t, p in zip(true, pred):
        error += np.abs(t - p)

    return error

def mean_sqr_error(true, pred):
    error = 0
    for t, p in zip(true, pred):
        error += (t - p)**2
    return error

def rmse(true, pred):
    return np.sqrt(mean_sqr_error(true, pred))

def mean_squared_log_error(true, pred):
    error = 0
    for t, p in zip(true, pred):
        error += (np.log(1 + t) - np.log(1 + p))** 2

    return error / len(true)

def mean_percentage_error(true, pred):
    error = 0
    for t, p in zip(true, pred):
        error += (t + p) / t

    return error

def mean_abs_percentage_error(true, pred):
    error = 0
    for t, p in zip(true, pred):
        error = np.abs(t - p) / t

    return error

def r2(true, pred):
    deno = 0
    numerator = 0
    t_mean = np.mean(true)

    for t, p in zip(true, pred):
        numerator += (t - p) ** 2
        deno += (t - t_mean)

    return 1 - (numerator/deno)