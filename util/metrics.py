import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape == y_predict.shape,\
        "the size of y_true must be equal to the size of y_predict"
    return (sum(y_true == y_predict) / len(y_true))

def mean_squarred_error(y_true, y_predict):
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

def root_mean_squarred_error(y_true, y_oredict):
    return np.sqrt(mean_squarred_error(y_true, y_oredict))

def mean_absolute_error(y_true, y_predict):
    assert len(y_true) == len(y_predict),\
        "the size of y_true must be equal to the size of y_predict"
    return np.sum(np.abs(y_true - y_predict)) / len(y_true)