import numpy as np

def accuracy_score(y_true, y_predict):
    assert y_true.shape == y_predict.shape,\
        "the size of y_true must be equal to the size of y_predict"
    return (sum(y_true == y_predict) / len(y_true))