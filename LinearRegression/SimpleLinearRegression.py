import numpy as np

class SimpleLinearRegressionV1:
    
    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1,\
            "needs single feature train data"
        assert len(x_train) == len(y_train),\
            "the size of x and y must be equals"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = np.sum((x_train - x_mean) * (y_train - y_mean)) / np.sum((x_train - x_mean) ** 2)
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert isinstance(x_predict, (int, float)) or x_predict.ndim == 1,\
            "needs single feature data to predict"
        assert self.a_ is not None and self.b_ is not None,\
            "must be fit before oredict"
        return self.a_ * x_predict + self.b_

    def __repr__(self):
        return "SimpleLinearRegressionV1()"

class SimpleLinearRegressionV2:
    
    def __init__(self):
        self.a_ = None
        self.b_ = None
    
    def fit(self, x_train, y_train):
        assert x_train.ndim == 1 and y_train.ndim == 1,\
            "needs single feature train data"
        assert len(x_train) == len(y_train),\
            "the size of x and y must be equals"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        self.a_ = (x_train - x_mean).dot(y_train - y_mean) / (x_train - x_mean).dot((x_train - x_mean))
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert isinstance(x_predict, (int, float)) or x_predict.ndim == 1,\
            "needs single feature data to predict"
        assert self.a_ is not None and self.b_ is not None,\
            "must be fit before oredict"
        return self.a_ * x_predict + self.b_

    def __repr__(self):
        return "SimpleLinearRegressionV2()"