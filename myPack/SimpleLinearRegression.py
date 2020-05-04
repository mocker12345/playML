import numpy as np
from myPack.metrics import r2_score


class SimpleLinearRegression1:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "The size of the x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        # 分子
        num = 0.0
        # 分母
        b = 0.0

        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            b += (x - x_mean) ** 2

        self.a_ = num / b
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        # 可以输入多个预测值
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "muse fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
            "The size of the x_train must be equal to the size of y_train"
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        # 向量化
        num = (x_train - x_mean).dot((y_train - y_mean))
        b = (x_train - x_mean).dot((x_train - x_mean))

        self.a_ = num / b
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        # 可以输入多个预测值
        assert x_predict.ndim == 1, \
            "Simple Linear Regression can only solve single feature training data"
        assert self.a_ is not None and self.b_ is not None, \
            "muse fit before predict"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """根据x_test y_test 确定当前模型的准确度"""
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression1()"
