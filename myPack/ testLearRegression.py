import numpy as np
from myPack.LinearRegression import LinearRegression
from myPack.model_selection import train_test_split
from sklearn import datasets

boston = datasets.load_boston()
X = boston.data
y = boston.target
X = X[y < 50]
y = y[y < 50]

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

print(X_train.shape, y_train.shape)

reg = LinearRegression()
reg.fit_normal(X_train, y_train)
print(reg.coef_)
print(reg.predict(X_test))
print(reg.score(X_test, y_test))

# 使用梯度下降法


print(X_train[:10, :])  # 数据规模不一
lin_reg2 = LinearRegression()
# 太耗时 lin_reg2.fit_gd(X_train, y_train, eta=0.000001, n_iters=1000000)

# 使用梯度下降前进行数据归一化

from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
standardScaler.fit(X_train)

X_train_standard = standardScaler.transform(X_train)

lin_reg3 = LinearRegression()
lin_reg3.fit_gd(X_train_standard, y_train)
X_test_standard = standardScaler.transform(X_test)
print(lin_reg3.score(X_test_standard, y_test))

# sklearn
from sklearn.linear_model import LinearRegression as SkLinearRegression

reg1 = SkLinearRegression()
reg1.fit(X, y)
coef = reg1.coef_
print(np.argsort(coef))
# 数据可解释性
print(boston.feature_names[np.argsort(coef)])

# 使用sklearn 随机梯度下降法
from sklearn.linear_model import SGDRegressor

standardScaler = StandardScaler()
standardScaler.fit(X_train)

X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

# n_iter = 5 默认
sgd_reg = SGDRegressor(n_iter=100)
sgd_reg.fit(X_train_standard, y_train)
print(sgd_reg.score(X_test_standard, y_test))
