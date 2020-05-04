from sklearn import datasets
import matplotlib.pyplot as plt
from myPack.SimpleLinearRegression import SimpleLinearRegression2
from myPack.model_selection import train_test_split
from myPack.metrics import mean_squared_error
from myPack.metrics import root_mean_squared_error
from myPack.metrics import mean_absolute_error
from myPack.metrics import r2_score

boston = datasets.load_boston()

print(boston.DESCR)
print(boston.feature_names)
x = boston.data[:, 5]  # 只取 RM 房间数目为特征
y = boston.target

print(x.shape)
print(y.shape)

x = x[y < 50.0]
y = y[y < 50.0]

# plt.scatter(x, y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

print(x_train.shape)
print(y_train.shape)
reg = SimpleLinearRegression2()

reg.fit(x_train, y_train)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='r')
plt.show()

y_predict = reg.predict(x_test)

mse_test = mean_squared_error(y_test, y_predict)
rmse_test = root_mean_squared_error(y_test, y_predict)
mae_test = mean_absolute_error(y_test, y_predict)

print(mse_test)
print(rmse_test)
print(mae_test)

r2_test = r2_score(y_test, y_predict)

print(r2_test)

print(reg.score(x_test, y_test))
