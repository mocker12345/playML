import numpy as np
import matplotlib.pyplot as plt
from myPack.SimpleLinearRegression import SimpleLinearRegression1
from myPack.SimpleLinearRegression import SimpleLinearRegression2

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

x_predict = 6.0

reg1 = SimpleLinearRegression1()

reg1.fit(x, y)

print(reg1.predict(np.array([x_predict])))
print(reg1.a_)
print(reg1.b_)
y_hat = reg1.predict(x)

plt.scatter(x, y)
plt.plot(x, y_hat, color='r')
plt.axis([0, 6, 0, 6])
plt.show()

reg2 = SimpleLinearRegression2()
reg2.fit(x, y)
print(reg2.predict(np.array([x_predict])))

big_num = 1000000

big_x = np.random.random(size=big_num)
big_y = big_x * 2.0 * 3.0 + np.random.normal(size=big_num)

# %timeit reg1.fit(big_x,big_y)
# 1 loop, best of 3: 748 ms per loop
# %timeit reg2.fit(big_x,big_y)
# 10 loops, best of 3: 14.9 ms per loop
