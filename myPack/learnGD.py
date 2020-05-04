import numpy as np
import matplotlib.pyplot as plt

plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5) ** 2 - 1

plt.plot(plot_x, plot_y)
plt.show()


def dj(theta):
    return 2 * (theta - 2.5)


def J(theta):
    return (theta - 2.5) ** 2 - 1


eta = 0.01
theta = 0.0
epsilon = 1e-8
theta_history = [theta]
while True:
    gradient = dj(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    if abs(J(theta) - J(last_theta)) < epsilon:
        break

print(theta)
print(J(theta))

plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()
