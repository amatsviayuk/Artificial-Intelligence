import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()


# TODO: calculate closed-form solution      (1.13)

theta_best = [0, 0]
ones_col = np.ones_like(x_train)
X = np.c_[ones_col, x_train]
Y = np.c_[y_train]
brackets = np.linalg.inv(X.T @ X)
theta_best = brackets @ X.T @ Y
print("THETA = ")
print(theta_best)


# TODO: calculate error      (1.3)
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

actual_output = y_test  # (1.2)
predicted_output = theta_best[0] + theta_best[1] * x_test  # y = theta_0 + theta_1 * x
MSE = np.mean((actual_output - predicted_output) ** 2)  # m = x_train.size
print("MSE ERROR = ")
print(MSE)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()


# TODO: standardization (1.15)  z = (x - miu) / sigma; miu -- srednia z populacji(mean); sigma -- odchylenie standardowe(std)

x_train_pred = x_train
y_train_pred = y_train

x_train = (x_train - np.mean(x_train))/np.std(x_train)
y_train = (y_train - np.mean(y_train))/np.std(y_train)


# TODO: calculate theta using Batch Gradient Descent
ones_col = np.ones_like(x_train)
X = np.c_[ones_col, x_train]
Y = np.c_[y_train]

learning_rate = 0.01
max_epochs = 1000

m = x_train.size
theta = np.random.rand(2, 1)  # instruction

for epoch in range(max_epochs):
    grad = 2/m * X.T.dot((X.dot(theta) - Y))  # (1.7)
    theta -= learning_rate * grad  # (1.14)

    actual_output = y_train
    predicted_output = theta[0] + theta[1] * x_train
    MSE = np.mean((actual_output - predicted_output) ** 2)
    print(MSE)

theta = np.array(theta.T)[0]
print("THETA =  ")
print(theta)

theta_best = theta
x_train = x_train_pred
y_train = y_train_pred


# TODO: calculate error

y_test = (y_test - np.mean(y_train))/np.std(y_train)  # 1
x_test = (x_test - np.mean(x_train))/np.std(x_train)  # 2

actual_output = y_test  # y_test for test error
actual_output_restandardized = actual_output * np.std(y_train_pred) + np.mean(y_train_pred)  # 3 restandardize

predicted_output = theta_best[0] + theta_best[1] * x_test  # y_test_pred for test error
predicted_output_restandardized = predicted_output * np.std(y_train_pred) + np.mean(y_train_pred)  # restandardize

MSE = np.mean((actual_output_restandardized - predicted_output_restandardized) ** 2)  # 4 (1.3)
print("LAST ERROR ")
print(MSE)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()