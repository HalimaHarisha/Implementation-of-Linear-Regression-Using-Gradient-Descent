# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize parameters (slope and intercept) with small values and choose a learning rate.
2.Compute predicted profit using the linear equation for all training data points.
3.Calculate the cost (error) and update parameters using gradient descent to minimize the error.
4.Repeat the process until convergence and use the final model to predict profit. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: A.Halima Harisha 
RegisterNumber: 212224040094
*/
import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
m = 0  # slope
b = 0  # intercept
learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db
print("Slope (m):", m)
print("Intercept (b):", b)
y_pred = m * X + b
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="1035" height="862" alt="image" src="https://github.com/user-attachments/assets/b8ff05f1-43f6-4acf-8896-b3030e8c641d" />


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
