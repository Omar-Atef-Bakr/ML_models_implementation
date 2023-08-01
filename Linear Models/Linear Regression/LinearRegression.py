import numpy as np
import pandas as pd



class LinearRegression:

    def __init__(self) -> None:
        self.theta = None
        self.bias = 0
        self.history = []

    def fit(self, x, t, lr = 0.001, iter_limit=1000):

        n_samples, n_features = x.shape

        self.theta = np.zeros(n_features)
        self.bias = 0


        for i in range(iter_limit):

            # calculate prediction
            y = np.dot(x, self.theta) + self.bias    

            # calc gradient steps
            diff = t - y
            grad_step = x.T @ (y - t) / n_samples
            bias_step = (y - t).sum() / n_samples

            # update history
            self.history.append(np.linalg.norm(t-y, 2))

            # update thetas and bias
            self.theta -= lr * grad_step
            self.bias -= lr * bias_step

            if i > 0 and i % 10 == 0:
                if abs(np.linalg.norm(t-y, 2) - self.history[-2]).sum() <= 10 ** -3:
                    break

    def predict(self, x):
        return np.dot(x, self.theta) + self.bias



model = LinearRegression()
x = np.array([[1,2],[3,4],[1,4]])
y = np.array([1,2,3])
model.fit(x, y)
print(model.theta)







