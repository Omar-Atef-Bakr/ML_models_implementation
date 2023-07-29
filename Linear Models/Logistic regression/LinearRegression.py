import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LinearRegression:

    def __init__(self):
        self.weights = None
        self.bias = 0
        self.lr = 0.001

    def train(self, x: np.ndarray, y: np.ndarray, iter_limit = 1000, lr = 0.001):

        n_examples, n_parameters = x.shape

        self.weights = np.zeros(n_parameters)
        self.bias = 0
        self.lr = lr
        
        for _ in range(iter_limit):

            linpred = np.dot(x, self.weights) + self.bias
            pred = sigmoid(linpred)

            grad_step = np.dot(x.T, pred - y) / n_examples

            old_theta = self.weights

            self.weights -= self.lr * grad_step
            self.bias -= self.lr * (pred - y).sum() / n_examples


    def predict(self, x: np.ndarray):
        return sigmoid(x @ self.weights + self.bias) > 0.5