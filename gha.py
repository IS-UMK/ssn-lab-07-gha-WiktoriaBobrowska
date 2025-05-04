import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import random

class GHA(object):
    def __init__(self, n_components=1, eta=0.001, n_epochs=100):
        self.eta = eta
        self.n_components = n_components
        self.n_epochs = n_epochs

    def init(self, X):
        n_features = X.shape[1]
        self.W = np.random.randn(self.n_components, n_features) * 0.01
        return self

    def fit(self, X):
        self.init(X)
        for epoch in range(self.n_epochs):
            for x in X:
                x = x.reshape(-1, 1)
                y = self.W @ x
                delta_W = self.eta * ((y @ x.T) - np.tril(y @ y.T) @ self.W)
                self.W += delta_W
        return self

    def transform(self, X):
        return (self.W @ X.T).T

    def inverse_transform(self, Y):
        return Y @ self.W
