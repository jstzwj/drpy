import numpy as np
import numpy.linalg

class LDA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.fit_impl(X, y)
        return self

    def fit_impl(self, X, y=None):
        

    def transform(self, X):
        return X
