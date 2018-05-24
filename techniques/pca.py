import numpy as np
import numpy.linalg

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        self.fit_impl(X, y)
        return self

    def fit_impl(self, X, y=None):
        self.fit_svd(X, y)
    def fit_eig(self, X, y=None):
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        w, v = np.linalg.eig(np.cov(X))
        idx = np.argsort(w)
        components_ = w[idx]

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:self.n_components]

        
    def fit_svd(self, X, y=None):
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        _, _, V = numpy.linalg.svd(X, full_matrices=False)
        components_ = V

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:self.n_components]

    def transform(self, X):
        return X
