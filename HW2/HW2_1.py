import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def fit(self, X):
        n_samples, n_features = X.shape

        # 1. Initialization
        self.weights_ = np.ones(self.n_components) / self.n_components
        random_idx = np.random.choice(
            n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx]
        self.covariances_ = [np.cov(X, rowvar=False)] * self.n_components

        log_likelihood = 0
        for _ in range(self.max_iter):
            # 2. E-Step
            responsibilities = self._e_step(X)

            # 3. M-Step
            self._m_step(X, responsibilities)

            # Compute the new log-likelihood
            new_log_likelihood = self._compute_log_likelihood(X)

            # Check convergence. If the new likelihood exceeds the tolerance, break the loop.
            # Use L1 norm for distance computation.
            if np.abs(new_log_likelihood - log_likelihood) < self.tol:
                break

            log_likelihood = new_log_likelihood

    def _e_step(self, X):
        responsibilities = np.zeros((X.shape[0], self.n_components))

        for i in range(self.n_components):
            # Update the responsibility based on the Gaussian PDF
            responsibilities[:, i] = self.weights_[
                i] * multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i])

        # Normalize the responsibilities
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        return responsibilities

    def _m_step(self, X, responsibilities):
        for i in range(self.n_components):
            # Update weights for each component
            total_responsibility = np.sum(responsibilities[:, i])
            self.weights_[i] = total_responsibility / X.shape[0]

            # Update means for each component
            self.means_[i] = np.sum(
                responsibilities[:, i][:, np.newaxis] * X, axis=0) / total_responsibility

            # Update covariances for each component
            diff = X - self.means_[i]
            self.covariances_[i] = np.dot(
                (responsibilities[:, i][:, np.newaxis] * diff).T, diff) / total_responsibility

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for i in range(self.n_components):
            log_likelihood += self.weights_[i] * multivariate_normal.pdf(
                X, mean=self.means_[i], cov=self.covariances_[i])

        final_likelihood = np.sum(np.log(log_likelihood))
        return final_likelihood

    def predict(self, X):
        cluster_idx = np.argmax(self._e_step(X), axis=1)
        return cluster_idx
