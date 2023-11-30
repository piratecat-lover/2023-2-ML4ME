import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
    
    
    def _predict(self, x):
        ## TODO: Similar to what we did in discussion, implement KNN
        distances = np.linalg.norm(self.X_train-x,axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

class NaiveBayesClassifier:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[y==c]
            self._mean[c] = X_c.mean(axis=0)
            self._var[c] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        
        for idx, c in enumerate(self._classes):
            prior =  np.log(self._priors[c]) # For each class, compute the log prior,
            class_conditional = np.sum(np.log(self._pdf(idx,x))) # Sum of log pdf 
            posterior = prior+class_conditional# Compute posterior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        pdf=1/np.sqrt((2*np.pi*var))*np.exp(-(x-mean)**2/(2*var))
        # Return gaussian pdf
        return pdf