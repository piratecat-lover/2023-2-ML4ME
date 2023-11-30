import numpy as np
from collections import defaultdict

class GaussianKDEParzenWindow:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.data = None
    
    def gaussian_kernel(self, u):
        #TODO: Implement Gaussian Kernel
        d=len(u)
        pdf=1/(2*np.pi)**(d/2)*np.exp(-0.5*u.T@u)
        return pdf
    
    def fit(self, data):
        self.data = np.array(data)
    
    def evaluate(self, x):
        N = self.data.shape[0]
        x = np.array(x)
        density = 0
        # TODO: Given new datapoint, for each existing data return probability of the new data point. 
        for i in range(N):
            v=(x-self.data[i])/self.bandwidth
            density+=self.gaussian_kernel(v)
        prob=density/N
        return prob
    
    def __call__(self, x):
        return self.evaluate(x)

class KDEClassifier:
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.kde_by_class = defaultdict(GaussianKDEParzenWindow)
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        # TODO: For each class, fit the data that belongs to Gaussian kernel per class.
        # then assign the probabliy to the kde-by_class dictionary. 
        
        for c in self.classes:
            kde = GaussianKDEParzenWindow(bandwidth=self.bandwidth)
            kde.fit(X[y==c])
            self.kde_by_class[c] = kde
    
    def predict(self, X):
        predicted_classes = []
        # TODO: Based on density, predict the most probable class based on the acquired probability. 
        # return array with items containing the predicted class for the given data point. 
        for x in X:
            prob = []
            for c in self.classes:
                prob.append(self.kde_by_class[c](x))
            predicted_classes.append(self.classes[np.argmax(prob)])
        return np.array(predicted_classes)
