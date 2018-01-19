import numpy as np
class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ----------
    eta : float
        Learning rate(between 0.0 and 1.0)
    n_iter: int
        Passes over the training set.(Epochs)
        
    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting
    errors_ : list
        Number of misclassifications in every epoch, y_true != y_pred
        
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        """Fit training data.
        
        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : {array-like}, shape = [n_samples]
            Target values.
            
        Returns
        -------
        self : object
        
        """
        self.w_ = np.zeros(1 + X.shape[1])                         # Plus 1 for the threshold
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))    # update(delta w) <---  learning_rate * (y_true - y_pred)
                self.w_[1:] += self.eta * (target - self.predict(xi)) * xi
                self.w_[0]  += update
                errors += int(update != 0.0)
            self.errors_.append(errors)                            # 紀錄每次分類錯誤筆數
        return self
            
    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]           # np.dot --> W.T*X(inner product)
    
    def predict(self, X):                                    # activation function 
        """Return class label after unit step."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)     # return class label
