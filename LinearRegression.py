import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_epoch=1000):
        self.w = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
    
    def cost(self, X, y):
        """
        Computes Cost J(w,b) matching the slide formula:
        J = 1/(2m) * sum((pred - y)^2)
        """
        m = len(y)
        y_hat = self.w * X + self.b
        return (1/(2*m)) * np.sum((y_hat - y) ** 2)
        
    def fit(self, X, y): -> None
        """
        utilizes gradient decent algorithm to reach to a local minima where the cost fuction is near or equal to zero
        Args:
            X (_type_): nd array
            y (_type_): nd array
        """
        X = np.array(X)
        y = np.array(y)
        
        m = len(y)
        
        for epoch in range(self.n_epoch):
            y_hat = self.w * X + self.b
            errors = y_hat - y

            d_dw = (1/m) * np.dot(X, errors)
            d_db = (1/m) * np.sum(errors)
            
            self.w -= self.lr * d_dw
            self.b -= self.lr * d_db
        
            if epoch % 100 == 0:
                print(f"epoch {epoch}: Cost={self.compute_cost(X, y):.4f}, w={self.w:.3f}, b={self.b:.3f}")
    
    def predict(self, x): -> int
        return self.w * X + self.b

