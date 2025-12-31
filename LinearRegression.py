import numpy as np

class LinearRegressionGD:
    """
    A simple Linear Regression model using Gradient Descent.
    
    This implementation uses iterative loops to calculate gradients,
    supporting multiple features (Multivariate Linear Regression).
    """

    def __init__(self, learning_rate=0.01, n_epoch=1000, n_features=1):
        """
        Initializes the model hyperparameters and parameters.

        Args:
            learning_rate (float): The step size for gradient descent (alpha). 
                                   Default is 0.01.
            n_epoch (int): The number of times to loop through the entire dataset. 
                           Default is 1000.
            n_features (int): The number of input features (variables) in the data.
        """
        # Model Parameters
        self.w = np.zeros(n_features) # Weights (slope), initialized to 0
        self.b = 0                    # Bias (intercept), initialized to 0
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.n_epoch = n_epoch
    
    def cost(self, X, y):
        """
        Computes the Mean Squared Error (MSE) cost function J(w,b).
        
        Formula: J = 1/(2m) * sum((prediction - actual)^2)

        Args:
            X (array-like): Training data of shape (m, n).
            y (array-like): Target values of shape (m,).

        Returns:
            float: The computed cost value.
        """
        errors = 0.0
        m = len(y)
        
        for i in range(m):
            # Prediction for single example i
            y_hat = np.dot(self.w, X[i]) + self.b
            
            # Accumulate squared error
            errors += (y_hat - y[i]) ** 2
            
        return errors / (2 * m)
        
    def fit(self, X, y):
        """
        Trains the linear regression model using Gradient Descent.

        Updates the weights (self.w) and bias (self.b) to minimize the cost.

        Args:
            X (array-like): Training matrix of shape (m samples, n features).
            y (array-like): Target vector of shape (m samples,).
        """
        X = np.array(X)
        y = np.array(y)
        
        m, n = X.shape # m = number of samples, n = number of features
        
        for epoch in range(self.n_epoch):
            # Initialize gradients to zero at start of each epoch
            d_dw = np.zeros(n)
            d_db = 0.
            
            # Iterate through each training example
            for i in range(m):
                y_hat = np.dot(self.w, X[i]) + self.b
                err = y_hat - y[i]
                
                # Calculate gradients for each feature j
                for j in range(n):
                    d_dw[j] += err * X[i, j]  
                d_db += err
            
            # Update parameters using the average gradient
            # Formula: w = w - alpha * (1/m * dJ/dw)
            self.w -= self.learning_rate * (d_dw / m)
            self.b -= self.learning_rate * (d_db / m)
        
            # Log progress every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost={self.cost(X, y):.4f}, w={self.w}, b={self.b:.3f}")
    
    def predict(self, x):
        """
        Predicts target values for given input data.

        Args:
            x (array-like): Input data. Can be a single row (n_features,) 
                            or a matrix (m_samples, n_features).

        Returns:
            array-like: Predicted values.
        """
        x = np.array(x)
        return np.dot(x, self.w) + self.b