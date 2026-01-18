import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionGD:
    """
    Linear Regression optimized via Gradient Descent.
    Uses calculus-based naming (dJ_dw, dJ_db) for educational clarity.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000, num_features=1):
        # Model Parameters (Optimized during training)
        self.weights = np.zeros(num_features) 
        self.bias = 0.0
        
        # Hyperparameters (Set before training)
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def calculate_cost(self, X, y):
        """
        Computes Mean Squared Error (MSE).
        Formula: J = (1/2m) * Σ(y_pred - y_actual)^2
        """
        num_samples = len(y)
        predictions = np.dot(X, self.weights) + self.bias
        squared_errors = (predictions - y) ** 2
        return np.sum(squared_errors) / (2 * num_samples)
        
    def fit(self, X_train, y_train):
        """
        Trains the model by calculating gradients and updating parameters.
        """
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        num_samples, num_features = X_train.shape
        
        for iteration in range(self.num_iterations):
            # dJ_dw and dJ_db store the partial derivatives (the 'slope' of the error)
            dJ_dw = np.zeros(num_features)
            dJ_db = 0.0
            
            # 1. Calculate Gradients (Summing errors across all training examples)
            for i in range(num_samples):
                prediction = np.dot(self.weights, X_train[i]) + self.bias
                error = prediction - y_train[i]
                
                # Update partial derivative for each weight
                for j in range(num_features):
                    dJ_dw[j] += error * X_train[i, j]
                
                # Update partial derivative for the bias
                dJ_db += error
            
            # 2. Update Parameters (The "Gradient Descent" step)
            # We divide by num_samples to get the average gradient
            self.weights -= self.learning_rate * (dJ_dw / num_samples)
            self.bias -= self.learning_rate * (dJ_db / num_samples)
        
            if iteration % 100 == 0:
                current_cost = self.calculate_cost(X_train, y_train)
                print(f"Iteration {iteration}: Cost J = {current_cost:.4f}")
    
    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias


class PolynomialRegressionGD(LinearRegressionGD):
    """
    Polynomial Regression: Fits a non-linear relationship by creating 
    polynomial features (x^2, x^3...) and training a linear model on them.
    """
    
    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000):
        super().__init__(learning_rate=learning_rate, num_iterations=num_iterations, num_features=1)
        self.degree = degree
        self.training_mean = None
        self.training_std = None

    def _create_polynomial_features(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Generates [x^1, x^2, ..., x^degree]
        poly_features = [X**p for p in range(1, self.degree + 1)]
        return np.hstack(poly_features)

    def fit(self, X_train, y_train):
        # Transform: Add non-linear powers
        X_poly = self._create_polynomial_features(X_train)
        
        # Scale: Standardize features to prevent gradient explosion
        # 
        self.training_mean = np.mean(X_poly, axis=0)
        self.training_std = np.std(X_poly, axis=0)
        self.training_std[self.training_std == 0] = 1.0 # Prevent div by zero
        
        X_poly_scaled = (X_poly - self.training_mean) / self.training_std
        
        # Prepare weight vector for the new number of features
        num_samples, num_poly_features = X_poly_scaled.shape
        self.weights = np.zeros(num_poly_features)
        
        # Run parent Gradient Descent
        super().fit(X_poly_scaled, y_train)
        
    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        X_poly_scaled = (X_poly - self.training_mean) / self.training_std
        return super().predict(X_poly_scaled)
    
    
