import numpy as np
import matplotlib.pyplot as plt

from Regression import LinearRegressionGD
from Regression import PolynomialRegressionGD


print("Generating synthetic data...")

# --- Step 1: Generate Data ---
# We create a simple quadratic curve: y = 0.5 * x^2 + noise
np.random.seed(42)
m = 30
X_train = np.random.rand(m, 1) * 6 - 3  # Random X between -3 and 3
y_train = 0.5 * X_train.flatten()**2 + X_train.flatten() + 2 + np.random.randn(m) * 0.5

# --- Step 2: Run Linear Regression (The "Bad" fit) ---
print("\nTraining Linear Regression...")
lin_model = LinearRegressionGD(learning_rate=0.01, num_iterations=500)
lin_model.fit(X_train, y_train)

# --- Step 3: Run Polynomial Regression (The "Good" fit) ---
print("\nTraining Polynomial Regression (Degree 2)...")
poly_model = PolynomialRegressionGD(degree=2, learning_rate=0.1, num_iterations=500)
poly_model.fit(X_train, y_train)

# --- Step 4: Visualization ---
# Create a smooth line of X values for plotting the models' predictions
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)

y_lin_pred = lin_model.predict(X_plot)
y_poly_pred = poly_model.predict(X_plot)



plt.figure(figsize=(10, 6))

# Plot the actual training data
plt.scatter(X_train, y_train, color='black', label='Training Data')

# Plot Linear Regression Result
plt.plot(X_plot, y_lin_pred, color='red', linestyle='--', label='Linear Fit (Underfitting)')

# Plot Polynomial Regression Result
plt.plot(X_plot, y_poly_pred, color='blue', linewidth=2, label=f'Polynomial Fit (Degree {poly_model.degree})')

plt.title("Linear vs. Polynomial Regression (Gradient Descent from Scratch)")
plt.xlabel("Input Feature (X)")
plt.ylabel("Target Value (y)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()