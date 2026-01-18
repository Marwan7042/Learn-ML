# Linear & Polynomial Regression from Scratch

A clean, educational implementation of Linear and Polynomial Regression in Python using **Gradient Descent**.

This project is designed to demystify how machine learning models "learn." Instead of using black-box libraries like Scikit-Learn, we build the optimization logic from the ground up using **NumPy**, focusing on clear variable naming and calculus-based concepts.

![Regression Results](regression_results.png)
*Visual comparison: The Linear model (red) underfits the curved data, while the Polynomial model (blue) captures the quadratic relationship.*

## 🎯 Key Features

* **No ML Libraries:** Logic is implemented purely with NumPy to understand the math.
* **Educational Naming:** Variables are named to reflect the calculus (e.g., `dJ_dw`, `dJ_db` for partial derivatives) rather than cryptic algebra.
* **Polynomial Support:** Shows how to fit non-linear curves using **Feature Engineering** and **Feature Scaling**.
* **Visualization:** Includes a Matplotlib script to compare Underfitting vs. Good Fitting.

## 🧠 How It Works

### 1. Linear Regression
The model tries to fit a straight line ($y = wx + b$) by minimizing the **Mean Squared Error (MSE)**.
* **Cost Function ($J$):** Measures the average squared difference between predictions and actual values.
* **Gradient Descent:** Iteratively updates the weights ($w$) and bias ($b$) by moving against the gradient of the error.

### 2. Polynomial Regression
To fit curves (like the parabola in the plot), we don't change the algorithm; we change the **data**.
* **Transformation:** We take input $x$ and create new features: $[x, x^2, x^3...]$.
* **Scaling (Crucial Step):** Since $x^2$ can be much larger than $x$, we normalize all features (Standardization) to keep Gradient Descent stable.

## 📦 Installation

You only need `numpy` for the math and `matplotlib` for the plotting.

```bash
pip install numpy matplotlib