from sklearn.linear_model import Lasso
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate sample data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create Lasso model with L1 regularization
lasso = Lasso(alpha=0.1)  # Alpha controls the strength of regularization

# Fit the model
lasso.fit(X_scaled, y)

# Get the coefficients (representing feature importance)
coefficients = lasso.coef_

# Print coefficients
print("Coefficients:", coefficients)

# Visualize coefficients using a bar plot
plt.figure(figsize=(8, 6))
plt.bar(range(len(coefficients)), coefficients)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficients of Lasso Model')
plt.show()
