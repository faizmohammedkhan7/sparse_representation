import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Loading data
URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
df = pd.read_csv(URL, header=None)

# selecting a single feature and target label
# using 100 instances for simplicity
X = df.iloc[:100, 5].values.reshape(-1, 1)  # Reshape directly using NumPy
y = df.iloc[:100, 13].values.reshape(-1, 1)  # Reshape directly using NumPy

# instantiating the lasso regression model
lasso = Lasso(alpha=10)

# training the model
lasso.fit(X, y)

# making predictions
y_pred = lasso.predict(X)

# evaluating the model
mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"Model Coefficients: {lasso.coef_}")

# plotting the line of best fit
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df.iloc[:100, 5], y=df.iloc[:100, 13])  # Scatter plot using seaborn
plt.plot(X, y_pred, color="red")
plt.title("Linear Regression Model with L1 Regularization (Lasso)")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()
