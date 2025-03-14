import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes

# Load the dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train models and evaluate performance
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MSE": mse, "R² Score": r2}

# Convert results to DataFrame for easy visualization
results_df = pd.DataFrame(results).T

# Display results
print("\nModel Performance:")
print(results_df)

# Determine the best model based on R² score
best_model = results_df["R² Score"].idxmax()
print(f"\nBest Performing Model: {best_model}")

# Plot actual vs predicted values for best model
best_model_instance = models[best_model]
y_pred_best = best_model_instance.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_best, alpha=0.7)
plt.xlabel("Actual Diabetes Progression")
plt.ylabel("Predicted Diabetes Progression")
plt.title(f"Actual vs Predicted Values ({best_model})")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', linestyle='dashed')
plt.show()

# Brief Summary
summary = f"\nThe best-performing model is {best_model}. The performance metrics indicate that it has the highest R² score, meaning it explains the largest variance in the data. "
if best_model == "Random Forest":
    summary += "Random Forest performed the best because it captures complex, non-linear relationships in the dataset, reducing prediction error. However, it is computationally expensive."
elif best_model == "Linear Regression":
    summary += "Linear Regression is the best model, suggesting that the dataset follows a relatively linear trend. It is simple and interpretable but may underperform if the data has complex interactions."
elif best_model == "Ridge Regression":
    summary += "Ridge Regression performed best, likely due to regularization, which helps prevent overfitting while still maintaining good predictive power."

print(summary)
