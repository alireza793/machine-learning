# 1️⃣ Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load the dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 3️⃣ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Create base models
estimators = [
    ('lr', LinearRegression()),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbr', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))
]

# 5️⃣ Create stacking model
stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# 6️⃣ Train the model
stack_model.fit(X_train, y_train)

# 7️⃣ Make predictions
y_pred = stack_model.predict(X_test)

# 8️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")