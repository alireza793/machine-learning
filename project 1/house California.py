# 1️⃣ Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load the dataset
california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names

# 3️⃣ Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Create and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# 5️⃣ Make predictions
y_pred = rf_model.predict(X_test)

# 6️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# 7️⃣ Show feature importances
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': rf_model.feature_importances_})
print(feat_imp)