# 1️⃣ Import libraries
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load dataset
california = fetch_california_housing()
X = california.data
y = california.target
feature_names = california.feature_names

# 3️⃣ Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5️⃣ Gradient Boosting with tuned parameters
gbr_model = GradientBoostingRegressor(
    n_estimators=400,      # more trees
    learning_rate=0.05,    # smaller steps
    max_depth=4,           # slightly deeper trees
    min_samples_split=5,   # require more samples to split
    random_state=42
)

# 6️⃣ Train
gbr_model.fit(X_train_scaled, y_train)

# 7️⃣ Predict
y_pred = gbr_model.predict(X_test_scaled)

# 8️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# 9️⃣ Feature importances
feat_imp = pd.DataFrame({'Feature': feature_names, 'Importance': gbr_model.feature_importances_})
print(feat_imp)