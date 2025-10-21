import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load your data

df = pd.read_csv("/storage/emulated/0/Earthquake.csv")  # full path

# Features and target
X = df[['depth', 'latitude', 'longitude', 'mmi', 'sig', 'nst', 'gap', 'dmin', 'tsunami']]
y = df['magnitude']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model'':()
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Predicted vs Actual Magnitude")
plt.show()