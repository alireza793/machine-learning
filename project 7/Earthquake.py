import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------
# 1. File path
# -------------------------------
path = r"Earthquak.csv"


# -------------------------------
# 2. Load data
# -------------------------------
df = pd.read_csv(path)
print("✅ File loaded successfully")
print(df.head())

# -------------------------------
# 3. Features and target
# -------------------------------
features = ['magnitude', 'depth', 'latitude', 'longitude', 'mmi', 'sig', 'nst', 'gap', 'dmin']
X = df[features]
y = df['tsunami']

# -------------------------------
# 4. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 5. Feature scaling (for Logistic Regression)
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 6. Ensemble model: Logistic Regression + Random Forest
# -------------------------------
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000, class_weight='balanced')),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ],
    voting='soft'
)

# Fit model
ensemble.fit(X_train_scaled, y_train)

# -------------------------------
# 7. Predict and evaluate
# -------------------------------
y_pred = ensemble.predict(X_test_scaled)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

#plt.show()

# -------------------------------
# 8. Feature importance (from Random Forest)
# -------------------------------
rf_model = ensemble.named_estimators_['rf']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8,5))
plt.barh([features[i] for i in indices], importances[indices])
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()
