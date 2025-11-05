# ============================================================
# 🪐 Exoplanet Confirmation using Neural Networks and SMOTE
# Author: Alireza Hakemi
# University National of Skills, Iran
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# ============================================================
# 1️⃣ Load the Dataset
# ============================================================

# Dataset: NASA Kepler Exoplanet Search Results (Kaggle)
data = pd.read_csv("/ your path /file.csv")

# Select relevant numerical features
features = [
    'koi_teq', 'koi_insol', 'koi_prad', 'koi_period',
    'koi_steff', 'koi_slogg', 'koi_srad',
    'koi_depth', 'koi_duration'
]

target = 'koi_disposition'

# Keep only confirmed and not confirmed planets
data = data[data[target].isin(['CONFIRMED', 'FALSE POSITIVE'])]

# Map classes: CONFIRMED → 1, NOT CONFIRMED → 0
data[target] = data[target].map({'CONFIRMED': 1, 'FALSE POSITIVE': 0})

# Drop missing values
data = data.dropna(subset=features)

X = data[features]
y = data[target]

print(f"Samples: {len(X)}, Features: {X.shape[1]}")
unique, counts = np.unique(y, return_counts=True)
print("Class Distribution:", dict(zip(unique, counts)))

# ============================================================
# 2️⃣ Split into Train (70%), Validation (15%), Test (15%)
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train size: {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
print(f"Validation size: {len(X_val)} ({len(X_val)/len(X)*100:.2f}%)")
print(f"Test size: {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")

# ============================================================
# 3️⃣ Scale the Features
# ============================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4️⃣ Balance the Training Data using SMOTE
# ============================================================

sm = SMOTE(random_state=42, sampling_strategy=0.7)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

unique, counts = np.unique(y_train_res, return_counts=True)
print("After SMOTE:", dict(zip(unique, counts)))

# ============================================================
# 5️⃣ Train the Neural Network (MLP)
# ============================================================

mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                    learning_rate_init=0.001, max_iter=300, random_state=42)
mlp.fit(X_train_res, y_train_res)

# ============================================================
# 6️⃣ Evaluate on Validation Data
# ============================================================

y_val_pred = mlp.predict(X_val_scaled)
y_val_proba = mlp.predict_proba(X_val_scaled)[:, 1]

print("\n=== Validation Results ===")
print("Accuracy:", round(accuracy_score(y_val, y_val_pred), 3))
print("Precision:", round(precision_score(y_val, y_val_pred), 3))
print("Recall:", round(recall_score(y_val, y_val_pred), 3))
print("F1-Score:", round(f1_score(y_val, y_val_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_val, y_val_proba), 3))

# ============================================================
# 7️⃣ Final Evaluation on Test Data
# ============================================================

y_test_pred = mlp.predict(X_test_scaled)
y_test_proba = mlp.predict_proba(X_test_scaled)[:, 1]

print("\n=== Final Test Results ===")
print("Accuracy:", round(accuracy_score(y_test, y_test_pred), 3))
print("Precision:", round(precision_score(y_test, y_test_pred), 3))
print("Recall:", round(recall_score(y_test, y_test_pred), 3))
print("F1-Score:", round(f1_score(y_test, y_test_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_test_proba), 3))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_test_pred, target_names=["Not Confirmed", "Confirmed"]))

# ============================================================
# 8️⃣ Example Prediction
# ============================================================

example = np.array([[280, 1.0, 1.2, 365, 5800, 4.5, 1.0, 1000, 10]])
example_scaled = scaler.transform(example)
prediction = mlp.predict(example_scaled)[0]
probability = mlp.predict_proba(example_scaled)[0][1]

print("\nPrediction for Example:")
print("Confirmed Planet:", "Yes" if prediction == 1 else "No")
print("Probability of Confirmation:", round(probability, 3))

print("\n[Program finished]")
