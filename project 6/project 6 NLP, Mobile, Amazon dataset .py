import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np

# =============== 1. Load dataset ===============
train = pd.read_csv('Documents/train (3).csv')

# Check data shape
print(f"Train shape: {train.shape}")
print(train.head(3))

# =============== 2. Prepare text and labels ===============
X_text = train['catalog_content']
y = train['price']

# =============== 3. TF-IDF + SVD (dimensionality reduction) ===============
vectorizer = TfidfVectorizer(max_features=8000, stop_words='english')
X_tfidf = vectorizer.fit_transform(X_text)

svd = TruncatedSVD(n_components=150, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# =============== 4. Split data ===============
X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.1, random_state=42)

# =============== 5. Train models ===============
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
lr = LinearRegression()

print("Training Gradient Boosting...")
gb.fit(X_train, y_train)

print("Training Linear Regression...")
lr.fit(X_train, y_train)

# =============== 6. Ensemble Predictions ===============
pred_gb = gb.predict(X_val)
pred_lr = lr.predict(X_val)

# Weighted average (you can tune weights)
y_pred = 0.7 * pred_gb + 0.3 * pred_lr

# =============== 7. Evaluate ===============
mae = mean_absolute_error(y_val, y_pred)
print(f"\nValidation MAE: {mae:.4f}")

