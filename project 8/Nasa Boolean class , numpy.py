import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# -----------------------------
# 1. خواندن داده‌ها
df = pd.read_csv(r"cumulative.csv/cumulative.csv")
features = [
    'koi_teq', 'koi_insol', 'koi_prad', 'koi_period',
    'koi_steff', 'koi_slogg', 'koi_srad', 'koi_depth', 'koi_duration'
]

# هدف: CONFIRMED یا نه
df['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
df_clean = df[features + ['target']].dropna()
X = df_clean[features].values
y = df_clean['target'].values

print(f"نمونه‌ها: {X.shape[0]}, ویژگی‌ها: {X.shape[1]}")
print(f"توزیع کلاس‌ها: {np.unique(y, return_counts=True)}")

# -----------------------------
# 2. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"توزیع داده آموزش: {np.bincount(y_train)}")
print(f"توزیع داده تست: {np.bincount(y_test)}")

# -----------------------------
# 3. نرمال‌سازی
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. SMOTE با حجم کمتر
smote = SMOTE(sampling_strategy=0.5, random_state=42)  # کلاس اقلیت را نصف کلاس اکثریت
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
print(f"پس از SMOTE: {np.bincount(y_train_smote)}")

# -----------------------------
# 5. شبکه عصبی ساده با Numpy
class SimpleNN:
    def __init__(self, input_size, hidden_size=8, lr=0.01):
        self.lr = lr
        # وزن‌ها و بایاس‌ها
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def forward(self, X):
        self.Z1 = X.dot(self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1.dot(self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, y_pred, y_true):
        y_true = y_true.reshape(-1, 1)
        loss = -np.mean(y_true*np.log(y_pred+1e-8) + (1-y_true)*np.log(1-y_pred+1e-8))
        return loss

    def backward(self, X, y):
        y = y.reshape(-1,1)
        m = X.shape[0]

        dZ2 = self.A2 - y
        dW2 = self.A1.T.dot(dZ2) / m
        db2 = np.sum(dZ2, axis=0) / m

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)
        dW1 = X.T.dot(dZ1) / m
        db1 = np.sum(dZ1, axis=0) / m

        # آپدیت وزن‌ها
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs=500, batch_size=64):
        for epoch in range(1, epochs+1):
            # Shuffle
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            # Mini-batch
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                xb, yb = X[start:end], y[start:end]
                y_pred = self.forward(xb)
                self.backward(xb, yb)
            
            if epoch % 50 == 0:
                y_pred_full = self.forward(X)
                loss = self.compute_loss(y_pred_full, y)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")

    def predict(self, X, threshold=0.5):
        return (self.forward(X) >= threshold).astype(int)

    def predict_proba(self, X):
        return self.forward(X)

# -----------------------------
# 6. آموزش
nn = SimpleNN(input_size=X_train_smote.shape[1], hidden_size=12, lr=0.01)
nn.train(X_train_smote, y_train_smote, epochs=10000, batch_size=64)

# -----------------------------
# 7. پیش‌بینی و ارزیابی
y_pred = nn.predict(X_test_scaled)
y_proba = nn.predict_proba(X_test_scaled)

print("\n=== نتایج شبکه عصبی ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.3f}")
print(f"F1-Score: {2 * (precision_score(y_test, y_pred, zero_division=0) * recall_score(y_test, y_pred, zero_division=0)) / 
(precision_score(y_test, y_pred, zero_division=0) + recall_score(y_test, y_pred, zero_division=0) + 1e-8):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Confirmed', 'Confirmed'], zero_division=0))

# -----------------------------
# 8. مثال پیش‌بینی
example_data = [280, 1.0, 1.2, 365, 5800, 4.5, 1.0, 1000, 10]
example_scaled = scaler.transform([example_data])
pred_example = nn.predict(example_scaled)[0]
proba_example = nn.predict_proba(example_scaled)[0][0]

print("\nپیش‌بینی برای نمونه:")
print(f"سیاره تایید شده: {'بله' if pred_example == 1 else 'خیر'}")
print(f"احتمال تایید: {proba_example:.3f}")