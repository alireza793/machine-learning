# regression_example.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1) دادهٔ ساختگی می‌سازیم (می‌تونی این بخش رو با دادهٔ خودت جایگزین کنی)
X, y = make_regression(n_samples=200, n_features=1, noise=15.0, random_state=42)
# X شکل (n_samples, n_features) و y شکل (n_samples,)

# 2) تقسیم به مجموعهٔ آموزش/آزمون
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) (اختیاری اما معمول) استانداردسازی ویژگی‌ها
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 4) مدل رگرسیون (خطی ساده)
model = LinearRegression()
model.fit(X_train_s, y_train)

# 5) پیش‌بینی
y_pred_train = model.predict(X_train_s)
y_pred_test = model.predict(X_test_s)

# 6) ارزیابی
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("Train MSE: {:.4f}, R2: {:.4f}".format(mse_train, r2_train))
print("Test  MSE: {:.4f}, R2: {:.4f}".format(mse_test, r2_test))

# 7) ساخت DataFrame نتایج (Actual / Predicted / Residual) برای دادهٔ آزمون
df_results = pd.DataFrame({
    "X": X_test.flatten(),
    "y_true": y_test,
    "y_pred": y_pred_test,
    "residual": y_test - y_pred_test
})
# مرتب‌سازی روی X برای نمایش بهتر
df_results = df_results.sort_values(by="X").reset_index(drop=True)

# نمایش جدول (در نوتبوک این جدول به صورت زیبا نمایش داده می‌شود)
print(df_results.head(15))

# 8) نمودار: نقاط واقعی و خط پیش‌بینی
plt.figure(figsize=(8,6))
# نقاط واقعی
plt.scatter(df_results["X"], df_results["y_true"], label="Actual (test)", alpha=0.7)
# خط پیش‌بینی (برای نمایش خط بهتر، نقاط X مرتب شده)
plt.plot(df_results["X"], df_results["y_pred"], color="red", linewidth=2, label="Predicted (model)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 9) اگر خواستی جدول کامل رو در CSV ذخیره کنی:
df_results.to_csv("regression_results.csv", index=False)
print("Results saved to regression_results.csv"):
