# 1. وارد کردن کتابخانه‌ها
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2. بارگذاری دیتاست Iris
iris = load_iris()
X = iris.data   # ویژگی‌ها
y = iris.target # برچسب‌ها (سه کلاس گل‌ها)

# 3. تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. ساخت مدل Random Forest
"""
Logistic Regression

مسائل دوتایی یا چندکلاسه ساده

وقتی رابطه بین ویژگی‌ها و کلاس‌ها تقریباً خطی باشد

مثال: پیش‌بینی اسپم/غیر اسپم


۲. K-Nearest Neighbors (KNN)

وقتی شباهت بین نمونه‌ها مهم باشد

داده‌ها کم حجم و ویژگی‌ها زیاد نباشند

مثال: سیستم پیشنهاد فیلم یا تشخیص دست‌خط


۳. Support Vector Machine (SVM)

مسائل پیچیده با مرز مشخص بین کلاس‌ها

داده‌ها خطی یا غیرخطی با kernel

مثال: تشخیص تصویر، طبقه‌بندی ژن


۴. Naive Bayes

داده‌های متنی یا دسته‌ای

وقتی ویژگی‌ها تقریباً مستقل باشند

مثال: تحلیل احساسات متن


۵. Gradient Boosting / XGBoost / LightGBM

داده‌های بزرگ و پیچیده

وقتی دقت بسیار بالا بخواهیم

مثال: پیش‌بینی خرید مشتری، Kaggle competitions

💡 انتخاب مدل به اندازه داده، نوع مسئله، دقت مورد نیاز و زمان اجرا بستگی دارد.
چ
"""
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. آموزش مدل
model.fit(X_train, y_train)

# 6. پیش‌بینی برچسب‌ها روی داده‌های تست
y_pred = model.predict(X_test)
print()

# 7. ارزیابی مدل
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))