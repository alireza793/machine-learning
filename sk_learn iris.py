# 1. import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 2.load   Iris dataset
iris = load_iris()
X = iris.data   # ویژگی‌ها
y = iris.target # برچسب‌ها (سه کلاس گل‌ها)

# 3.split dataset between train  and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5. train model
model.fit(X_train, y_train)

# 6. predict labels on dataset
y_pred = model.predict(X_test)
print()

# 7. accuracy model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
