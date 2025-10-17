import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. خواندن داده‌ها
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv('Fake.csv')

# 2. افزودن برچسب
true_df["label"] = 1   # خبر درست
fake_df["label"] = 0   # خبر غلط

# 3. ترکیب دیتافریم‌ها
df = pd.concat([true_df, fake_df], ignore_index=True)

# فرض می‌کنیم متن خبر در ستونی به نام 'text' است
X = df["text"]
y = df["label"]

# 4. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. تبدیل متن به بردار عددی با TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 6. آموزش مدل Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 7. پیش‌بینی و ارزیابی
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
