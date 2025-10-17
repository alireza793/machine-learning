# ======================================
# 🧠 Fake News Detection - Logistic Regression
# ======================================

# نصب کتابخانه لازم (فقط اولین بار)
!pip install kagglehub[pandas-datasets]

# -----------------------------
# 1. Import libraries
# -----------------------------
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# 2. Load dataset from Kaggle
# -----------------------------
print("🔹 Loading dataset from Kaggle...")

true_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "clmentbisaillon/fake-and-real-news-dataset",
    "True.csv"
)

fake_df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "clmentbisaillon/fake-and-real-news-dataset",
    "Fake.csv"
)

# -----------------------------
# 3. Add labels (1 = true, 0 = fake)
# -----------------------------
true_df["label"] = 1
fake_df["label"] = 0

# -----------------------------
# 4. Combine datasets
# -----------------------------
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# -----------------------------
# 5. Check for missing values
# -----------------------------
print("\n🔍 Checking missing values...")
print(df.isna().sum())

# -----------------------------
# 6. Prepare data for training
# -----------------------------
X = df["text"]    # فرض بر اینکه متن خبر در ستون 'text' است
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Text vectorization (TF-IDF)
# -----------------------------
print("\n🔹 Converting text to TF-IDF features...")
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -----------------------------
# 8. Train Logistic Regression model
# -----------------------------
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# 9. Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\n📊 Evaluation Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 10. Example prediction
# -----------------------------
example = ["The president announced a new plan for economic recovery."]
example_tfidf = vectorizer.transform(example)
pred = model.predict(example_tfidf)[0]
print("\n📰 Example prediction:", "✅ Real" if pred == 1 else "❌ Fake")