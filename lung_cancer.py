import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Encode categorical column (GENDER)
data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})

# Encode target variable
data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"YES": 1, "NO": 0})

# ---------------- IMPORTANT FIXES ----------------

# Drop duplicates (very important)
data = data.drop_duplicates()

# Check class balance
print("Class Distribution:")
print(data["LUNG_CANCER"].value_counts())

# Features and target
X = data.drop("LUNG_CANCER", axis=1)
y = data["LUNG_CANCER"]

# Split dataset (stratified to balance classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train improved model
model = RandomForestClassifier(
    n_estimators=300,        # more trees = better learning
    max_depth=8,             # prevent overfitting
    class_weight='balanced', # handle imbalance
    random_state=42
)

model.fit(X_train, y_train)

# Predict on test data
pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, pred)
print("Model Accuracy:", accuracy)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Feature importance
importance = model.feature_importances_
print("\nFeature Importance:")
for col, imp in zip(X.columns, importance):
    print(f"{col}: {imp:.4f}")

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("\n✅ Model retrained successfully!")