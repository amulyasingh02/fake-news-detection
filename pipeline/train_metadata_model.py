import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# Load labeled dataset
# -----------------------------

df = pd.read_csv("C:/Users/mynam/OneDrive/Desktop/social media/data/metadata_labeled_dataset.csv")

print("Dataset shape:", df.shape)

# -----------------------------
# Remove text columns
# -----------------------------

drop_cols = [
    "Post text",
    "Post URL",
    "Author nick",
    "Author profile URL"
]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# -----------------------------
# Remove community one-hot columns
# (they cause overfitting)
# -----------------------------

community_cols = [c for c in df.columns if "Community name_" in c]

df = df.drop(columns=community_cols)

# -----------------------------
# Convert numeric fields
# -----------------------------

numeric_cols = [
    "# upvotes",
    "# comments",
    "# awards",
    "Community members",
    "Years of membership",
    "# Post Karma",
    "# Comment Karma"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.fillna(0)

# -----------------------------
# Feature matrix
# -----------------------------

X = df[numeric_cols]
y = df["label"]

print("Training features:", X.columns.tolist())

# -----------------------------
# Train-test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------

pred = model.predict(X_test)

print("\nModel Performance:\n")
print(classification_report(y_test, pred))

# -----------------------------
# Save model
# -----------------------------

joblib.dump(model, "metadata_model.pkl")

print("\nModel saved as metadata_model.pkl")