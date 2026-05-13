import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Fake News.csv", sep=";")

# ----------------------------
# Rename columns (if needed)
# ----------------------------

df = df.rename(columns={
    "Post text": "Post text",
    "Post type": "Post type",
    "Flair": "Flair",
    "# upvotes": "# upvotes",
    "# comments": "# comments",
    "# awards": "# awards",
    "Community name": "Community name",
    "Community members": "Community members",
    "Years of membership": "Years of membership",
    "# Post Karma": "# Post Karma",
    "# Comment Karma": "# Comment Karma"
})

# ----------------------------
# Drop unused columns
# ----------------------------

df = df[[
    "Post text",
    "Post type",
    "Flair",
    "# upvotes",
    "# comments",
    "# awards",
    "Community name",
    "Community members",
    "Years of membership",
    "# Post Karma",
    "# Comment Karma",
    "Post tone"
]]

# ----------------------------
# Create label
# ----------------------------

df["label"] = df["Post tone"].map({
    "Fake":1,
    "fake":1,
    "REAL":0,
    "real":0
})

df = df.drop(columns=["Post tone"])

# ----------------------------
# Handle missing values
# ----------------------------

df.fillna(0, inplace=True)

# ----------------------------
# Encode categorical features
# ----------------------------

df["Post type"] = df["Post type"].astype(str)
df["Flair"] = df["Flair"].astype(str)
df["Community name"] = df["Community name"].astype(str)

df = pd.get_dummies(df, columns=[
    "Post type",
    "Flair",
    "Community name"
])

# ----------------------------
# Save processed dataset
# ----------------------------

df.to_csv("metadata_training_dataset.csv", index=False)

print("Dataset preprocessing complete")