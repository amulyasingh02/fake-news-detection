import pandas as pd

# -----------------------------
# Load dataset safely
# -----------------------------

df = pd.read_csv(
    "C:/Users/mynam/OneDrive/Desktop/social media/models/mt.csv",
    engine="python",
    on_bad_lines="skip"
)

print("Dataset loaded:", df.shape)

# -----------------------------
# Clean column names
# -----------------------------

df.columns = df.columns.str.strip()

print("Columns detected:")
print(df.columns.tolist())

# -----------------------------
# Identify community columns
# -----------------------------

community_cols = [c for c in df.columns if "Community name_" in c]

print("Community columns detected:", len(community_cols))

# -----------------------------
# Define fake news communities
# -----------------------------

fake_communities = [
    "Community name_r/fakenews",
    "Community name_r/fakenewsnetwork",
    "Community name_r/farialimabets",
]

# -----------------------------
# Create label column
# -----------------------------

df["label"] = 0

for col in fake_communities:
    if col in df.columns:
        df.loc[df[col] == 1, "label"] = 1

# -----------------------------
# Save new dataset
# -----------------------------

df.to_csv("metadata_labeled_dataset.csv", index=False)

print("\nDataset with labels saved as metadata_labeled_dataset.csv")

print("\nLabel distribution:")
print(df["label"].value_counts())