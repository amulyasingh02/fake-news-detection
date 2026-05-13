import pandas as pd
import joblib
import time
from transformers import pipeline

# -----------------------------
# File paths (UNCHANGED)
# -----------------------------

INPUT_PATH = "C:/Users/mynam/OneDrive/Desktop/social media/data/reddit_stream.csv"
OUTPUT_PATH = "C:/Users/mynam/OneDrive/Desktop/social media/data/scored_reddit_posts.csv"

MODEL_PATH = "C:/Users/mynam/OneDrive/Desktop/social media/models/metadata_model.pkl"

# -----------------------------
# Load models
# -----------------------------

meta_model = joblib.load(MODEL_PATH)

bert_model = pipeline(
    "text-classification",
    model="mrm8488/bert-tiny-finetuned-fake-news-detection"
)

# -----------------------------
# Metadata features
# -----------------------------

meta_features = [
    "# upvotes",
    "# comments",
    "# awards",
    "Community members",
    "Years of membership",
    "# Post Karma",
    "# Comment Karma"
]

processed_ids = set()

print("Realtime multimodal prediction started...")

# -----------------------------
# Continuous monitoring loop
# -----------------------------

while True:

    try:

        df = pd.read_csv(INPUT_PATH)

        new_rows = df[~df["post_id"].isin(processed_ids)]

        if len(new_rows) == 0:
            time.sleep(5)
            continue

        new_rows[meta_features] = new_rows[meta_features].fillna(0)

        # -----------------------------
        # Metadata model prediction
        # -----------------------------

        meta_scores = meta_model.predict_proba(new_rows[meta_features])[:,1]

        # -----------------------------
        # BERT fake news prediction
        # -----------------------------

        text_scores = []

        for text in new_rows["Post text"]:

            try:

                result = bert_model(text[:512])[0]

                label = result["label"].lower()
                score = result["score"]

                if "fake" in label:
                    text_scores.append(score)
                else:
                    text_scores.append(1 - score)

            except:

                text_scores.append(0.5)

        # -----------------------------
        # Combine scores
        # -----------------------------

        new_rows["text_score"] = text_scores
        new_rows["meta_score"] = meta_scores

        new_rows["fake_probability"] = (
            0.7 * new_rows["text_score"] +
            0.3 * new_rows["meta_score"]
        )

        new_rows["prediction"] = (
            new_rows["fake_probability"] > 0.3
        ).astype(int)

        # -----------------------------
        # Save results
        # -----------------------------

        try:

            existing = pd.read_csv(OUTPUT_PATH)

            combined = pd.concat([existing, new_rows])

        except:

            combined = new_rows

        combined.to_csv(OUTPUT_PATH, index=False)

        processed_ids.update(new_rows["post_id"])

        print(len(new_rows), "posts predicted")

    except Exception as e:

        print("Prediction error:", e)

    time.sleep(5)