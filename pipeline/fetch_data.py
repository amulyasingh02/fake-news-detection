import requests
import pandas as pd
import time
import os
from datetime import datetime

SUBREDDITS = [
    "news",
    "worldnews",
    "politics",
    "technology",
    "science"
]

FETCH_INTERVAL = 120

DATA_PATH = "C:/Users/mynam/OneDrive/Desktop/social media/data/reddit_stream.csv"

HEADERS = {
    "User-Agent": "fake-news-stream-script"
}

columns = [
    "post_id",
    "Post text",
    "# upvotes",
    "# comments",
    "# awards",
    "Community members",
    "Years of membership",
    "# Post Karma",
    "# Comment Karma",
    "timestamp"
]

if not os.path.exists(DATA_PATH):
    pd.DataFrame(columns=columns).to_csv(DATA_PATH, index=False)


def load_existing_ids():
    df = pd.read_csv(DATA_PATH)
    if "post_id" not in df.columns:
        return set()
    return set(df["post_id"].astype(str))


def fetch_posts():

    existing_ids = load_existing_ids()
    records = []

    for subreddit in SUBREDDITS:

        url = f"https://www.reddit.com/r/{subreddit}/new.json?limit=50"

        try:

            response = requests.get(url, headers=HEADERS)

            data = response.json()

            posts = data["data"]["children"]

            for p in posts:

                post = p["data"]

                post_id = post["id"]

                if post_id in existing_ids:
                    continue

                title = post.get("title", "")
                body = post.get("selftext", "")

                text = f"{title} {body}".strip()

                record = {

                    "post_id": post_id,
                    "Post text": text,

                    "# upvotes": post.get("score", 0),
                    "# comments": post.get("num_comments", 0),
                    "# awards": post.get("total_awards_received", 0),

                    "Community members": 0,
                    "Years of membership": 0,
                    "# Post Karma": 0,
                    "# Comment Karma": 0,

                    "timestamp": datetime.utcnow()

                }

                records.append(record)

        except Exception as e:

            print("Fetch error:", e)

    return records


def save_data(records):

    if len(records) == 0:
        print("No new posts")
        return

    df = pd.DataFrame(records)

    df.to_csv(DATA_PATH, mode="a", header=False, index=False)

    print(f"{len(records)} new posts saved")


def run():

    print("Starting Reddit stream...")

    while True:

        try:

            records = fetch_posts()

            save_data(records)

        except Exception as e:

            print("Pipeline error:", e)

        print("Sleeping...\n")

        time.sleep(FETCH_INTERVAL)


if __name__ == "__main__":

    run()