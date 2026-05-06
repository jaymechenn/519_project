import os
import re
from urllib.parse import urlparse

import pandas as pd


def clean_headline(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in ["nan", "none", ""]:
        return ""
    return text


def get_source_from_url(url):
    domain = urlparse(str(url)).netloc.lower()
    if "foxnews.com" in domain:
        return "FoxNews"
    if "nbcnews.com" in domain:
        return "NBC"
    return "Unknown"


def label_from_source(source):
    return 1 if source == "FoxNews" else 0


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    if "headline_clean" in df.columns:
        X = df["headline_clean"].fillna("").astype(str).map(clean_headline)
    elif "headline" in df.columns:
        X = df["headline"].fillna("").astype(str).map(clean_headline)
    else:
        raise ValueError("CSV must contain a 'headline_clean' or 'headline' column.")

    if "label" in df.columns:
        y = df["label"].astype(int)
    elif "source" in df.columns:
        y = df["source"].map(label_from_source).astype(int)
    elif "url" in df.columns:
        y = df["url"].map(get_source_from_url).map(label_from_source).astype(int)
    else:
        raise ValueError("CSV must contain 'label', 'source', or 'url' for evaluation labels.")

    return X.tolist(), y.tolist()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, "Resources", "processed", "test.csv")
    X, y = prepare_data(test_path)
    print(f"Loaded {len(X)} examples.")
    print(f"First headline: {X[0]}")
    print(f"First label: {y[0]}")
