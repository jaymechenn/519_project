import os
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extra_features(texts):
    feats = []
    for t in texts:
        raw = str(t)
        words = raw.split()

        feats.append([
            len(words),
            len(raw),
            np.mean([len(w) for w in words]) if words else 0,
            sum(1 for w in words if w.isupper()),
            sum(c.isdigit() for c in raw),
            raw.count("!"),
            raw.count("?"),
            raw.count(","),
            raw.count(":"),
            raw.count(";"),
            int('"' in raw or "'" in raw),
            int(":" in raw),
            int("?" in raw),
            int(any(c.isdigit() for c in raw)),
            int(raw.lower().startswith("how")),
            int(raw.lower().startswith("why")),
            int(raw.lower().startswith("what")),
        ])

    return np.array(feats)

def load_labels(df):
    if "label" not in df.columns:
        return None

    y = []
    for label in df["label"]:
        s = str(label).lower()
        if s in ["0", "0.0"]:
            y.append(0)
        elif s in ["1", "1.0"]:
            y.append(1)
        elif "nbc" in s:
            y.append(1)
        elif "fox" in s:
            y.append(0)
        else:
            raise ValueError(f"Unknown label: {label}")

    return np.array(y)

def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    if "headline_clean" in df.columns:
        texts = df["headline_clean"].astype(str).apply(clean_text)
    elif "headline" in df.columns:
        texts = df["headline"].astype(str).apply(clean_text)
    else:
        raise ValueError("CSV must contain headline or headline_clean")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    vectorizer_path = os.path.join(
        base_dir,
        "..",
        "Resources",
        "model_inputs",
        "baseline_tfidf",
        "tfidf_vectorizer.joblib"
    )

    vectorizer = joblib.load(vectorizer_path)

    X_tfidf = vectorizer.transform(texts)
    X_extra = extra_features(texts)

    X = hstack([X_tfidf, X_extra]).tocsr()
    y = load_labels(df)

    return X, y