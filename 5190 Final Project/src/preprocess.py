import pandas as pd
import re


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_data(csv_path):
    df = pd.read_csv(csv_path)

    # -------- TEXT COLUMN --------
    if "headline_clean" in df.columns:
        text_col = "headline_clean"
    elif "headline" in df.columns:
        text_col = "headline"
    elif "title" in df.columns:
        text_col = "title"
    else:
        raise ValueError(f"No valid text column found. Columns: {df.columns}")

    X = df[text_col].apply(clean_text).tolist()

    # -------- LABELS --------
    if "label" in df.columns:
        y = df["label"].tolist()

    elif "url" in df.columns:
        y = []
        for url in df["url"]:
            url = str(url).lower()
            if "foxnews" in url:
                y.append(1)
            elif "nbcnews" in url:
                y.append(0)
            else:
                y.append(0)

    else:
        raise ValueError("No label or URL column found for label extraction.")

    return X, y