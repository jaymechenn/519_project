import os
import re
from urllib.parse import urlparse

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# -----------------------------
# File paths
# -----------------------------

# This assumes your project folder looks like:
# 5190 Final Project/
# ├── Resources/
# │   └── csv/
# │       └── url_with_headlines.csv
# ├── outputs/
# └── src/
#     └── data_processing_eda.py

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(BASE_DIR, "Resources", "csv", "url_with_headlines.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "Resources", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------

def get_source_from_url(url):
    """
    Extract source label from the URL domain.
    The model will not use the URL as a feature later.
    We only use it here to create the target label.
    """
    domain = urlparse(str(url)).netloc.lower()

    if "foxnews.com" in domain:
        return "FoxNews"
    elif "nbcnews.com" in domain:
        return "NBC"
    else:
        return "Unknown"


def clean_headline(text):
    """
    Light headline cleaning.

    We do not aggressively remove punctuation or lowercase here because
    headline style may contain useful information for source classification.
    """
    text = str(text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove invalid text values
    if text.lower() in ["nan", "none", ""]:
        return None

    return text


# -----------------------------
# Load data
# -----------------------------

print("\nLoading CSV from:")
print(RAW_DATA_PATH)

df = pd.read_csv(RAW_DATA_PATH)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nFirst few rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())


# -----------------------------
# Basic cleaning
# -----------------------------

# Keep only needed columns
df = df[["url", "headline"]].copy()

# Create label from URL
df["source"] = df["url"].apply(get_source_from_url)

# Clean headline text
df["headline_clean"] = df["headline"].apply(clean_headline)

# Drop rows with invalid headlines
df = df.dropna(subset=["headline_clean"])

# Drop rows where source could not be identified
df = df[df["source"] != "Unknown"]

# Drop exact duplicate headlines
df = df.drop_duplicates(subset=["headline_clean"])

# Numeric label for modeling
# FoxNews = 1, NBC = 0
df["label"] = df["source"].apply(lambda x: 1 if x == "FoxNews" else 0)

# Add simple EDA features
df["headline_length_words"] = df["headline_clean"].apply(lambda x: len(x.split()))
df["headline_length_chars"] = df["headline_clean"].apply(len)


print("\nCleaned dataset shape:")
print(df.shape)

print("\nClass counts:")
print(df["source"].value_counts())

print("\nClass percentages:")
print((df["source"].value_counts(normalize=True) * 100).round(2))

print("\nHeadline word length summary:")
print(df["headline_length_words"].describe())

print("\nHeadline character length summary:")
print(df["headline_length_chars"].describe())


# -----------------------------
# Save cleaned full dataset
# -----------------------------

cleaned_path = os.path.join(PROCESSED_DIR, "news_cleaned.csv")
df.to_csv(cleaned_path, index=False)

print(f"\nSaved cleaned dataset to:")
print(cleaned_path)


# -----------------------------
# Train / validation / test split
# -----------------------------

# 80% train+validation, 20% test
train_val_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# From the remaining 80%, split into 60% train and 20% validation overall
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    random_state=42,
    stratify=train_val_df["label"]
)

print("\nSplit sizes:")
print(f"Train: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")

print("\nTrain class balance:")
print(train_df["source"].value_counts(normalize=True).round(3))

print("\nValidation class balance:")
print(val_df["source"].value_counts(normalize=True).round(3))

print("\nTest class balance:")
print(test_df["source"].value_counts(normalize=True).round(3))


# Save splits
train_path = os.path.join(PROCESSED_DIR, "train.csv")
val_path = os.path.join(PROCESSED_DIR, "val.csv")
test_path = os.path.join(PROCESSED_DIR, "test.csv")

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\nSaved train/validation/test splits to:")
print(train_path)
print(val_path)
print(test_path)


# -----------------------------
# EDA plots
# -----------------------------

# 1. Class balance
plt.figure(figsize=(6, 4))
df["source"].value_counts().plot(kind="bar")
plt.title("Number of Headlines by News Source")
plt.xlabel("Source")
plt.ylabel("Number of Headlines")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"))
plt.close()


# 2. Headline length distribution in words
plt.figure(figsize=(7, 4))
df["headline_length_words"].hist(bins=25)
plt.title("Headline Length Distribution")
plt.xlabel("Headline Length, Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "headline_length_distribution.png"))
plt.close()


# 3. Headline length by source
plt.figure(figsize=(6, 4))
df.boxplot(column="headline_length_words", by="source")
plt.title("Headline Length by Source")
plt.suptitle("")
plt.xlabel("Source")
plt.ylabel("Headline Length, Words")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "headline_length_by_source.png"))
plt.close()


# 4. Average headline length by source
avg_lengths = df.groupby("source")["headline_length_words"].mean().sort_values()

plt.figure(figsize=(6, 4))
avg_lengths.plot(kind="bar")
plt.title("Average Headline Length by Source")
plt.xlabel("Source")
plt.ylabel("Average Number of Words")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "average_headline_length_by_source.png"))
plt.close()


print("\nSaved EDA plots to outputs folder:")
print("- class_balance.png")
print("- headline_length_distribution.png")
print("- headline_length_by_source.png")
print("- average_headline_length_by_source.png")


# -----------------------------
# EDA summary text file
# -----------------------------

summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("News Source Classification: Data Processing and EDA Summary\n")
    f.write("=" * 65 + "\n\n")

    f.write(f"Raw data path: {RAW_DATA_PATH}\n")
    f.write(f"Original dataset shape: {pd.read_csv(RAW_DATA_PATH).shape}\n")
    f.write(f"Cleaned dataset shape: {df.shape}\n\n")

    f.write("Class counts:\n")
    f.write(df["source"].value_counts().to_string())
    f.write("\n\n")

    f.write("Class percentages:\n")
    f.write((df["source"].value_counts(normalize=True) * 100).round(2).to_string())
    f.write("\n\n")

    f.write("Headline word length summary:\n")
    f.write(df["headline_length_words"].describe().to_string())
    f.write("\n\n")

    f.write("Headline character length summary:\n")
    f.write(df["headline_length_chars"].describe().to_string())
    f.write("\n\n")

    f.write("Split sizes:\n")
    f.write(f"Train: {train_df.shape}\n")
    f.write(f"Validation: {val_df.shape}\n")
    f.write(f"Test: {test_df.shape}\n\n")

    f.write("Cleaning procedure:\n")
    f.write("- Source labels were extracted from the article URLs.\n")
    f.write("- URLs were not used as model features because the task is to classify source from headline text only.\n")
    f.write("- Headlines were lightly cleaned by normalizing whitespace.\n")
    f.write("- Rows with missing or invalid headlines were removed.\n")
    f.write("- Rows with unknown source domains were removed.\n")
    f.write("- Exact duplicate headlines were removed.\n")
    f.write("- Numeric labels were created with FoxNews = 1 and NBC = 0.\n")
    f.write("- The dataset was split into train, validation, and test sets using stratified sampling.\n")

print(f"\nSaved EDA summary to:")
print(summary_path)

print("\nDone.")