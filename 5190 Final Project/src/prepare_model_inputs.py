import os
import re
import joblib
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "Resources", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "Resources", "processed", "val.csv")
TEST_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")

MODEL_INPUTS_DIR = os.path.join(BASE_DIR, "Resources", "model_inputs")

BASELINE_TFIDF_DIR = os.path.join(MODEL_INPUTS_DIR, "baseline_tfidf")
ADVANCED_FEATURES_DIR = os.path.join(MODEL_INPUTS_DIR, "advanced_features")
TRANSFORMER_DIR = os.path.join(MODEL_INPUTS_DIR, "transformer")

os.makedirs(BASELINE_TFIDF_DIR, exist_ok=True)
os.makedirs(ADVANCED_FEATURES_DIR, exist_ok=True)
os.makedirs(TRANSFORMER_DIR, exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------

def count_uppercase_words(text):
    """
    Counts words that are fully uppercase.
    Example: "NASA launches rocket" gives 1.
    """
    words = str(text).split()
    return sum(1 for word in words if word.isupper() and len(word) > 1)


def count_digits(text):
    """
    Counts the number of digit characters in a headline.
    """
    return sum(char.isdigit() for char in str(text))


def count_punctuation(text, punct):
    """
    Counts a specific punctuation mark in a headline.
    """
    return str(text).count(punct)


def contains_quote(text):
    """
    Checks whether the headline contains quotation marks.
    """
    text = str(text)
    return int('"' in text or "'" in text or "“" in text or "”" in text)


def contains_colon(text):
    """
    Checks whether the headline contains a colon.
    News headlines often use colon structures.
    """
    return int(":" in str(text))


def contains_question(text):
    """
    Checks whether the headline is framed as a question.
    """
    return int("?" in str(text))


def contains_number(text):
    """
    Checks whether the headline contains any number.
    """
    return int(any(char.isdigit() for char in str(text)))


def average_word_length(text):
    """
    Average word length in the headline.
    """
    words = str(text).split()

    if len(words) == 0:
        return 0

    return sum(len(word) for word in words) / len(words)


def create_advanced_features(df):
    """
    Creates handcrafted headline-level features.

    These features can be used by models that work better with regular
    tabular inputs, such as random forests, gradient boosting, or MLPs.

    The text itself is not included here because this feature table is meant
    to represent headline style/structure rather than TF-IDF content.
    """
    features = pd.DataFrame()

    text = df["headline_clean"].astype(str)

    features["headline_length_words"] = text.apply(lambda x: len(x.split()))
    features["headline_length_chars"] = text.apply(len)
    features["avg_word_length"] = text.apply(average_word_length)

    features["num_uppercase_words"] = text.apply(count_uppercase_words)
    features["num_digits"] = text.apply(count_digits)

    features["num_exclamation_marks"] = text.apply(lambda x: count_punctuation(x, "!"))
    features["num_question_marks"] = text.apply(lambda x: count_punctuation(x, "?"))
    features["num_commas"] = text.apply(lambda x: count_punctuation(x, ","))
    features["num_colons"] = text.apply(lambda x: count_punctuation(x, ":"))
    features["num_semicolons"] = text.apply(lambda x: count_punctuation(x, ";"))

    features["contains_quote"] = text.apply(contains_quote)
    features["contains_colon"] = text.apply(contains_colon)
    features["contains_question"] = text.apply(contains_question)
    features["contains_number"] = text.apply(contains_number)

    # Simple lexical/style indicators.
    # These are not meant to be perfect. They are useful for exploratory
    # analysis and advanced classical models.
    features["starts_with_how"] = text.str.lower().str.startswith("how").astype(int)
    features["starts_with_why"] = text.str.lower().str.startswith("why").astype(int)
    features["starts_with_what"] = text.str.lower().str.startswith("what").astype(int)

    # Keep the label at the end.
    features["label"] = df["label"].values
    features["source"] = df["source"].values
    features["headline_clean"] = df["headline_clean"].values

    return features


# -----------------------------
# Load processed data
# -----------------------------

print("\nLoading processed train/validation/test splits...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

X_train_text = train_df["headline_clean"].astype(str)
X_val_text = val_df["headline_clean"].astype(str)
X_test_text = test_df["headline_clean"].astype(str)

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]


# -----------------------------
# 1. Baseline TF-IDF inputs
# -----------------------------

print("\nCreating baseline TF-IDF inputs...")

tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=1,
    max_df=0.85,
    sublinear_tf=True,
    max_features=20000
)

# Important:
# Fit only on the training data to prevent validation/test leakage.
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_val_tfidf = tfidf_vectorizer.transform(X_val_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

save_npz(os.path.join(BASELINE_TFIDF_DIR, "X_train_tfidf.npz"), X_train_tfidf)
save_npz(os.path.join(BASELINE_TFIDF_DIR, "X_val_tfidf.npz"), X_val_tfidf)
save_npz(os.path.join(BASELINE_TFIDF_DIR, "X_test_tfidf.npz"), X_test_tfidf)

y_train.to_csv(os.path.join(BASELINE_TFIDF_DIR, "y_train.csv"), index=False)
y_val.to_csv(os.path.join(BASELINE_TFIDF_DIR, "y_val.csv"), index=False)
y_test.to_csv(os.path.join(BASELINE_TFIDF_DIR, "y_test.csv"), index=False)

joblib.dump(
    tfidf_vectorizer,
    os.path.join(BASELINE_TFIDF_DIR, "tfidf_vectorizer.joblib")
)

tfidf_vocab = pd.DataFrame({
    "feature": tfidf_vectorizer.get_feature_names_out()
})
tfidf_vocab.to_csv(
    os.path.join(BASELINE_TFIDF_DIR, "tfidf_vocabulary.csv"),
    index=False
)

print("Saved baseline TF-IDF inputs to:")
print(BASELINE_TFIDF_DIR)
print(f"X_train_tfidf shape: {X_train_tfidf.shape}")
print(f"X_val_tfidf shape: {X_val_tfidf.shape}")
print(f"X_test_tfidf shape: {X_test_tfidf.shape}")


# -----------------------------
# 2. Advanced handcrafted features
# -----------------------------

print("\nCreating advanced handcrafted feature inputs...")

train_features = create_advanced_features(train_df)
val_features = create_advanced_features(val_df)
test_features = create_advanced_features(test_df)

train_features.to_csv(
    os.path.join(ADVANCED_FEATURES_DIR, "train_features.csv"),
    index=False
)
val_features.to_csv(
    os.path.join(ADVANCED_FEATURES_DIR, "val_features.csv"),
    index=False
)
test_features.to_csv(
    os.path.join(ADVANCED_FEATURES_DIR, "test_features.csv"),
    index=False
)

feature_columns = [
    col for col in train_features.columns
    if col not in ["label", "source", "headline_clean"]
]

feature_summary_path = os.path.join(ADVANCED_FEATURES_DIR, "feature_summary.txt")

with open(feature_summary_path, "w", encoding="utf-8") as f:
    f.write("Advanced Feature Summary\n")
    f.write("=" * 40 + "\n\n")

    f.write("These features describe headline structure/style.\n")
    f.write("They are intended for classical models such as random forests,\n")
    f.write("gradient boosting, MLPs, or stacked ensembles.\n\n")

    f.write("Feature columns:\n")
    for col in feature_columns:
        f.write(f"- {col}\n")

    f.write("\nTrain feature shape:\n")
    f.write(str(train_features.shape))
    f.write("\n\nValidation feature shape:\n")
    f.write(str(val_features.shape))
    f.write("\n\nTest feature shape:\n")
    f.write(str(test_features.shape))
    f.write("\n")

print("Saved advanced feature inputs to:")
print(ADVANCED_FEATURES_DIR)
print(f"Train features shape: {train_features.shape}")
print(f"Validation features shape: {val_features.shape}")
print(f"Test features shape: {test_features.shape}")


# -----------------------------
# 3. Transformer-ready inputs
# -----------------------------

print("\nCreating transformer-ready input CSVs...")

# Transformers should use raw cleaned text, not TF-IDF vectors.
# Tokenization should happen later in train_transformer.py using the tokenizer
# for the specific transformer model, such as distilbert-base-uncased.

transformer_train = train_df[["headline_clean", "label", "source"]].copy()
transformer_val = val_df[["headline_clean", "label", "source"]].copy()
transformer_test = test_df[["headline_clean", "label", "source"]].copy()

transformer_train.to_csv(
    os.path.join(TRANSFORMER_DIR, "train_transformer.csv"),
    index=False
)
transformer_val.to_csv(
    os.path.join(TRANSFORMER_DIR, "val_transformer.csv"),
    index=False
)
transformer_test.to_csv(
    os.path.join(TRANSFORMER_DIR, "test_transformer.csv"),
    index=False
)

print("Saved transformer-ready inputs to:")
print(TRANSFORMER_DIR)


# -----------------------------
# 4. Overall model input summary
# -----------------------------

summary_path = os.path.join(MODEL_INPUTS_DIR, "model_input_summary.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Model Input Preparation Summary\n")
    f.write("=" * 50 + "\n\n")

    f.write("Input files used:\n")
    f.write(f"Train: {TRAIN_PATH}\n")
    f.write(f"Validation: {VAL_PATH}\n")
    f.write(f"Test: {TEST_PATH}\n\n")

    f.write("Label mapping:\n")
    f.write("- NBC = 0\n")
    f.write("- FoxNews = 1\n\n")

    f.write("1. Baseline TF-IDF inputs\n")
    f.write("- Directory: Resources/model_inputs/baseline_tfidf/\n")
    f.write("- Uses sklearn TfidfVectorizer\n")
    f.write("- Features: unigrams and bigrams\n")
    f.write("- stop_words='english'\n")
    f.write("- min_df=2\n")
    f.write("- max_df=0.95\n")
    f.write("- Vectorizer fit only on training data\n")
    f.write(f"- X_train_tfidf shape: {X_train_tfidf.shape}\n")
    f.write(f"- X_val_tfidf shape: {X_val_tfidf.shape}\n")
    f.write(f"- X_test_tfidf shape: {X_test_tfidf.shape}\n\n")

    f.write("2. Advanced handcrafted features\n")
    f.write("- Directory: Resources/model_inputs/advanced_features/\n")
    f.write("- Contains numeric headline-style features\n")
    f.write("- Intended for Random Forest, Gradient Boosting, MLP, stacking, etc.\n")
    f.write("- Feature columns:\n")
    for col in feature_columns:
        f.write(f"  - {col}\n")
    f.write("\n")

    f.write("3. Transformer-ready inputs\n")
    f.write("- Directory: Resources/model_inputs/transformer/\n")
    f.write("- Contains cleaned headline text, label, and source\n")
    f.write("- No TF-IDF used for transformers\n")
    f.write("- Tokenization should happen in train_transformer.py\n")
    f.write("- Recommended first transformer: distilbert-base-uncased\n\n")

    f.write("Recommended script order:\n")
    f.write("1. python src/data_processing_eda.py\n")
    f.write("2. python src/prepare_model_inputs.py\n")
    f.write("3. python src/train_baseline_models.py\n")
    f.write("4. python src/train_advanced_classifiers.py\n")
    f.write("5. python src/train_transformer.py\n")
    f.write("6. python src/ensemble_model.py\n")

print("\nSaved overall model input summary to:")
print(summary_path)

print("\nDone preparing all model inputs.")