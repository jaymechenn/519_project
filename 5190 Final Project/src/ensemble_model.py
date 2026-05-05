import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_logistic_regression_baseline.joblib")
FEATURE_MODEL_PATH = os.path.join(BASE_DIR, "models", "feature_model_best.joblib")

# Data
VAL_TEXT_PATH = os.path.join(BASE_DIR, "Resources", "processed", "val.csv")
VAL_FEATURE_PATH = os.path.join(BASE_DIR, "Resources", "model_inputs", "advanced_features", "val_features.csv")

TEST_TEXT_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")
TEST_FEATURE_PATH = os.path.join(BASE_DIR, "Resources", "model_inputs", "advanced_features", "test_features.csv")


# -----------------------------
# Load models
# -----------------------------

tfidf_model = joblib.load(TFIDF_MODEL_PATH)
feature_model = joblib.load(FEATURE_MODEL_PATH)


# -----------------------------
# Load validation data
# -----------------------------

val_text_df = pd.read_csv(VAL_TEXT_PATH)
val_feat_df = pd.read_csv(VAL_FEATURE_PATH)

X_val_text = val_text_df["headline_clean"]
y_val = val_text_df["label"]

feature_cols = [
    col for col in val_feat_df.columns
    if col not in ["label", "source", "headline_clean"]
]

X_val_feat = val_feat_df[feature_cols]


# -----------------------------
# Get validation probabilities
# -----------------------------

probs_text_val = tfidf_model.predict_proba(X_val_text)[:, 1]
probs_feat_val = feature_model.predict_proba(X_val_feat)[:, 1]


# -----------------------------
# Tune ensemble weight (VALIDATION ONLY)
# -----------------------------

best_acc = 0
best_w = 0

for w in np.linspace(0.7, 0.95, 15):
    probs = w * probs_text_val + (1 - w) * probs_feat_val
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_val, preds)

    if acc > best_acc:
        best_acc = acc
        best_w = w

print(f"[VALIDATION] Best TF-IDF weight: {best_w}, Accuracy: {best_acc:.4f}")


# -----------------------------
# Load test data
# -----------------------------

test_text_df = pd.read_csv(TEST_TEXT_PATH)
test_feat_df = pd.read_csv(TEST_FEATURE_PATH)

X_test_text = test_text_df["headline_clean"]
y_test = test_text_df["label"]

X_test_feat = test_feat_df[feature_cols]


# -----------------------------
# Get test probabilities
# -----------------------------

probs_text_test = tfidf_model.predict_proba(X_test_text)[:, 1]
probs_feat_test = feature_model.predict_proba(X_test_feat)[:, 1]


# -----------------------------
# Final ensemble (apply best weight)
# -----------------------------

final_probs = best_w * probs_text_test + (1 - best_w) * probs_feat_test
final_preds = (final_probs >= 0.5).astype(int)


# -----------------------------
# Evaluate (TEST ONLY ONCE)
# -----------------------------

print("\nENSEMBLE RESULTS (TEST):\n")
print("Accuracy:", round(accuracy_score(y_test, final_preds), 4))

print("\nClassification Report:\n")
print(classification_report(y_test, final_preds, target_names=["NBC", "FoxNews"]))