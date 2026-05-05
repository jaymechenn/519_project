import os
import joblib
import pandas as pd
import numpy as np
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TFIDF_MODEL_PATH = os.path.join(BASE_DIR, "models", "tfidf_logistic_regression_baseline.joblib")
BERT_MODEL_PATH = os.path.join(BASE_DIR, "models", "distilbert")

TEST_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")


# -----------------------------
# Load test data
# -----------------------------

test_df = pd.read_csv(TEST_PATH)
X_test = test_df["headline_clean"]
y_test = test_df["label"]


# -----------------------------
# TF-IDF predictions
# -----------------------------

tfidf_model = joblib.load(TFIDF_MODEL_PATH)
tfidf_probs = tfidf_model.predict_proba(X_test)[:, 1]


# -----------------------------
# BERT predictions
# -----------------------------

tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)

model.eval()

bert_probs = []

with torch.no_grad():
    for text in X_test:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0][1].item()
        bert_probs.append(probs)

bert_probs = np.array(bert_probs)


# -----------------------------
# Ensemble
# -----------------------------

final_probs = 0.7 * bert_probs + 0.3 * tfidf_probs
final_preds = (final_probs >= 0.5).astype(int)


# -----------------------------
# Evaluate
# -----------------------------

print("\nENSEMBLE RESULTS:\n")
print("Accuracy:", accuracy_score(y_test, final_preds))
print("\nClassification Report:\n")
print(classification_report(y_test, final_preds, target_names=["NBC", "FoxNews"]))