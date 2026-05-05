import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "Resources", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "Resources", "processed", "val.csv")
TEST_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load processed data
# -----------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df["headline_clean"]
y_train = train_df["label"]

X_val = val_df["headline_clean"]
y_val = val_df["label"]

X_test = test_df["headline_clean"]
y_test = test_df["label"]


print("\nLoaded data:")
print(f"Train: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")


# -----------------------------
# Build baseline pipeline
# -----------------------------

# TF-IDF turns each headline into a numerical vector.
# Logistic regression then classifies the vector as FoxNews or NBC.

baseline_model = Pipeline([
    (
        "features",
        FeatureUnion([
            ("word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    stop_words="english",
                    min_df=2,
                    max_df=0.9,
                    sublinear_tf=True
                )
            ),
            ("char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True
                )
            )
        ])
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=3000,
            C=2.0,
            class_weight="balanced"
        )
    )
])


# -----------------------------
# Train model
# -----------------------------

print("\nTraining TF-IDF + Logistic Regression baseline...")
baseline_model.fit(X_train, y_train)


# -----------------------------
# Evaluate on validation set
# -----------------------------

val_preds = baseline_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

print("\nValidation Accuracy:")
print(round(val_acc, 4))

print("\nValidation Classification Report:")
print(classification_report(
    y_val,
    val_preds,
    target_names=["NBC", "FoxNews"]
))


# -----------------------------
# Evaluate on test set
# -----------------------------

test_preds = baseline_model.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)

print("\nTest Accuracy:")
print(round(test_acc, 4))

print("\nTest Classification Report:")
print(classification_report(
    y_test,
    test_preds,
    target_names=["NBC", "FoxNews"]
))


# -----------------------------
# Save confusion matrix
# -----------------------------

cm = confusion_matrix(y_test, test_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["NBC", "FoxNews"]
)

disp.plot()
plt.title("Baseline Model Confusion Matrix")
plt.tight_layout()

confusion_matrix_path = os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.png")
plt.savefig(confusion_matrix_path)
plt.close()

print(f"\nSaved confusion matrix to:")
print(confusion_matrix_path)


# -----------------------------
# Save trained model
# -----------------------------

model_path = os.path.join(MODEL_DIR, "tfidf_logistic_regression_baseline.joblib")
joblib.dump(baseline_model, model_path)

print(f"\nSaved trained baseline model to:")
print(model_path)


# -----------------------------
# Save metrics summary
# -----------------------------

metrics_path = os.path.join(OUTPUT_DIR, "baseline_metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("Baseline Model: TF-IDF + Logistic Regression\n")
    f.write("=" * 55 + "\n\n")

    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")

    f.write("Validation Classification Report:\n")
    f.write(classification_report(
        y_val,
        val_preds,
        target_names=["NBC", "FoxNews"]
    ))

    f.write("\n\nTest Classification Report:\n")
    f.write(classification_report(
        y_test,
        test_preds,
        target_names=["NBC", "FoxNews"]
    ))

print(f"\nSaved metrics summary to:")
print(metrics_path)

print("\nDone.")