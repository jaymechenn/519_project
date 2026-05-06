import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "5190_project_matplotlib"))

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TRAIN_PATH = os.path.join(BASE_DIR, "Resources", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "Resources", "processed", "val.csv")
TEST_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "news_source_ensemble.joblib")
METRICS_PATH = os.path.join(OUTPUT_DIR, "ensemble_metrics.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "ensemble_confusion_matrix.png")

RANDOM_STATE = 42
LABEL_NAMES = ["NBC", "FoxNews"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_split(path):
    df = pd.read_csv(path)
    X = df["headline_clean"].fillna("").astype(str)
    y = df["label"].astype(int)
    return df, X, y


def make_member_models():
    """Create diverse text classifiers that make different TF-IDF views."""
    word_lr = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=2.0,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    char_svm = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    LinearSVC(
                        C=1.0,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                    cv=3,
                ),
            ),
        ]
    )

    word_nb = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            ("classifier", ComplementNB(alpha=0.5)),
        ]
    )

    word_sgd = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="modified_huber",
                    alpha=1e-4,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return [
        ("word_lr", word_lr),
        ("char_svm", char_svm),
        ("word_nb", word_nb),
        ("word_sgd", word_sgd),
    ]


def validation_weights(member_models, X_train, y_train, X_val, y_val):
    """Use validation accuracy to give stronger models a little more vote weight."""
    rows = []
    weights = []

    for name, model in member_models:
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)
        rows.append((name, acc))
        weights.append(max(acc - 0.5, 0.05))

    return rows, weights


def write_metrics(member_rows, val_acc, test_acc, val_report, test_report):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("Ensemble Model: Soft Voting over TF-IDF Classifiers\n")
        f.write("=" * 60 + "\n\n")

        f.write("Member Validation Accuracy:\n")
        for name, acc in member_rows:
            f.write(f"- {name}: {acc:.4f}\n")

        f.write(f"\nEnsemble Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Ensemble Test Accuracy: {test_acc:.4f}\n\n")

        f.write("Validation Classification Report:\n")
        f.write(val_report)
        f.write("\n\nTest Classification Report:\n")
        f.write(test_report)


def save_confusion_matrix(y_test, test_preds):
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    disp.plot()
    plt.title("Ensemble Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def main():
    train_df, X_train, y_train = load_split(TRAIN_PATH)
    val_df, X_val, y_val = load_split(VAL_PATH)
    test_df, X_test, y_test = load_split(TEST_PATH)

    print("\nLoaded data:")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test: {test_df.shape}")

    member_models = make_member_models()

    print("\nTraining member models for validation weighting...")
    member_rows, weights = validation_weights(member_models, X_train, y_train, X_val, y_val)
    for (name, acc), weight in zip(member_rows, weights):
        print(f"{name}: val_acc={acc:.4f}, vote_weight={weight:.4f}")

    val_ensemble = VotingClassifier(
        estimators=make_member_models(),
        voting="soft",
        weights=weights,
        n_jobs=1,
    )

    print("\nTraining validation ensemble on train data only...")
    val_ensemble.fit(X_train, y_train)
    val_preds = val_ensemble.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_report = classification_report(y_val, val_preds, target_names=LABEL_NAMES)

    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    ensemble = VotingClassifier(
        estimators=make_member_models(),
        voting="soft",
        weights=weights,
        n_jobs=1,
    )

    print("\nTraining final soft-voting ensemble on train + validation data...")
    ensemble.fit(X_train_full, y_train_full)

    test_preds = ensemble.predict(X_test)

    test_acc = accuracy_score(y_test, test_preds)

    test_report = classification_report(y_test, test_preds, target_names=LABEL_NAMES)

    print("\nEnsemble Validation Accuracy:")
    print(round(val_acc, 4))
    print("\nEnsemble Test Accuracy:")
    print(round(test_acc, 4))
    print("\nTest Classification Report:")
    print(test_report)

    save_confusion_matrix(y_test, test_preds)
    print(f"\nSaved confusion matrix to:\n{CONFUSION_MATRIX_PATH}")

    joblib.dump(ensemble, MODEL_PATH)
    print(f"\nSaved trained ensemble model to:\n{MODEL_PATH}")

    write_metrics(member_rows, val_acc, test_acc, val_report, test_report)
    print(f"\nSaved metrics summary to:\n{METRICS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
