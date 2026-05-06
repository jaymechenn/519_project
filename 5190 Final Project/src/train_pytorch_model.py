import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "5190_project_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from model import FEATURES_PER_VIEW, NewsClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "Resources", "processed", "train.csv")
VAL_PATH = os.path.join(BASE_DIR, "Resources", "processed", "val.csv")
TEST_PATH = os.path.join(BASE_DIR, "Resources", "processed", "test.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "news_source_pytorch.pt")
METRICS_PATH = os.path.join(OUTPUT_DIR, "pytorch_metrics.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "pytorch_confusion_matrix.png")

LABEL_NAMES = ["NBC", "FoxNews"]
RANDOM_STATE = 42
SVC_C = 1.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_split(path: str):
    df = pd.read_csv(path)
    X = df["headline_clean"].fillna("").astype(str)
    y = df["label"].astype(int)
    return df, X, y


def make_char_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "hash",
                HashingVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    n_features=FEATURES_PER_VIEW,
                    alternate_sign=False,
                    norm=None,
                    lowercase=True,
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ]
    )


def make_word_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "hash",
                HashingVectorizer(
                    n_features=FEATURES_PER_VIEW,
                    alternate_sign=False,
                    norm=None,
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                ),
            ),
            ("tfidf", TfidfTransformer(sublinear_tf=True)),
        ]
    )


def make_feature_pipeline() -> Pipeline:
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion(
        [
            ("word", make_word_pipeline()),
            ("char", make_char_pipeline()),
        ]
    )


def make_classifier() -> LinearSVC:
    return LinearSVC(
        C=SVC_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def build_checkpoint_model(feature_pipeline: Pipeline, classifier: LinearSVC) -> NewsClassifier:
    model = NewsClassifier()
    word_pipeline = feature_pipeline.transformer_list[0][1]
    char_pipeline = feature_pipeline.transformer_list[1][1]
    word_idf = word_pipeline.named_steps["tfidf"].idf_
    char_idf = char_pipeline.named_steps["tfidf"].idf_
    coef = classifier.coef_[0]
    intercept = classifier.intercept_[0]

    with torch.no_grad():
        model.word_idf.copy_(torch.tensor(word_idf, dtype=torch.float32))
        model.char_idf.copy_(torch.tensor(char_idf, dtype=torch.float32))
        model.classifier.weight[0].copy_(torch.tensor(-coef, dtype=torch.float32))
        model.classifier.weight[1].copy_(torch.tensor(coef, dtype=torch.float32))
        model.classifier.bias[0] = -float(intercept)
        model.classifier.bias[1] = float(intercept)

    return model


def save_confusion_matrix(y_test, test_preds):
    cm = confusion_matrix(y_test, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    disp.plot()
    plt.title("PyTorch Checkpoint Model Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()


def write_metrics(val_acc, test_acc, val_report, test_report):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write("PyTorch Checkpoint Model: Character Hash TF-IDF + Linear Classifier\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Validation Accuracy: {val_acc:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Validation Classification Report:\n")
        f.write(val_report)
        f.write("\n\nTest Classification Report:\n")
        f.write(test_report)


def main():
    train_df, X_train, y_train = load_split(TRAIN_PATH)
    val_df, X_val, y_val = load_split(VAL_PATH)
    test_df, X_test, y_test = load_split(TEST_PATH)

    print("\nLoaded data:")
    print(f"Train: {train_df.shape}")
    print(f"Validation: {val_df.shape}")
    print(f"Test: {test_df.shape}")

    feature_pipeline = make_feature_pipeline()
    classifier = make_classifier()

    print("\nTraining validation model...")
    X_train_features = feature_pipeline.fit_transform(X_train)
    classifier.fit(X_train_features, y_train)

    val_model = build_checkpoint_model(feature_pipeline, classifier)
    val_preds = val_model.predict(X_val.tolist())
    val_acc = accuracy_score(y_val, val_preds)
    val_report = classification_report(y_val, val_preds, target_names=LABEL_NAMES)

    print("\nTraining final checkpoint model on train + validation data...")
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    final_feature_pipeline = make_feature_pipeline()
    final_classifier = make_classifier()
    X_train_full_features = final_feature_pipeline.fit_transform(X_train_full)
    final_classifier.fit(X_train_full_features, y_train_full)

    model = build_checkpoint_model(final_feature_pipeline, final_classifier)
    test_preds = model.predict(X_test.tolist())
    test_acc = accuracy_score(y_test, test_preds)
    test_report = classification_report(y_test, test_preds, target_names=LABEL_NAMES)

    print("\nValidation Accuracy:")
    print(round(val_acc, 4))
    print("\nTest Accuracy:")
    print(round(test_acc, 4))
    print("\nTest Classification Report:")
    print(test_report)

    save_confusion_matrix(y_test, test_preds)
    print(f"\nSaved confusion matrix to:\n{CONFUSION_MATRIX_PATH}")

    torch.save({"state_dict": model.state_dict()}, CHECKPOINT_PATH)
    print(f"\nSaved PyTorch checkpoint to:\n{CHECKPOINT_PATH}")

    write_metrics(val_acc, test_acc, val_report, test_report)
    print(f"\nSaved metrics summary to:\n{METRICS_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()
