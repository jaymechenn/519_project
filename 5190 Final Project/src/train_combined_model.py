import os
import pandas as pd
import joblib

from scipy.sparse import load_npz, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TFIDF_DIR = os.path.join(BASE_DIR, "Resources", "model_inputs", "baseline_tfidf")
FEATURE_DIR = os.path.join(BASE_DIR, "Resources", "model_inputs", "advanced_features")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load TF-IDF
# -----------------------------

X_train_tfidf = load_npz(os.path.join(TFIDF_DIR, "X_train_tfidf.npz"))
X_val_tfidf = load_npz(os.path.join(TFIDF_DIR, "X_val_tfidf.npz"))
X_test_tfidf = load_npz(os.path.join(TFIDF_DIR, "X_test_tfidf.npz"))

y_train = pd.read_csv(os.path.join(TFIDF_DIR, "y_train.csv")).values.ravel()
y_val = pd.read_csv(os.path.join(TFIDF_DIR, "y_val.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(TFIDF_DIR, "y_test.csv")).values.ravel()


# -----------------------------
# Load features
# -----------------------------

train_feat = pd.read_csv(os.path.join(FEATURE_DIR, "train_features.csv"))
val_feat = pd.read_csv(os.path.join(FEATURE_DIR, "val_features.csv"))
test_feat = pd.read_csv(os.path.join(FEATURE_DIR, "test_features.csv"))

feature_cols = [
    col for col in train_feat.columns
    if col not in ["label", "source", "headline_clean"]
]

X_train_feat = train_feat[feature_cols].values
X_val_feat = val_feat[feature_cols].values
X_test_feat = test_feat[feature_cols].values


# -----------------------------
# Scale features
# -----------------------------

scaler = StandardScaler()
X_train_feat = scaler.fit_transform(X_train_feat)
X_val_feat = scaler.transform(X_val_feat)
X_test_feat = scaler.transform(X_test_feat)


# -----------------------------
# Combine TF-IDF + features
# -----------------------------

X_train = hstack([X_train_tfidf, X_train_feat])
X_val = hstack([X_val_tfidf, X_val_feat])
X_test = hstack([X_test_tfidf, X_test_feat])


# -----------------------------
# Train model
# -----------------------------

model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced"
)

model.fit(X_train, y_train)


# -----------------------------
# Evaluate
# -----------------------------

val_preds = model.predict(X_val)
print("\nVALIDATION:")
print(accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

test_preds = model.predict(X_test)
print("\nTEST:")
print(accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))


# -----------------------------
# Save model
# -----------------------------

joblib.dump(model, os.path.join(MODEL_DIR, "combined_model.joblib"))
print("\nSaved combined model.")