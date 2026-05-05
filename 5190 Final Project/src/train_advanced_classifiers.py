import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier


# -----------------------------
# Paths
# -----------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURE_DIR = os.path.join(BASE_DIR, "Resources", "model_inputs", "advanced_features")

TRAIN_PATH = os.path.join(FEATURE_DIR, "train_features.csv")
VAL_PATH = os.path.join(FEATURE_DIR, "val_features.csv")
TEST_PATH = os.path.join(FEATURE_DIR, "test_features.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# -----------------------------
# Load data
# -----------------------------

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

feature_cols = [
    col for col in train_df.columns
    if col not in ["label", "source", "headline_clean"]
]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_val = val_df[feature_cols]
y_val = val_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]


models = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0))
    ]),
    
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ),
    
    "gradient_boosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3
    )
}

best_model = None
best_acc = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    
    val_preds = model.predict(X_val)
    acc = accuracy_score(y_val, val_preds)
    
    print(f"\n{name.upper()} VAL:")
    print(acc)
    print(classification_report(y_val, val_preds))
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name


# -----------------------------
# Save best model
# -----------------------------

model_path = os.path.join(MODEL_DIR, "feature_model_best.joblib")
joblib.dump(best_model, model_path)

print(f"\nBest model: {best_name} with acc={best_acc}")
print(f"Saved to: {model_path}")