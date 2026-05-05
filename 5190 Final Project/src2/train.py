import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier

from preprocess import prepare_data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

train_csv = os.path.join(BASE_DIR, "..", "Resources", "processed", "train.csv")
val_csv   = os.path.join(BASE_DIR, "..", "Resources", "processed", "val.csv")

print("Loading training data...")
X_train, y_train = prepare_data(train_csv)

print("Loading validation data...")
X_val, y_val = prepare_data(val_csv)

# -------------------------
# MODELS
# -------------------------
print("Training LR...")
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

print("Training SVM...")
svm = CalibratedClassifierCV(LinearSVC(max_iter=5000))
svm.fit(X_train, y_train)

print("Training GB...")
gb = GradientBoostingClassifier(n_estimators=200)
gb.fit(X_train.toarray(), y_train)

# -------------------------
# VALIDATION (optional)
# -------------------------
p1 = lr.predict_proba(X_val)[:,1]
p2 = svm.predict_proba(X_val)[:,1]
p3 = gb.predict_proba(X_val.toarray())[:,1]

final = (0.4*p1 + 0.3*p2 + 0.3*p3) > 0.5
acc = np.mean(final == y_val)

print("Validation accuracy:", acc)