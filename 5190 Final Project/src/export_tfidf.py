import joblib
import json
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_logistic_regression_baseline.joblib"))

features = model.named_steps["features"]
word_vec = features.transformer_list[0][1]
char_vec = features.transformer_list[1][1]
clf = model.named_steps["classifier"]

# --- convert vocab to python ints ---
word_vocab = {k: int(v) for k, v in word_vec.vocabulary_.items()}
char_vocab = {k: int(v) for k, v in char_vec.vocabulary_.items()}

export = {
    "word_vocab": word_vocab,
    "word_idf": [float(x) for x in word_vec.idf_],

    "char_vocab": char_vocab,
    "char_idf": [float(x) for x in char_vec.idf_],

    "coef": [[float(x) for x in row] for row in clf.coef_],
    "intercept": [float(x) for x in clf.intercept_]
}

with open("tfidf_export.json", "w") as f:
    json.dump(export, f)