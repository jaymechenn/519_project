import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
import os
from scipy.sparse import vstack, hstack


class Model:
    def __init__(self, weights_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # -------------------------
        # LOAD TRAIN DATA
        # -------------------------
        tfidf_dir = os.path.join(base_dir, "..", "Resources", "model_inputs", "baseline_tfidf")
        feat_dir = os.path.join(base_dir, "..", "Resources", "model_inputs", "advanced_features")

        X_train_tfidf = load_npz(os.path.join(tfidf_dir, "X_train_tfidf.npz"))
        y_train = pd.read_csv(os.path.join(tfidf_dir, "y_train.csv")).values.ravel()

        X_train_extra = pd.read_csv(os.path.join(feat_dir, "train_features.csv"))
        X_train_extra = X_train_extra.apply(pd.to_numeric, errors="coerce").fillna(0).values

        X_train = hstack([X_train_tfidf, X_train_extra])

        # -------------------------
        # TRAIN MODELS
        # -------------------------
        self.lr = LogisticRegression(max_iter=5000)
        self.lr.fit(X_train, y_train)

        svm_base = LinearSVC()
        self.svm = CalibratedClassifierCV(svm_base)
        self.svm.fit(X_train, y_train)

        self.gb = GradientBoostingClassifier(n_estimators=200)
        self.gb.fit(X_train.toarray(), y_train)


    def predict(self, X):
        from scipy.sparse import vstack, hstack
        import numpy as np

        if isinstance(X, list):
            X = vstack(X)

        X = X.tocsr()

        expected_dim = self.lr.n_features_in_

        if X.shape[1] < expected_dim:
            diff = expected_dim - X.shape[1]
            padding = np.zeros((X.shape[0], diff))
            X = hstack([X, padding]).tocsr()
        elif X.shape[1] > expected_dim:
            X = X[:, :expected_dim].tocsr()

        X_dense = X.toarray()

        p1 = self.lr.predict_proba(X)[:, 1]
        p2 = self.svm.predict_proba(X)[:, 1]
        p3 = self.gb.predict_proba(X_dense)[:, 1]

        final_probs = 0.4 * p1 + 0.3 * p2 + 0.3 * p3

        return (final_probs > 0.5).astype(int)