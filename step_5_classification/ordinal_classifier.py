"""OrdinalClassifier: Frank & Hall (2001) decomposition into K-1 binary classifiers.

Decomposes a K-class ordinal problem into K-1 binary classifiers P(y > k), then
reconstructs class probabilities:
  P(y=0)   = 1 - P(y>0)
  P(y=k)   = P(y>k-1) - P(y>k)  for 0 < k < K-1
  P(y=K-1) = P(y>K-2)
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Frank & Hall (2001) ordinal decomposition.

    Args:
        estimator: Base binary classifier (must implement predict_proba).
        balanced_weights: If True, recompute balanced sample weights for each
            binary sub-problem P(y > k) independently. This is important because
            each sub-problem has a different class imbalance: P(y > 2) — i.e.
            rosso vs rest — is far more skewed than P(y > 0).
    """

    def __init__(self, estimator: BaseEstimator, balanced_weights: bool = True) -> None:
        self.estimator = estimator
        self.balanced_weights = balanced_weights

    def fit(self, X, y, **fit_params) -> "OrdinalClassifier":
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)
        K = len(self.classes_)
        if K < 2:
            raise ValueError(f"OrdinalClassifier requires >= 2 classes, got {K}.")

        self.classifiers_: List[BaseEstimator] = []
        for k in range(K - 1):
            y_binary = (y_arr > k).astype(int)
            clf = clone(self.estimator)
            kw = dict(fit_params)
            if self.balanced_weights:
                kw["sample_weight"] = compute_sample_weight("balanced", y_binary)
            clf.fit(X, y_binary, **kw)
            self.classifiers_.append(clf)

        return self

    def predict_proba(self, X) -> np.ndarray:
        K = len(self.classes_)
        # shape (n, K-1): P(y > k) for k = 0, ..., K-2
        p_gt = np.column_stack([clf.predict_proba(X)[:, 1] for clf in self.classifiers_])

        proba = np.empty((p_gt.shape[0], K), dtype=float)
        proba[:, 0] = 1.0 - p_gt[:, 0]
        for k in range(1, K - 1):
            proba[:, k] = p_gt[:, k - 1] - p_gt[:, k]
        proba[:, K - 1] = p_gt[:, K - 2]

        # Clip negatives: can occur when binary classifiers are not perfectly isotonic
        np.clip(proba, 0.0, 1.0, out=proba)
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return proba / row_sums

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
