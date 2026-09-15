"""For a simulator we don't want a single predicted label — we want PROBABILITIES
(P(win), P(draw), P(loss)). And those probabilities must be CALIBRATED: when the
model says 70%, the team should really win about 70% of such matches."""

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier


def train_calibrated(X, y):
    base = GradientBoostingClassifier()
    # Wrap the model so its probabilities are calibrated on held-out folds.
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X, y)
    return model


def match_probabilities(model, features_row):
    # Returns P for each class in model.classes_ order, e.g. [P(D), P(L), P(W)].
    proba = model.predict_proba([features_row])[0]
    return dict(zip(model.classes_, proba))
