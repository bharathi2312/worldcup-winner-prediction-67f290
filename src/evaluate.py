"""A forecast is only as good as its track record. We score the MATCH model's
probabilities against real held-out results with two proper scoring rules: LOG
LOSS and the BRIER score. Both punish confident wrong predictions hardest."""

import numpy as np


def log_loss_one(prob_assigned_to_actual):
    # Penalty for one prediction: -log(probability you gave the true outcome).
    # Confident-and-wrong (small p) -> huge penalty; confident-and-right -> ~0.
    return -np.log(max(prob_assigned_to_actual, 1e-15))


def brier_one(probs, actual_index):
    # Squared error between the probability vector and the one-hot truth.
    truth = np.zeros(len(probs))
    truth[actual_index] = 1.0
    return float(np.sum((np.array(probs) - truth) ** 2))


def evaluate(pred_probs, actuals):
    """pred_probs: list of [P(W),P(D),P(L)]; actuals: index of the true class."""
    ll = np.mean([log_loss_one(p[a]) for p, a in zip(pred_probs, actuals)])
    br = np.mean([brier_one(p, a) for p, a in zip(pred_probs, actuals)])
    return {"log_loss": ll, "brier": br}
