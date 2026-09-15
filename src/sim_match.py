"""The simulator's atom: play ONE match by SAMPLING from the model's probability
distribution, not by taking the most likely result. A favorite usually advances
— but sometimes the underdog wins, exactly as often as the model says."""

import numpy as np


def play_match(probs, rng):
    """probs: {'W': p, 'D': p, 'L': p} from the home team's view. Returns the
    winner ('home' or 'away'); a knockout draw is resolved by a coin flip."""
    outcomes = ["W", "D", "L"]
    p = [probs["W"], probs["D"], probs["L"]]
    result = rng.choice(outcomes, p=p)          # SAMPLE, weighted by probability

    if result == "W":
        return "home"
    if result == "L":
        return "away"
    # A draw in a knockout must produce a winner — penalties as a 50/50 flip.
    return rng.choice(["home", "away"])


if __name__ == "__main__":
    rng = np.random.default_rng(42)             # seeded: reproducible simulations
    probs = {"W": 0.55, "D": 0.25, "L": 0.20}
    wins = sum(play_match(probs, rng) == "home" for _ in range(10000))
    print(f"home advanced {wins/100:.1f}% of 10000 sampled matches")
