"""MONTE CARLO: play the whole tournament thousands of times. Each trial walks
the bracket, sampling every match; we tally who's champion. Over many trials the
fraction of wins converges to each team's true title probability."""

import numpy as np
from collections import Counter
from sim_match import play_match


def play_bracket(teams, prob_fn, rng):
    """teams: a list whose length is a power of 2 (seeded bracket order).
    prob_fn(home, away) -> {'W','D','L'} probabilities. Returns the champion."""
    round_teams = list(teams)
    while len(round_teams) > 1:
        next_round = []
        # Pair adjacent teams; sample each match; winner advances.
        for i in range(0, len(round_teams), 2):
            home, away = round_teams[i], round_teams[i + 1]
            who = play_match(prob_fn(home, away), rng)
            next_round.append(home if who == "home" else away)
        round_teams = next_round
    return round_teams[0]


def simulate(teams, prob_fn, trials=10000, seed=7):
    rng = np.random.default_rng(seed)
    champions = Counter()
    for _ in range(trials):
        champions[play_bracket(teams, prob_fn, rng)] += 1   # tally each trial
    return champions
