"""Form counts wins, but not HOW MUCH a team wins by. Average GOAL DIFFERENCE
captures dominance: beating opponents 3-0 signals more strength than scraping
1-0. Same leakage-free rule: only matches before this one."""

import pandas as pd
from collections import defaultdict, deque


def points(goals_for, goals_against):
    if goals_for > goals_against:
        return 3
    return 1 if goals_for == goals_against else 0


def add_features(df, window=5):
    df = df.sort_values("date").reset_index(drop=True)
    pts = defaultdict(lambda: deque(maxlen=window))
    gd  = defaultdict(lambda: deque(maxlen=window))   # recent goal differences
    rows = {"home_form": [], "away_form": [], "home_gd": [], "away_gd": []}

    for _, m in df.iterrows():
        rows["home_form"].append(sum(pts[m.home]))
        rows["away_form"].append(sum(pts[m.away]))
        # Average goal difference so far (0.0 when no history yet).
        rows["home_gd"].append(sum(gd[m.home]) / max(len(gd[m.home]), 1))
        rows["away_gd"].append(sum(gd[m.away]) / max(len(gd[m.away]), 1))

        diff = m.home_goals - m.away_goals
        pts[m.home].append(points(m.home_goals, m.away_goals))
        pts[m.away].append(points(m.away_goals, m.home_goals))
        gd[m.home].append(diff)      # + for home
        gd[m.away].append(-diff)     # - for away (signed per team)

    for k, v in rows.items():
        df[k] = v
    return df
