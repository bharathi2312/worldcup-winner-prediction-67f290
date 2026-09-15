"""Predicting a World Cup winner is NOT one prediction — it's a tournament of
dependent matches. We start from historical match results and build up to title
odds. This file just loads the data and frames the problem."""

import pandas as pd


def load_matches(path="data/matches.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    # The raw outcome of each match, from the home team's perspective.
    df["result"] = df.apply(
        lambda r: "W" if r.home_goals > r.away_goals
        else ("L" if r.home_goals < r.away_goals else "D"),
        axis=1,
    )
    return df


if __name__ == "__main__":
    df = load_matches()
    print(f"{len(df)} matches loaded")
    print(df["result"].value_counts())
    print("\nNote: home win/draw/loss — the target we'll learn to predict.")
