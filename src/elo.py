"""ELO gives every team a single strength number, updated after each match by how
much the result beat expectations. It's the strongest single feature because it
compares teams on one scale — and it naturally accounts for opponent quality."""

from collections import defaultdict

BASE_K = 30.0       # how fast ratings move per match
START   = 1500.0    # every team starts here


def expected(rating_a, rating_b):
    # Logistic expectation: chance A beats B given the rating gap.
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update(rating, exp_score, actual, k=BASE_K):
    # Move the rating toward reality: surprise (actual - expected) scaled by K.
    return rating + k * (actual - exp_score)


def add_elo(df):
    elo = defaultdict(lambda: START)
    home_elo, away_elo = [], []

    for _, m in df.iterrows():
        ra, rb = elo[m.home], elo[m.away]
        home_elo.append(ra)         # pre-match ratings = the leakage-free feature
        away_elo.append(rb)

        exp_home = expected(ra, rb)
        actual = 1.0 if m.home_goals > m.away_goals else (0.5 if m.home_goals == m.away_goals else 0.0)
        elo[m.home] = update(ra, exp_home, actual)
        elo[m.away] = update(rb, 1 - exp_home, 1 - actual)

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    return df
