"""Sanity tests: the pipeline must be leakage-free and the odds must be a valid
probability distribution. A failure here means the forecast can't be trusted."""

from src.odds import title_odds


def test_odds_sum_to_one():
    counts = {"Brazil": 1500, "France": 1200, "Argentina": 1100, "Spain": 6200}
    odds = title_odds(counts, trials=10000)
    assert abs(sum(odds.values()) - 1.0) < 1e-9      # valid distribution

def test_favorite_ranked_first():
    counts = {"Brazil": 1500, "France": 1200}
    odds = title_odds(counts, trials=10000)
    assert list(odds)[0] == "Brazil"                 # highest count ranks first
