"""The simulator returns a count of how often each team was champion. Title odds
are just those counts normalized to probabilities — and sorted, they ARE the
answer: each team's chance of winning the World Cup."""


def title_odds(champion_counts, trials):
    # Empirical probability = wins / trials, for every team that ever won.
    odds = {team: wins / trials for team, wins in champion_counts.items()}
    # Sort favorites first.
    return dict(sorted(odds.items(), key=lambda kv: kv[1], reverse=True))


def show(odds, top=8):
    print("Title odds (top teams):")
    for team, p in list(odds.items())[:top]:
        bar = "#" * int(p * 50)
        print(f"  {team:<14} {p*100:5.1f}%  {bar}")
