# Predict Who Will Win the World Cup

An advanced machine-learning capstone in Python that answers a real question with real method — who will win the World Cup? You start from a table of historical international matches and engineer the features that actually predict outcomes: recent form, goal difference, and an Elo-style strength rating you update match by match. You train a match-outcome model that outputs calibrated win/draw/loss probabilities, then confront the hard part a single prediction can't solve: a tournament is a sequence of dependent matches with a bracket, so you build a Monte-Carlo simulator that plays the entire tournament thousands of times, advancing teams by sampled outcomes, to estimate each team's probability of winning it all. You learn why calibration matters more than accuracy here, how to read the simulated distribution, and how to evaluate a probabilistic forecast honestly against a held-out tournament with log loss and the Brier score. By the end you can take any bracket sport and produce defensible odds — the same pipeline used for real forecasts.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- pandas
- scikit-learn
- NumPy
- Monte Carlo
- Elo Ratings
