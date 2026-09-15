"""Now the supervised model you know: features in, win/draw/loss out. The twist
for forecasting is the SPLIT — we train on the past and test on the future, never
randomly, because random splits leak future matches into training."""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from features import add_features
from elo import add_elo

FEATURES = ["home_form", "away_form", "home_gd", "away_gd", "home_elo", "away_elo"]


def build_dataset(df):
    df = add_features(df)
    df = add_elo(df)
    return df


def time_split(df, cutoff):
    # Train on matches before the cutoff date; test on the ones after.
    train = df[df.date < cutoff]
    test  = df[df.date >= cutoff]
    return train, test


def train_model(train):
    X = train[FEATURES]
    y = train["result"]                 # 'W' / 'D' / 'L'
    model = GradientBoostingClassifier()
    model.fit(X, y)
    return model
