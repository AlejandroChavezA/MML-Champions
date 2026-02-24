#!/usr/bin/env python3
"""Prediction interface for upcoming jornadas.

This module is a placeholder MVP to illustrate how predictions would be produced
from a trained model. It expects a trained model and a dataframe of fixtures with
computed features, and returns probabilities for Home/Draw/Away.
"""
from __future__ import annotations

import pandas as pd
from typing import Tuple


def predict_with_model(model, features_df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder: in a full MVP, this would call model.predict_proba and map
    # to a structured dataframe with columns: home_win, draw, away_win
    if model is None:
        raise ValueError("Model is not provided")
    if features_df is None or features_df.empty:
        return pd.DataFrame()
    probs = model.predict_proba(features_df)
    # Align with 3 classes: [HomeWin, Draw, AwayWin]
    result = pd.DataFrame(probs, columns=["prob_home_win", "prob_draw", "prob_away_win"])
    return result
