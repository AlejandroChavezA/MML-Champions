#!/usr/bin/env python3
"""Lightweight feature engineering for per-jornada predictions.

This is an MVP scaffold. It provides helpers to load historical data and
compute basic features such as last-5 form, head-to-head, and home/away
advantage. The heavy lifting (rolling windows across multiple leagues) will
be expanded in follow-up iterations.
"""
from __future__ import annotations

import csv
from typing import List, Dict
import pandas as pd


def load_historical_matches(paths: List[str]) -> pd.DataFrame:
    # Load and concatenate historical matches from a list of CSV files.
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except FileNotFoundError:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_last_n_form(team: str, up_to_date: str, history: pd.DataFrame, n: int = 5) -> List[str]:
    # Returns a list of last n results in chronological order: 'W','D','L'
    # history should have: date, home, away, home_goals, away_goals
    # We'll compute the results for the given team across all rows prior to up_to_date.
    mask_date = history["date"] < up_to_date
    # Build a small helper to check if team is home or away and result
    results = []
    df = history[mask_date].copy()
    # sort by date just in case
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    for _, row in df.iterrows():
        home = str(row.get("home"))
        away = str(row.get("away"))
        if team in (home, away):
            gh = int(row.get("home_goals", 0))
            ga = int(row.get("away_goals", 0))
            if home == team:
                if gh > ga:
                    results.append("W")
                elif gh == ga:
                    results.append("D")
                else:
                    results.append("L")
            else:
                if ga > gh:
                    results.append("W")
                elif ga == gh:
                    results.append("D")
                else:
                    results.append("L")
            if len(results) >= n:
                break
    return results[-n:][::-1]  # return in chronological order (oldest first)


def compute_head_to_head(home: str, away: str, up_to_date: str, history: pd.DataFrame, n: int = 5) -> Dict[str, int]:
    # Return simple count of W/D/L for home and away in last n head-to-head matches
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    mask = (history["date"] < up_to_date) & (
        ((history["home"] == home) & (history["away"] == away)) |
        ((history["home"] == away) & (history["away"] == home))
    )
    hh = history[mask].sort_values("date").tail(n)
    home_w = away_w = draw = 0
    for _, r in hh.iterrows():
        gh = int(r.get("home_goals", 0))
        ga = int(r.get("away_goals", 0))
        if r["home"] == home and r["away"] == away:
            if gh > ga:
                home_w += 1
            elif gh == ga:
                draw += 1
            else:
                away_w += 1
        elif r["home"] == away and r["away"] == home:
            if ga > gh:
                home_w += 1
            elif ga == gh:
                draw += 1
            else:
                away_w += 1
    return {"home_w": home_w, "away_w": away_w, "draw": draw}


def placeholder_feature_engineering():
    # Placeholder for future, extended features
    return None
