# Prediction Pipeline (MVP)

- Objective: provide predictions per jornada and per jornada detallada using engineered features from historical matches.
- Data sources: historical matches from data/cleaned/*/matches_*.csv and canonical team names from data/all_teams_canonical.csv.
- Output: predictions_by_round.csv and predictions_by_match.csv with probabilities for Home/Draw/Away and feature usage detail.
- Features (initial MVP):
  - Last 5 games form for home/away (W/D/L)
  - Last 5 goals for/against (avg)
  - Head-to-head last 5 matches (results and goals)
  - Home/away flag
  - Knockout stage flag and rest days since last match
  - League strength proxy (simple league ranking proxy until UEFA coefficient is available)
- Model: minimal baseline classifier (multiclass: HomeWin, Draw, AwayWin) using scikit-learn (logistic regression or gradient boosting).
- Output for tomorrow's jornada: a CSV with predicted probabilities per fixture.

- Next steps:
 1) Implement data loading/normalization helpers.
 2) Implement feature_engineering module to compute features per fixture.
 3) Implement training routine and a predictor for the upcoming jornada.
 4) Generate predictions for the next jornada.
