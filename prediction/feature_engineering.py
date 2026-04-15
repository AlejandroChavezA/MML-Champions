#!/usr/bin/env python3
"""Lightweight feature engineering for per-jornada predictions.

This is an MVP scaffold. It provides helpers to load historical data and
compute basic features such as last-5 form, head-to-head, and home/away
advantage. The heavy lifting (rolling windows across multiple leagues) will
be expanded in follow-up iterations.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd


def load_historical_matches(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate historical matches from CSV files.
    
    Expected columns: date, home_team, away_team, home_score, away_score
    """
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df = df.rename(columns={
                'home': 'home_team', 
                'away': 'away_team',
                'home_goals': 'home_score',
                'away_goals': 'away_score'
            }, errors='ignore')
        except FileNotFoundError:
            continue
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def compute_last_n_form(team: str, up_to_date: str, history: pd.DataFrame, n: int = 5, is_home: bool = None) -> List[str]:
    """Returns list of last n results in chronological order: ['W','D','L']
    
    Args:
        team: Team name
        up_to_date: Date string to filter matches before this date
        history: DataFrame with matches
        n: Number of matches to consider
        is_home: If True, only home matches. If False, only away. If None, all matches.
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    up_to_dt = pd.to_datetime(up_to_date, errors="coerce")
    if up_to_dt is None:
        return []
    
    if is_home is not None:
        if is_home:
            df = history[(history["date"] < up_to_dt) & (history["home_team"] == team)]
        else:
            df = history[(history["date"] < up_to_dt) & (history["away_team"] == team)]
    else:
        df = history[(history["date"] < up_to_dt) & 
                    ((history["home_team"] == team) | (history["away_team"] == team))]
    
    df = df.sort_values("date", ascending=False).head(n)
    
    results = []
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        if team not in (home, away):
            continue
            
        if is_home is not None:
            if is_home and home != team:
                continue
            if not is_home and away != team:
                continue
                
        gh = float(row.get("home_score", 0) or 0)
        ga = float(row.get("away_score", 0) or 0)
        
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
    return list(reversed(results[-n:])) if results else []


def compute_goals_stats(team: str, up_to_date: str, history: pd.DataFrame, n: int = 5, is_home: bool = None) -> Dict[str, float]:
    """Compute goals for/against average in last n matches.
    
    Returns: {'goals_for': float, 'goals_against': float, 'goals_per_game': float}
    """
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    up_to_dt = pd.to_datetime(up_to_date, errors="coerce")
    if up_to_dt is None:
        return {'goals_for': 0.0, 'goals_against': 0.0, 'goals_per_game': 0.0}
    
    if is_home is not None:
        if is_home:
            team_matches = history[(history["date"] < up_to_dt) & (history["home_team"] == team)]
        else:
            team_matches = history[(history["date"] < up_to_dt) & (history["away_team"] == team)]
    else:
        team_matches = history[(history["date"] < up_to_dt) & 
                               ((history["home_team"] == team) | (history["away_team"] == team))]
    
    df = team_matches.sort_values("date", ascending=False).head(n)
    
    gf, ga = 0.0, 0.0
    matches = 0
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        if team not in (home, away):
            continue
            
        if is_home is not None:
            if is_home and home != team:
                continue
            if not is_home and away != team:
                continue
                
        gh = float(row.get("home_score", 0) or 0)
        ga_val = float(row.get("away_score", 0) or 0)
        
        if home == team:
            gf += gh
            ga += ga_val
        else:
            gf += ga_val
            ga += gh
        matches += 1
    
    if matches == 0:
        return {'goals_for': 0.0, 'goals_against': 0.0, 'goals_per_game': 0.0}
    
    return {
        'goals_for': gf / matches,
        'goals_against': ga / matches,
        'goals_per_game': (gf + ga) / matches
    }


def compute_head_to_head(home: str, away: str, up_to_date: str, history: pd.DataFrame, n: int = 5) -> Dict[str, int]:
    """Return simple count of W/D/L for home and away in last n head-to-head matches."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    mask = (history["date"] < up_to_date) & (
        ((history["home_team"] == home) & (history["away_team"] == away)) |
        ((history["home_team"] == away) & (history["away_team"] == home))
    )
    hh = history[mask].sort_values("date").tail(n)
    home_w = away_w = draw = 0
    for _, r in hh.iterrows():
        gh = int(r.get("home_score", 0) or 0)
        ga = int(r.get("away_score", 0) or 0)
        if r["home_team"] == home and r["away_team"] == away:
            if gh > ga:
                home_w += 1
            elif gh == ga:
                draw += 1
            else:
                away_w += 1
        elif r["home_team"] == away and r["away_team"] == home:
            if ga > gh:
                home_w += 1
            elif ga == gh:
                draw += 1
            else:
                away_w += 1
    return {"home_w": home_w, "away_w": away_w, "draw": draw}


def compute_days_rest(team: str, match_date: str, history: pd.DataFrame) -> int:
    """Days since team's last match before the given date."""
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    
    match_dt = pd.to_datetime(match_date, errors='coerce')
    if match_dt is None:
        return 0
    
    mask = history["date"] < match_dt
    df = history[mask].sort_values("date", ascending=False)
    
    for _, row in df.iterrows():
        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))
        if team in (home, away):
            last_match = row["date"]
            return (match_dt - last_match).days
    return 0


def compute_ranking_points(team: str, standings: Dict[str, Dict]) -> int:
    """Get current league points for a team from standings dict."""
    return standings.get(team, {}).get('pts', 0)


def is_knockout_stage(matchday: int, stage: str = None) -> bool:
    """Check if match is in knockout stage (Round of 16, QF, SF, Final)."""
    if stage:
        stage_upper = stage.upper()
        if any(s in stage_upper for s in ['ROUND_OF_16', 'QUARTER', 'SEMI', 'FINAL', 'PLAY_OFF']):
            return True
    return matchday is not None and matchday > 8


_UEFA_COEFFICIENTS: Dict[str, float] = {}


def load_uefa_coefficients(path: str = None) -> Dict[str, float]:
    """Load UEFA club coefficients from CSV file."""
    global _UEFA_COEFFICIENTS
    if _UEFA_COEFFICIENTS:
        return _UEFA_COEFFICIENTS
    
    if path is None:
        path = Path(__file__).parent.parent / "config" / "uefa_coefficients.csv"
    
    try:
        import csv
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team = row.get('team_name', '').strip()
                coef = row.get('coefficient', '0').strip()
                if team and coef:
                    _UEFA_COEFFICIENTS[team] = float(coef)
    except FileNotFoundError:
        pass
    return _UEFA_COEFFICIENTS


def get_uefa_coefficient(team: str) -> float:
    """Get UEFA coefficient for a team."""
    coeffs = load_uefa_coefficients()
    return coeffs.get(team, 50.0)  # Default to mid-tier if not found


def get_knockout_experience(team: str, history: pd.DataFrame) -> int:
    """Get number of knockout matches played by a team in Champions history."""
    knockout_rounds = ['16AVOS', 'OCTAVOS', 'CUARTOS', 'SEMIS', 'FINAL']
    team_matches = history[
        (history['home_team'] == team) | (history['away_team'] == team)
    ]
    knockout_matches = team_matches[team_matches.get('matchday', 0) > 8]
    return len(knockout_matches)


def get_best_round_reached(team: str, history: pd.DataFrame) -> int:
    """Get the best round reached by a team (0=groups, 1=16avos, 2=octavos, 3=cuartos, 4=semis, 5=final, 6=winner)."""
    knockout_rounds = {
        'GRUPO': 0,
        '16AVOS': 1,
        'OCTAVOS': 2,
        'CUARTOS': 3,
        'SEMIS': 4,
        'FINAL': 5,
    }
    
    team_matches = history[
        (history['home_team'] == team) | (history['away_team'] == team)
    ]
    
    best = 0
    for _, match in team_matches.iterrows():
        md = match.get('matchday', 0)
        if md > 8:  # Knockout phase
            if md <= 16:  # 16avos (jornadas 9-16 in some files)
                best = max(best, 1)
            elif md <= 24:  # Octavos
                best = max(best, 2)
            elif md <= 28:  # Cuartos
                best = max(best, 3)
            elif md <= 30:  # Semis
                best = max(best, 4)
            else:  # Final
                best = max(best, 5)
                # Check if won
                if match.get('home_team') == team:
                    if match.get('home_score', 0) > match.get('away_score', 0):
                        best = 6
                elif match.get('away_team') == team:
                    if match.get('away_score', 0) > match.get('home_score', 0):
                        best = 6
    
    return best


def compute_h2h_champions(home_team: str, away_team: str, history: pd.DataFrame, n: int = 10) -> dict:
    """Compute head-to-head specifically in Champions League."""
    h2h = {'home_w': 0, 'away_w': 0, 'draw': 0}
    
    matches = history[
        ((history['home_team'] == home_team) & (history['away_team'] == away_team)) |
        ((history['home_team'] == away_team) & (history['away_team'] == home_team))
    ].tail(n)
    
    for _, m in matches.iterrows():
        if m['home_team'] == home_team:
            if m['home_score'] > m['away_score']:
                h2h['home_w'] += 1
            elif m['home_score'] < m['away_score']:
                h2h['away_w'] += 1
            else:
                h2h['draw'] += 1
        else:
            if m['home_score'] > m['away_score']:
                h2h['away_w'] += 1
            elif m['home_score'] < m['away_score']:
                h2h['home_w'] += 1
            else:
                h2h['draw'] += 1
    
    return h2h


def extract_features_for_match(
    home_team: str,
    away_team: str,
    match_date: str,
    history: pd.DataFrame,
    standings: Dict[str, Dict] = None,
    n_form: int = 5,
    matchday: int = None,
    stage: str = None
) -> Dict[str, any]:
    """Extract all features for a single match prediction.
    
    Returns dict with all features ready for model input.
    """
    features = {}
    
    features['home_last5'] = compute_last_n_form(home_team, match_date, history, n_form, is_home=True)
    features['away_last5'] = compute_last_n_form(away_team, match_date, history, n_form, is_home=False)
    features['home_last5_all'] = compute_last_n_form(home_team, match_date, history, n_form)
    features['away_last5_all'] = compute_last_n_form(away_team, match_date, history, n_form)
    
    features['home_goals'] = compute_goals_stats(home_team, match_date, history, n_form, is_home=True)
    features['away_goals'] = compute_goals_stats(away_team, match_date, history, n_form, is_home=False)
    
    h2h = compute_head_to_head(home_team, away_team, match_date, history, n_form)
    features['h2h_home_w'] = h2h['home_w']
    features['h2h_away_w'] = h2h['away_w']
    features['h2h_draw'] = h2h['draw']
    
    features['home_days_rest'] = compute_days_rest(home_team, match_date, history)
    features['away_days_rest'] = compute_days_rest(away_team, match_date, history)
    
    features['home_ranking_pts'] = compute_ranking_points(home_team, standings or {})
    features['away_ranking_pts'] = compute_ranking_points(away_team, standings or {})
    
    features['home_uefa_coef'] = get_uefa_coefficient(home_team)
    features['away_uefa_coef'] = get_uefa_coefficient(away_team)
    features['uefa_coef_diff'] = features['home_uefa_coef'] - features['away_uefa_coef']
    
    features['is_knockout'] = 1 if is_knockout_stage(matchday, stage) else 0
    features['matchday'] = matchday or 0
    
    # New knockout features
    features['phase_type'] = features['is_knockout']
    features['home_knockout_exp'] = get_knockout_experience(home_team, history)
    features['away_knockout_exp'] = get_knockout_experience(away_team, history)
    features['home_best_round'] = get_best_round_reached(home_team, history)
    features['away_best_round'] = get_best_round_reached(away_team, history)
    
    # Champions-specific h2h
    h2h_champ = compute_h2h_champions(home_team, away_team, history)
    features['h2h_champions_w'] = h2h_champ['home_w']
    features['h2h_champions_away_w'] = h2h_champ['away_w']
    features['h2h_champions_draw'] = h2h_champ['draw']
    
    features['home_team'] = home_team
    features['away_team'] = away_team
    features['match_date'] = match_date
    
    return features


def features_to_numeric(features: Dict[str, any]) -> Dict[str, float]:
    """Convert feature dict to numeric values for model training."""
    result = {}
    
    def wdl_to_numeric(wdl_list):
        if not wdl_list:
            return {'w': 0, 'd': 0, 'l': 0, 'pts': 0}
        w = sum(1 for x in wdl_list if x == 'W')
        d = sum(1 for x in wdl_list if x == 'D')
        l = sum(1 for x in wdl_list if x == 'L')
        return {'w': w, 'd': d, 'l': l, 'pts': w * 3 + d}
    
    home_last5 = wdl_to_numeric(features.get('home_last5', []))
    away_last5 = wdl_to_numeric(features.get('away_last5', []))
    home_all = wdl_to_numeric(features.get('home_last5_all', []))
    away_all = wdl_to_numeric(features.get('away_last5_all', []))
    
    result['home_last5_w'] = home_last5['w']
    result['home_last5_d'] = home_last5['d']
    result['home_last5_l'] = home_last5['l']
    result['home_last5_pts'] = home_last5['pts']
    
    result['away_last5_w'] = away_last5['w']
    result['away_last5_d'] = away_last5['d']
    result['away_last5_l'] = away_last5['l']
    result['away_last5_pts'] = away_last5['pts']
    
    result['home_last5_all_pts'] = home_all['pts']
    result['away_last5_all_pts'] = away_all['pts']
    
    hg = features.get('home_goals', {})
    ag = features.get('away_goals', {})
    result['home_goals_for'] = hg.get('goals_for', 0)
    result['home_goals_against'] = hg.get('goals_against', 0)
    result['away_goals_for'] = ag.get('goals_for', 0)
    result['away_goals_against'] = ag.get('goals_against', 0)
    
    result['h2h_home_w'] = features.get('h2h_home_w', 0)
    result['h2h_away_w'] = features.get('h2h_away_w', 0)
    result['h2h_draw'] = features.get('h2h_draw', 0)
    
    result['home_days_rest'] = features.get('home_days_rest', 0)
    result['away_days_rest'] = features.get('away_days_rest', 0)
    
    result['home_ranking_pts'] = features.get('home_ranking_pts', 0)
    result['away_ranking_pts'] = features.get('away_ranking_pts', 0)
    result['ranking_diff'] = result['home_ranking_pts'] - result['away_ranking_pts']
    
    result['home_uefa_coef'] = features.get('home_uefa_coef', 50.0)
    result['away_uefa_coef'] = features.get('away_uefa_coef', 50.0)
    result['uefa_coef_diff'] = features.get('uefa_coef_diff', 0)
    
    result['is_knockout'] = features.get('is_knockout', 0)
    result['matchday'] = features.get('matchday', 0)
    
    # New knockout features
    result['phase_type'] = features.get('phase_type', 0)  # 0=groups, 1=knockout
    result['home_knockout_exp'] = features.get('home_knockout_exp', 0)
    result['away_knockout_exp'] = features.get('away_knockout_exp', 0)
    result['home_best_round'] = features.get('home_best_round', 0)
    result['away_best_round'] = features.get('away_best_round', 0)
    result['h2h_champions_w'] = features.get('h2h_champions_w', 0)
    result['h2h_champions_away_w'] = features.get('h2h_champions_away_w', 0)
    result['h2h_champions_draw'] = features.get('h2h_champions_draw', 0)
    
    return result


def build_training_dataset(
    match_files: List[str],
    n_form: int = 5
) -> pd.DataFrame:
    """Build training dataset from historical match files.
    
    Args:
        match_files: List of paths to match CSV files (e.g., ['data/.../matches_2023_24.csv', ...])
        n_form: Number of matches to consider for form features
    
    Returns:
        DataFrame with features and target variable 'result' (0=Draw, 1=HomeWin, 2=AwayWin)
    """
    history = load_historical_matches(match_files)
    if history.empty:
        return pd.DataFrame()
    
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.sort_values("date")
    
    rows = []
    for idx, match in history.iterrows():
        home = str(match.get("home_team", ""))
        away = str(match.get("away_team", ""))
        match_date = match.get("date")
        
        if not home or not away or pd.isna(match_date):
            continue
        
        home_score = match.get("home_score")
        away_score = match.get("away_score")
        
        if home_score is None or away_score is None:
            continue
        
        home_score = float(home_score)
        away_score = float(away_score)
        
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        
        match_date_str = match.isoformat() if hasattr(match, 'isoformat') else str(match_date)
        
        history_before = history[history["date"] < match_date].copy()
        
        if len(history_before) < 3:
            continue
        
        matchday = match.get("matchday")
        try:
            matchday = int(float(matchday)) if matchday is not None else None
        except (ValueError, TypeError):
            matchday = None
        
        features = extract_features_for_match(
            home, away, match_date_str, history_before,
            standings=None, n_form=n_form, matchday=matchday
        )
        
        numeric = features_to_numeric(features)
        
        if home_score > away_score:
            result = 1  # HomeWin
        elif home_score < away_score:
            result = 2  # AwayWin
        else:
            result = 0  # Draw
        
        numeric['result'] = result
        numeric['home_team'] = home
        numeric['away_team'] = away
        numeric['match_date'] = match_date_str
        numeric['home_score'] = home_score
        numeric['away_score'] = away_score
        
        rows.append(numeric)
    
    return pd.DataFrame(rows)


def save_training_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save training dataset to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Training dataset saved to {output_path} ({len(df)} rows)")


def compute_rankings(matches: List[Dict]) -> List[tuple]:
    """Compute league standings from matches.
    
    Returns list of (team, pts, played, wins, draws, losses, gf, ga)
    """
    standings = {}
    for m in matches:
        h = m.get('home_team')
        a = m.get('away_team')
        sh = m.get('home_score')
        sa = m.get('away_score')
        
        if not h or not a or sh is None or sa is None:
            continue
        if pd.isna(sh) or pd.isna(sa):
            continue
            
        sh = float(sh)
        sa = float(sa)
        
        if h not in standings:
            standings[h] = {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'pts': 0}
        if a not in standings:
            standings[a] = {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'gf': 0, 'ga': 0, 'pts': 0}
        
        standings[h]['played'] += 1
        standings[a]['played'] += 1
        standings[h]['gf'] += sh
        standings[h]['ga'] += sa
        standings[a]['gf'] += sa
        standings[a]['ga'] += sh
        
        if sh > sa:
            standings[h]['wins'] += 1
            standings[h]['pts'] += 3
            standings[a]['losses'] += 1
        elif sh < sa:
            standings[a]['wins'] += 1
            standings[a]['pts'] += 3
            standings[h]['losses'] += 1
        else:
            standings[h]['draws'] += 1
            standings[a]['draws'] += 1
            standings[h]['pts'] += 1
            standings[a]['pts'] += 1
    
    ranking = []
    for t, s in standings.items():
        gd = s['gf'] - s['ga']
        ranking.append((t, s['pts'], s['played'], s['wins'], s['draws'], s['losses'], gd, s['gf'], s['ga']))
    
    ranking.sort(key=lambda x: (x[1], x[6]), reverse=True)
    return ranking


def placeholder_feature_engineering():
    return None
