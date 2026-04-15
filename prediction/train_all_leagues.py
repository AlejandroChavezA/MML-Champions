#!/usr/bin/env python3
"""
Script genérico para entrenar modelos de predicción para cualquier competición UEFA.
"""

import pickle
import glob
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction.feature_engineering import (
    load_historical_matches,
    extract_features_for_match,
    features_to_numeric
)

FEATURE_COLS = [
    'home_last5_w', 'home_last5_d', 'home_last5_l', 'home_last5_pts',
    'away_last5_w', 'away_last5_d', 'away_last5_l', 'away_last5_pts',
    'home_last5_all_pts', 'away_last5_all_pts',
    'home_goals_for', 'home_goals_against',
    'away_goals_for', 'away_goals_against',
    'h2h_home_w', 'h2h_away_w', 'h2h_draw',
    'home_days_rest', 'away_days_rest',
    'home_ranking_pts', 'away_ranking_pts', 'ranking_diff',
    'home_uefa_coef', 'away_uefa_coef', 'uefa_coef_diff',
    'is_knockout', 'matchday',
    'phase_type', 'home_knockout_exp', 'away_knockout_exp',
    'home_best_round', 'away_best_round',
    'h2h_champions_w', 'h2h_champions_away_w', 'h2h_champions_draw'
]

LABEL_MAP = {0: 'Empate', 1: 'Local', 2: 'Visitante'}


def generate_training_data(league: str) -> pd.DataFrame:
    """Genera datos de entrenamiento para una competición específica."""
    files = sorted(glob.glob(f'data/cleaned/{league}/matches_*.csv'))
    
    if not files:
        print(f"No se encontraron datos para {league}")
        return pd.DataFrame()
    
    print(f"Cargando {len(files)} archivos de {league}...")
    
    history = load_historical_matches(files)
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
            standings=None, n_form=5, matchday=matchday
        )
        
        numeric = features_to_numeric(features)
        
        if home_score > away_score:
            result = 1
        elif home_score < away_score:
            result = 2
        else:
            result = 0
        
        numeric['result'] = result
        numeric['home_team'] = home
        numeric['away_team'] = away
        numeric['match_date'] = match_date_str
        numeric['home_score'] = home_score
        numeric['away_score'] = away_score
        
        rows.append(numeric)
    
    return pd.DataFrame(rows)


def train_model(df: pd.DataFrame, league_name: str) -> Pipeline:
    """Entrena el modelo RandomForest."""
    X = df[FEATURE_COLS].fillna(0)
    y = df['result'].astype(int)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print(f"Entrenando modelo para {league_name}...")
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline: Pipeline, df: pd.DataFrame) -> dict:
    """Evalúa el modelo con cross-validation."""
    X = df[FEATURE_COLS].fillna(0)
    y = df['result'].astype(int)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
    
    return {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std()
    }


def save_model(pipeline: Pipeline, league: str):
    """Guarda el modelo entrenado."""
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / f'{league}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Modelo guardado en {model_path}")


def train_league(league: str, league_name: str):
    """Entrena y guarda el modelo para una competición."""
    print(f"\n{'='*60}")
    print(f"ENTRENANDO MODELO: {league_name}")
    print("="*60)
    
    # Generar datos
    df = generate_training_data(league)
    
    if df.empty:
        print(f"No hay datos para {league_name}")
        return None
    
    print(f"Datos generados: {len(df)} partidos")
    
    # Distribución de clases
    print("\nDistribución de clases:")
    for label, name in LABEL_MAP.items():
        count = (df['result'] == label).sum()
        pct = count / len(df) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Entrenar
    pipeline = train_model(df, league_name)
    
    # Evaluar
    results = evaluate_model(pipeline, df)
    print(f"\nAccuracy: {results['accuracy_mean']:.1%} (+/- {results['accuracy_std']:.1%})")
    print(f"F1 Score: {results['f1_mean']:.3f}")
    
    # Guardar
    save_model(pipeline, league)
    
    return pipeline


def main():
    leagues = [
        ('champions', 'Champions League'),
        ('europa_league', 'Europa League'),
        ('conference', 'Conference League')
    ]
    
    import argparse
    parser = argparse.ArgumentParser(description='Entrenar modelos de predicción')
    parser.add_argument('--league', choices=['champions', 'europa', 'conference', 'all'], 
                       default='all', help='Competición a entrenar')
    args = parser.parse_args()
    
    if args.league == 'all':
        for league, name in leagues:
            train_league(league, name)
    elif args.league == 'europa':
        train_league('europa_league', 'Europa League')
    elif args.league == 'conference':
        train_league('conference', 'Conference League')
    else:
        train_league('champions', 'Champions League')
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
