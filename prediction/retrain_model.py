#!/usr/bin/env python3
"""
Script para reentrenar el modelo automáticamente.
Se ejecuta cada 50 partidos nuevos o manualmente.
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
TRAINING_PATH = 'data/training/champions_training_full.csv'
MODEL_PATH = 'models/champions_model.pkl'


def load_existing_training_data():
    """Carga datos de entrenamiento existentes."""
    if Path(TRAINING_PATH).exists():
        return pd.read_csv(TRAINING_PATH)
    return pd.DataFrame()


def generate_new_training_data():
    """Genera nuevos datos de entrenamiento desde los CSV."""
    files = sorted(glob.glob('data/cleaned/champions/matches_*.csv'))
    
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


def train_model(df: pd.DataFrame) -> Pipeline:
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
    
    print("Entrenando modelo RandomForest...")
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


def analyze_by_phase(df: pd.DataFrame) -> dict:
    """Analiza accuracy por fase."""
    results = {}
    
    # Clasificar por fase según el mes
    def get_phase(match_date):
        try:
            month = pd.to_datetime(match_date).month
            if month in [9, 10, 11, 12]:
                return 'GRUPO'
            elif month in [1, 2]:
                return '16AVOS'
            elif month == 3:
                return 'OCTAVOS'
            elif month == 4:
                return 'CUARTOS/SEMIS'
            elif month == 5:
                return 'FINAL'
            return 'UNKNOWN'
        except:
            return 'UNKNOWN'
    
    df['phase'] = df['match_date'].apply(get_phase)
    
    for phase in ['GRUPO', '16AVOS', 'OCTAVOS', 'CUARTOS/SEMIS', 'FINAL']:
        phase_df = df[df['phase'] == phase]
        if len(phase_df) > 0:
            X = phase_df[FEATURE_COLS].fillna(0)
            y = phase_df['result']
            
            # Cargar modelo actual
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                
                # Calcular accuracy manually
                preds = model.predict(X)
                correct = (preds == y).sum()
                acc = correct / len(y) * 100
                results[phase] = {'accuracy': acc, 'samples': len(y)}
            except:
                results[phase] = {'accuracy': 0, 'samples': len(y)}
    
    return results


def analyze_by_season(df: pd.DataFrame) -> dict:
    """Analiza accuracy por temporada."""
    results = {}
    
    def get_season(match_date):
        try:
            date = pd.to_datetime(match_date)
            if date.month >= 7:
                return f"{date.year}-{date.year+1}"
            else:
                return f"{date.year-1}-{date.year}"
        except:
            return 'UNKNOWN'
    
    df['season'] = df['match_date'].apply(get_season)
    
    for season in sorted(df['season'].unique()):
        season_df = df[df['season'] == season]
        if len(season_df) > 0:
            X = season_df[FEATURE_COLS].fillna(0)
            y = season_df['result']
            
            try:
                with open(MODEL_PATH, 'rb') as f:
                    model = pickle.load(f)
                
                preds = model.predict(X)
                correct = (preds == y).sum()
                acc = correct / len(y) * 100
                results[season] = {'accuracy': acc, 'samples': len(y)}
            except:
                results[season] = {'accuracy': 0, 'samples': len(y)}
    
    return results


def analyze_trends(df: pd.DataFrame, window: int = 20) -> list:
    """Analiza tendencias de accuracy en ventanas de partidos."""
    df = df.sort_values('match_date').reset_index(drop=True)
    
    trends = []
    
    for i in range(window, len(df)):
        window_df = df.iloc[i-window:i]
        X = window_df[FEATURE_COLS].fillna(0)
        y = window_df['result']
        
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            
            preds = model.predict(X)
            correct = (preds == y).sum()
            acc = correct / len(y) * 100
            
            trends.append({
                'window_start': i - window,
                'window_end': i,
                'accuracy': acc,
                'date': str(window_df.iloc[-1]['match_date'])[:10]
            })
        except:
            continue
    
    return trends


def save_model(pipeline: Pipeline):
    """Guarda el modelo entrenado."""
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"Modelo guardado en {MODEL_PATH}")


def run_retrain():
    """Ejecuta el reentrenamiento."""
    print("=" * 60)
    print("REENTRENAMIENTO DEL MODELO")
    print("=" * 60)
    
    # Cargar datos existentes
    existing_df = load_existing_training_data()
    existing_count = len(existing_df)
    
    # Generar nuevos datos
    new_df = generate_new_training_data()
    new_count = len(new_df)
    
    print(f"\nDatos existentes: {existing_count}")
    print(f"Nuevos datos: {new_count}")
    
    if new_count == 0:
        print("No hay datos nuevos para entrenar.")
        return
    
    # Verificar si hay suficientes datos nuevos
    new_matches = new_count - existing_count
    print(f"Partidos nuevos desde el último entrenamiento: {new_matches}")
    
    # Guardar nuevos datos
    Path(TRAINING_PATH).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(TRAINING_PATH, index=False)
    print(f"Datos de entrenamiento guardados en {TRAINING_PATH}")
    
    # Entrenar modelo
    pipeline = train_model(new_df)
    
    # Evaluar
    results = evaluate_model(pipeline, new_df)
    print(f"\nAccuracy: {results['accuracy_mean']:.1%} (+/- {results['accuracy_std']:.1%})")
    print(f"F1 Score: {results['f1_mean']:.3f} (+/- {results['f1_std']:.3f})")
    
    # Guardar modelo
    save_model(pipeline)
    
    print("\n" + "=" * 60)
    print("REENTRENAMIENTO COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Reentrenar modelo de predicción')
    parser.add_argument('--force', action='store_true', 
                       help='Forzar reentrenamiento sin verificar 50 partidos')
    args = parser.parse_args()
    
    # Cargar datos actuales
    existing_df = load_existing_training_data()
    existing_count = len(existing_df)
    
    # Generar nuevos datos
    new_df = generate_new_training_data()
    new_count = len(new_df)
    
    new_matches = new_count - existing_count
    
    if args.force or new_matches >= 50:
        run_retrain()
    else:
        print(f"Solo hay {new_matches} partidos nuevos (se necesitan 50)")
        print(f"Usa --force para reentrenar de todos modos")
