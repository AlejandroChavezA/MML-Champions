#!/usr/bin/env python3
"""
Prediction pipeline for Champions League.
Step 7: Generate predictions for the next jornada.
"""

import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import glob

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
    'is_knockout', 'matchday'
]


LABEL_MAP = {0: 'Empate', 1: 'Local', 2: 'Visitante'}


def load_model(path: str = 'models/champions_model.pkl'):
    """Load trained model from file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_history():
    """Load all historical matches for feature extraction."""
    files = sorted(glob.glob('data/cleaned/champions/matches_*.csv'))
    return load_historical_matches(files)


def get_upcoming_matches(season: str = '2025-26') -> pd.DataFrame:
    """Get upcoming matches from current season (no result yet)."""
    path = f'data/cleaned/champions/matches_{season.replace("-", "_")}.csv'
    df = pd.read_csv(path)
    
    # Filter matches without result AND with valid teams
    upcoming = df[
        ((df['home_score'].isna()) | (df['status'] != 'FINISHED')) &
        (df['home_team'].notna()) & 
        (df['away_team'].notna()) &
        (df['home_team'] != '') &
        (df['away_team'] != '')
    ].copy()
    
    return upcoming


def predict_match(home_team: str, away_team: str, match_date: str, 
                 history: pd.DataFrame, model, matchday: int = None) -> dict:
    """Predict outcome for a single match."""
    features = extract_features_for_match(
        home_team, away_team, match_date, history,
        standings=None, n_form=5, matchday=matchday
    )
    
    numeric = features_to_numeric(features)
    
    X = pd.DataFrame([numeric])[FEATURE_COLS].fillna(0)
    
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'match_date': match_date,
        'matchday': matchday,
        'prediction': LABEL_MAP[pred],
        'confidence': max(proba) * 100,
        'prob_home': proba[1] * 100,
        'prob_draw': proba[0] * 100,
        'prob_away': proba[2] * 100
    }


def print_predictions(predictions: list):
    """Print predictions in a formatted table."""
    print("\n" + "=" * 80)
    print("PREDICCIONES CHAMPIONS LEAGUE - PRÓXIMA JORNADA")
    print("=" * 80)
    print(f"{'Local':<25} {'Visitante':<25} {'Predicción':<12} {'Confianza':<10}")
    print("-" * 80)
    
    for p in predictions:
        print(f"{p['home_team']:<25} {p['away_team']:<25} {p['prediction']:<12} {p['confidence']:.1f}%")
    
    print("-" * 80)
    print(f"Total partidos: {len(predictions)}")
    print("=" * 80)


def print_detailed_prediction(p: dict):
    """Print detailed prediction for a match."""
    print(f"\n{'=' * 60}")
    print(f"⚽ {p['home_team']} vs {p['away_team']}")
    print(f"{'=' * 60}")
    print(f"📅 Fecha: {p['match_date']}")
    print(f"📊 Jornada: {p['matchday']}")
    print()
    print(f"🎯 PREDICCIÓN: {p['prediction']}")
    print(f"   Confianza: {p['confidence']:.1f}%")
    print()
    print("Probabilidades:")
    print(f"   🏠 Local ({p['home_team']}): {p['prob_home']:.1f}%")
    print(f"   🤝 Empate: {p['prob_draw']:.1f}%")
    print(f"   ✈️  Visitante ({p['away_team']}): {p['prob_away']:.1f}%")
    print()


def main():
    print("=" * 60)
    print("CHAMPIONS LEAGUE PREDICTOR - PREDICCIONES")
    print("=" * 60)
    
    print("\n📥 Cargando modelo...")
    model = load_model()
    
    print("📥 Cargando historial...")
    history = load_history()
    print(f"   {len(history)} partidos en historial")
    
    print("📅 Obteniendo próximos partidos...")
    upcoming = get_upcoming_matches()
    print(f"   {len(upcoming)} partidos sin jugar")
    
    if upcoming.empty:
        print("\n⚠️  No hay partidos programados.")
        return
    
    # Sort by date
    upcoming['date'] = pd.to_datetime(upcoming['date'])
    upcoming = upcoming.sort_values('date')
    
    # Get current date for filtering
    now = datetime.now(timezone.utc)
    
    # Generate predictions
    predictions = []
    print("\n🔮 Generando predicciones...")
    
    for _, match in upcoming.iterrows():
        home = match['home_team']
        away = match['away_team']
        date = match['date']
        matchday = match.get('matchday')
        
        # Convert matchday to int if possible
        try:
            matchday = int(float(matchday)) if pd.notna(matchday) else None
        except:
            matchday = None
        
        # Only predict if match is in the future
        if pd.notna(date) and date > now:
            pred = predict_match(home, away, date.isoformat(), history, model, matchday)
            predictions.append(pred)
    
    if not predictions:
        print("\n⚠️  No hay partidos futuros para predecir.")
        return
    
    # Print summary table
    print_predictions(predictions)
    
    # Ask for detailed view
    print("\n¿Ver detalles de algún partido? (número o Enter para salir)")
    try:
        choice = input("> ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(predictions):
                print_detailed_prediction(predictions[idx])
    except (ValueError, EOFError):
        pass
    
    # Save predictions to CSV
    df_pred = pd.DataFrame(predictions)
    df_pred.to_csv('data/predictions/champions_next_jornada.csv', index=False)
    print(f"\n💾 Predicciones guardadas en data/predictions/champions_next_jornada.csv")


if __name__ == "__main__":
    main()
