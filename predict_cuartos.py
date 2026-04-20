#!/usr/bin/env python3
"""
Genera predicciones de Champions League para los 4tos de final (ida y vuelta)
en formato compatible con el dashboard safesports-panel.
"""

import pickle
import pandas as pd
import glob
import json
from datetime import datetime
from prediction.feature_engineering import load_historical_matches, extract_features_for_match, features_to_numeric

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

LABEL_MAP = {0: 'EMPATE', 1: 'LOCAL', 2: 'VISITANTE'}
LABEL_MAP_REVERSE = {'EMPATE': 0, 'LOCAL': 1, 'VISITANTE': 2}

TEAM_CODES = {
    'Arsenal FC': 'ARS',
    'Manchester City FC': 'MCI',
    'Real Madrid CF': 'RMA',
    'Paris Saint-Germain FC': 'PSG',
    'FC Barcelona': 'BAR',
    'FC Bayern München': 'BAY',
    'Borussia Dortmund': 'BVB',
    'Club Atlético de Madrid': 'ATM',
    'Sporting Clube de Portugal': 'SPORT',
    'Liverpool FC': 'LIV',
    'FC Internazionale Milano': 'INT',
}

TEAM_FULL_NAMES = {
    'ARS': 'Arsenal FC',
    'MCI': 'Manchester City FC',
    'RMA': 'Real Madrid CF',
    'PSG': 'Paris Saint-Germain FC',
    'BAR': 'FC Barcelona',
    'BAY': 'FC Bayern München',
    'BVB': 'Borussia Dortmund',
    'ATM': 'Club Atlético de Madrid',
    'SPORT': 'Sporting Clube de Portugal',
    'LIV': 'Liverpool FC',
    'INT': 'FC Internazionale Milano',
}

def get_team_code(team_name):
    for full_name, code in TEAM_CODES.items():
        if full_name in team_name or team_name in full_name:
            return code
    return team_name[:3].upper()

def load_model():
    with open('models/champions_model.pkl', 'rb') as f:
        return pickle.load(f)

def load_history():
    files = sorted(glob.glob('data/cleaned/champions/matches_*.csv'))
    history = load_historical_matches(files)
    history['date'] = pd.to_datetime(history['date'], utc=True, errors='coerce').dt.tz_localize(None)
    return history

def get_team_full_name(code):
    return TEAM_FULL_NAMES.get(code, code)

def get_team_logo(code):
    return f"/team-logos/soccer/soccer.png"

def predict_match(home, away, date, matchday, model, history):
    try:
        # Ensure matchday is an integer
        matchday_int = int(matchday) if matchday is not None else 7
        features = extract_features_for_match(home, away, date, history, standings=None, n_form=5, matchday=matchday_int)
        numeric = features_to_numeric(features)
        X = pd.DataFrame([numeric])[FEATURE_COLS].fillna(0)
        
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        
        return {
            'prediction': LABEL_MAP[pred],
            'confidence': max(proba) * 100,
            'home_prob': proba[LABEL_MAP_REVERSE['LOCAL']] * 100,
            'away_prob': proba[LABEL_MAP_REVERSE['VISITANTE']] * 100,
            'draw_prob': proba[LABEL_MAP_REVERSE['EMPATE']] * 100,
            'proba': proba
        }
    except Exception as e:
        print(f"Error predicting {home} vs {away}: {e}")
        return None

def get_quarterfinals_ida():
    """Partidos de ida de 4tos - ajustar según los datos reales"""
    return [
        {'home': 'Arsenal FC', 'away': 'Real Madrid CF', 'date': '2026-04-08', 'matchday': 9, 'leg': 'IDA'},
        {'home': 'FC Bayern München', 'away': 'FC Internazionale Milano', 'date': '2026-04-08', 'matchday': 9, 'leg': 'IDA'},
        {'home': 'FC Barcelona', 'away': 'Borussia Dortmund', 'date': '2026-04-09', 'matchday': 9, 'leg': 'IDA'},
        {'home': 'Paris Saint-Germain FC', 'away': 'Liverpool FC', 'date': '2026-04-09', 'matchday': 9, 'leg': 'IDA'},
    ]

def get_quarterfinals_vuelta():
    """Partidos de vuelta de 4tos"""
    return [
        {'home': 'Real Madrid CF', 'away': 'Arsenal FC', 'date': '2026-04-16', 'matchday': 10, 'leg': 'VUELTA'},
        {'home': 'FC Internazionale Milano', 'away': 'FC Bayern München', 'date': '2026-04-16', 'matchday': 10, 'leg': 'VUELTA'},
        {'home': 'Borussia Dortmund', 'away': 'FC Barcelona', 'date': '2026-04-17', 'matchday': 10, 'leg': 'VUELTA'},
        {'home': 'Liverpool FC', 'away': 'Paris Saint-Germain FC', 'date': '2026-04-17', 'matchday': 10, 'leg': 'VUELTA'},
    ]

def main():
    print("=" * 70)
    print("PREDICCIONES CHAMPIONS LEAGUE - CUARTOS DE FINAL")
    print("=" * 70)
    
    model = load_model()
    history = load_history()
    
    all_predictions = []
    
    # Ida
    print("\n📅 PARTIDOS DE IDA (8-9 Abril)")
    print("-" * 70)
    ida_matches = get_quarterfinals_ida()
    
    for match in ida_matches:
        home = match['home']
        away = match['away']
        date = match['date']
        matchday = match['matchday']
        
        result = predict_match(home, away, date, matchday, model, history)
        
        if result:
            home_code = get_team_code(home)
            away_code = get_team_code(away)
            home_full = get_team_full_name(home_code)
            away_full = get_team_full_name(away_code)
            
            result_map = {'LOCAL': home_code, 'VISITANTE': away_code, 'EMPATE': 'DRAW'}
            predicted_winner = result_map.get(result['prediction'], 'DRAW')
            
            conf = result['confidence']
            risk = 'low' if conf >= 75 else 'medium' if conf >= 55 else 'high'
            
            pred = {
                'sport': 'soccer',
                'homeTeam': home_code,
                'homeTeamFullName': home_full,
                'homeTeamLogo': get_team_logo(home_code),
                'awayTeam': away_code,
                'awayTeamFullName': away_full,
                'awayTeamLogo': get_team_logo(away_code),
                'predictedWinner': predicted_winner,
                'confidence': int(conf),
                'riskLevel': risk,
                'gameDate': date,
                'status': 'active',
                'notes': f"Champions League - Cuartos de Final - {match['leg']}",
                'arguments': {
                    'forWinner': [f"Confianza del modelo: {conf:.0f}%"],
                    'forLoser': [f"Factor de riesgo: {100-conf:.0f}%"],
                    'summary': {
                        'winnerFactors': int(conf / 10),
                        'loserFactors': int((100-conf) / 10),
                        'matchupType': 'champions_quartos',
                        'betRecommendation': f"{predicted_winner} with {conf:.0f}% confidence"
                    }
                }
            }
            all_predictions.append(pred)
            
            print(f"{home_code} vs {away_code}")
            print(f"  → {result['prediction']} ({conf:.1f}%)")
            print(f"  Probabilidades: {home_code}={result['home_prob']:.1f}% | Empate={result['draw_prob']:.1f}% | {away_code}={result['away_prob']:.1f}%")
    
    # Vuelta
    print("\n📅 PARTIDOS DE VUELTA (16-17 Abril)")
    print("-" * 70)
    vuelta_matches = get_quarterfinals_vuelta()
    
    for match in vuelta_matches:
        home = match['home']
        away = match['away']
        date = match['date']
        matchday = match['matchday']
        
        result = predict_match(home, away, date, matchday, model, history)
        
        if result:
            home_code = get_team_code(home)
            away_code = get_team_code(away)
            home_full = get_team_full_name(home_code)
            away_full = get_team_full_name(away_code)
            
            result_map = {'LOCAL': home_code, 'VISITANTE': away_code, 'EMPATE': 'DRAW'}
            predicted_winner = result_map.get(result['prediction'], 'DRAW')
            
            conf = result['confidence']
            risk = 'low' if conf >= 75 else 'medium' if conf >= 55 else 'high'
            
            pred = {
                'sport': 'soccer',
                'homeTeam': home_code,
                'homeTeamFullName': home_full,
                'homeTeamLogo': get_team_logo(home_code),
                'awayTeam': away_code,
                'awayTeamFullName': away_full,
                'awayTeamLogo': get_team_logo(away_code),
                'predictedWinner': predicted_winner,
                'confidence': int(conf),
                'riskLevel': risk,
                'gameDate': date,
                'status': 'active',
                'notes': f"Champions League - Cuartos de Final - {match['leg']}",
                'arguments': {
                    'forWinner': [f"Confianza del modelo: {conf:.0f}%"],
                    'forLoser': [f"Factor de riesgo: {100-conf:.0f}%"],
                    'summary': {
                        'winnerFactors': int(conf / 10),
                        'loserFactors': int((100-conf) / 10),
                        'matchupType': 'champions_quartos',
                        'betRecommendation': f"{predicted_winner} with {conf:.0f}% confidence"
                    }
                }
            }
            all_predictions.append(pred)
            
            print(f"{home_code} vs {away_code}")
            print(f"  → {result['prediction']} ({conf:.1f}%)")
            print(f"  Probabilidades: {home_code}={result['home_prob']:.1f}% | Empate={result['draw_prob']:.1f}% | {away_code}={result['away_prob']:.1f}%")
    
    # Save JSON for dashboard import
    print("\n" + "=" * 70)
    with open('predictions_champions_cuartos.json', 'w') as f:
        json.dump(all_predictions, f, indent=2)
    print(f"✅ Predicciones guardadas en predictions_champions_cuartos.json")
    print(f"   Total: {len(all_predictions)} partidos")

if __name__ == '__main__':
    main()