#!/usr/bin/env python3
"""
Menú principal del Champions League Predictor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import glob
import pandas as pd
from datetime import datetime, timezone

from prediction.feature_engineering import (
    load_historical_matches,
    extract_features_for_match,
    features_to_numeric,
    compute_rankings,
    compute_last_n_form,
    compute_goals_stats,
    compute_head_to_head
)
from prediction.output_contract import (
    MatchPredictionRow,
    DetailedMatchPrediction,
    DetailedSignals,
    render_prediccion_jornada_completa,
    render_prediccion_detallada_match
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

LABEL_MAP = {0: 'EMPATE', 1: 'LOCAL', 2: 'VISITANTE'}
LABEL_MAP_REVERSE = {'LOCAL': 1, 'EMPATE': 0, 'VISITANTE': 2}


def load_model():
    """Load trained model."""
    with open('models/champions_model.pkl', 'rb') as f:
        return pickle.load(f)


def load_history():
    """Load all historical matches."""
    files = sorted(glob.glob('data/cleaned/champions/matches_*.csv'))
    return load_historical_matches(files)


def load_current_season():
    """Load current season matches."""
    return pd.read_csv('data/cleaned/champions/matches_2025_26.csv')


def get_matchday_matches(df: pd.DataFrame, matchday: int):
    """Get matches for a specific matchday."""
    return df[df['matchday'] == matchday]


def predict_match(home: str, away: str, date: str, history, model, matchday=None):
    """Predict a single match."""
    features = extract_features_for_match(home, away, date, history, matchday=matchday)
    numeric = features_to_numeric(features)
    X = pd.DataFrame([numeric])[FEATURE_COLS].fillna(0)
    
    proba = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    
    return {
        'home': home,
        'away': away,
        'prediction': LABEL_MAP[pred],
        'confidence': max(proba) * 100,
        'prob_home': proba[1] * 100,
        'prob_draw': proba[0] * 100,
        'prob_away': proba[2] * 100,
        'features': numeric
    }


def get_signals(numeric: dict, home: str, away: str, prediction: str):
    """Generate signals for detailed prediction matching the exact format."""
    pro = []
    con = []
    
    home_short = home[:3].upper()
    away_short = away[:3].upper()
    
    if prediction == 'LOCAL':
        winner_short = away_short
        loser_short = home_short
    elif prediction == 'VISITANTE':
        winner_short = away_short
        loser_short = home_short
    else:
        winner_short = "EMPATE"
        loser_short = "EMPATE"
    
    coef_diff = numeric.get('uefa_coef_diff', 0)
    if coef_diff > 15:
        pro.append(f"Señal del modelo: Diferencia coeficiente UEFA (+{coef_diff/100:.3f}) ⭐")
    elif coef_diff < -15:
        con.append(f"Señal del modelo: Diferencia coeficiente UEFA ({coef_diff/100:.3f}) ⭐")
    
    home_pts = numeric.get('home_last5_pts', 0)
    away_pts = numeric.get('away_last5_pts', 0)
    if home_pts > away_pts:
        pro.append(f"Señal del modelo: Local victoria últimos 5 partidos (+{(home_pts - away_pts)/15:.3f}) ⭐")
    else:
        con.append(f"Señal del modelo: Local victoria últimos 5 partidos ({home_pts/15:.3f}) ⭐")
    
    away_form = numeric.get('away_last5_all_pts', 0)
    if away_form > home_pts:
        pro.append(f"Señal del modelo: Visitante forma últimos 5 partidos (+{(away_form - home_pts)/15:.3f}) ⭐")
    
    home_gf = numeric.get('home_goals_for', 0)
    away_gf = numeric.get('away_goals_for', 0)
    if home_gf > away_gf:
        pro.append(f"Señal del modelo: Goles local promedio (+{home_gf/5:.3f}) ⭐")
    
    if away_gf > home_gf:
        con.append(f"Señal del modelo: Goles visitante promedio (--{away_gf/5:.3f}) ⭐")
    
    h2h_w = numeric.get('h2h_home_w', 0)
    h2h_l = numeric.get('h2h_away_w', 0)
    if h2h_w > h2h_l:
        pro.append(f"Señal del modelo: Diferencia margen últimos 20 partidos (+{h2h_w/5:.3f}) ⭐")
    
    if numeric.get('is_knockout', 0) == 1:
        pro.append(f"Señal del modelo: Eliminación directa (+0.150) ⭐")
    
    while len(pro) < 4:
        pro.append("Señal del modelo: N/A")
    while len(con) < 4:
        con.append("Señal del modelo: N/A")
    
    return DetailedSignals(pro=pro[:4], con=con[:4])


def option_1_prediction_jornada():
    """Opción 1: Predicción jornada completa."""
    print("\n" + "=" * 50)
    print("PREDICCIÓN DE JORNADA COMPLETA")
    print("=" * 50)
    
    df = load_current_season()
    available_matchdays = sorted(df['matchday'].dropna().unique())
    
    print(f"Jornadas disponibles: {int(min(available_matchdays))} - {int(max(available_matchdays))}")
    
    try:
        choice = input("Ingresa el número de jornada: ").strip()
        jornada = int(choice)
    except ValueError:
        print("Número inválido.")
        return
    
    matches = get_matchday_matches(df, jornada)
    matches = matches[(matches['home_team'].notna()) & (matches['away_team'].notna())]
    
    if matches.empty:
        print(f"No hay partidos para la jornada {jornada}.")
        return
    
    model = load_model()
    history = load_history()
    
    rows = []
    for _, m in matches.iterrows():
        home = m['home_team']
        away = m['away_team']
        date = m['date']
        
        pred = predict_match(home, away, date, history, model, jornada)
        
        rows.append(MatchPredictionRow(
            local=home,
            visitante=away,
            prediccion=pred['prediction'],
            confianza_pct=pred['confidence']
        ))
    
    print(render_prediccion_jornada_completa(
        jornadas_min=int(min(available_matchdays)),
        jornadas_max=int(max(available_matchdays)),
        jornada=jornada,
        rows=rows
    ))
    
    input("\nPresiona Enter para continuar...")


def option_2_prediction_detallada():
    """Opción 2: Predicción detallada por partido."""
    print("\n" + "=" * 50)
    print("PREDICCIÓN POR PARTIDO (DETALLADA)")
    print("=" * 50)
    
    df = load_current_season()
    available_matchdays = sorted(df['matchday'].dropna().unique())
    completed = df[df['status'] == 'FINISHED']['matchday'].max()
    
    print(f"Jornadas disponibles: {int(min(available_matchdays))} - {int(max(available_matchdays))}")
    print(f"Jornadas completadas: {int(completed) if pd.notna(completed) else 0}")
    print(f"Próxima jornada: {int(completed) + 1 if pd.notna(completed) else int(min(available_matchdays))}")
    
    print("\nOpciones:")
    print("1. Seleccionar jornada específica")
    print("2. Siguiente jornada no completada")
    print("3. Ver última jornada completada")
    
    choice = input("Selecciona opción (1-3): ").strip()
    
    if choice == "1":
        jornada = input("Número de jornada: ").strip()
        try:
            jornada = int(jornada)
        except ValueError:
            return
    elif choice == "2":
        jornada = int(completed) + 1 if pd.notna(completed) else int(min(available_matchdays))
    elif choice == "3":
        jornada = int(completed) if pd.notna(completed) else int(min(available_matchdays))
    else:
        return
    
    matches = get_matchday_matches(df, jornada)
    matches = matches[(matches['home_team'].notna()) & (matches['away_team'].notna())]
    
    if matches.empty:
        print(f"No hay partidos para la jornada {jornada}.")
        return
    
    model = load_model()
    history = load_history()
    
    print(f"\nAnalizando jornada {jornada}...")
    
    predictions_summary = []
    
    for _, m in matches.iterrows():
        home = m['home_team']
        away = m['away_team']
        date = m['date']
        
        pred = predict_match(home, away, date, history, model, jornada)
        signals = get_signals(pred['features'], home, away, pred['prediction'])
        
        predictions_summary.append(pred)
        
        fecha = datetime.fromisoformat(date.replace("Z", "+00:00")).strftime("%A %-I:%M %p") if date else "Fecha por confirmar"
        
        emoji = "🏠" if pred['prediction'] == 'LOCAL' else "✈️" if pred['prediction'] == 'VISITANTE' else "🤝"
        winner = home if pred['prediction'] == 'LOCAL' else away if pred['prediction'] == 'VISITANTE' else "EMPATE"
        loser = away if pred['prediction'] == 'LOCAL' else home if pred['prediction'] == 'VISITANTE' else "EMPATE"
        
        detallado = DetailedMatchPrediction(
            partido=f"{home[:3].upper()} @ {away[:3].upper()}",
            fecha_linea=f"📅 {fecha}",
            modelo_dice=f"{emoji} GANA {winner}",
            confianza_pct=pred['confidence'],
            breakdown_line=f"{home}: {pred['prob_home']:.1f}% | {away}: {pred['prob_away']:.1f}% | Empate: {pred['prob_draw']:.1f}%",
            pro_title=f"✅ ¿POR QUÉ FAVORECE A {winner[:3].upper()}?",
            con_title=f"❌ ¿POR QUÉ FAVORECE A {loser[:3].upper()}?",
            signals=signals
        )
        
        print(render_prediccion_detallada_match(m=detallado))
    
    local_win = sum(1 for p in predictions_summary if p['prediction'] == 'LOCAL')
    away_win = sum(1 for p in predictions_summary if p['prediction'] == 'VISITANTE')
    draw = sum(1 for p in predictions_summary if p['prediction'] == 'EMPATE')
    avg_conf = sum(p['confidence'] for p in predictions_summary) / len(predictions_summary) if predictions_summary else 0
    
    print(f"\n────────────────────────────────────────────────────────────────────────────────")
    print(f"📊 RESUMEN JORNADA {jornada}")
    print(f"────────────────────────────────────────────────────────────────────────────────")
    print(f"Victorias locales: {local_win}")
    print(f"Victorias visitantes: {away_win}")
    print(f"Empates: {draw}")
    print(f"Total partidos: {len(predictions_summary)}")
    print(f"Confianza promedio: {avg_conf:.1f}%")
    print(f"Modelo utilizado: random_forest")
    print("────────────────────────────────────────────────────────────────────────────────")
    
    input("\nPresiona Enter para continuar...")


def option_3_clasificacion():
    """Opción 3: Ver clasificación/ranking."""
    print("\n" + "=" * 50)
    print("CLASIFICACIÓN CHAMPIONS LEAGUE")
    print("=" * 50)
    
    df = load_current_season()
    df = df[df['status'] == 'FINISHED']
    
    if df.empty:
        print("No hay partidos jugados aún.")
        return
    
    ranking = compute_rankings(df.to_dict('records'))
    
    print("\nPos  Equipo                 Pts  PJ  V  E  D  GF  GA  DG")
    print("-" * 70)
    
    for i, (team, pts, played, wins, draws, losses, gd, gf, ga) in enumerate(ranking[:20], 1):
        print(f"{i:<4} {team:<22} {pts:<4} {played:<4} {wins:<4} {draws:<4} {losses:<4} {gf:<4} {ga:<4} {int(gd):+d}")
    
    input("\nPresiona Enter para continuar...")


def option_4_actualizar():
    """Opción 4: Actualizar datos desde API."""
    print("\n" + "=" * 50)
    print("ACTUALIZAR DATOS")
    print("=" * 50)
    
    from prediction.champions_auto_update import auto_update_champions_current_season
    
    print("Actualizando datos desde football-data.org...")
    auto_update_champions_current_season(force=True)
    
    input("\nPresiona Enter para continuar...")


def main():
    while True:
        print("\n" + "=" * 50)
        print("CHAMPIONS LEAGUE PREDICTOR")
        print("=" * 50)
        print("1. Predicción jornada completa")
        print("2. Predicción detallada por partido")
        print("3. Ver clasificación/ranking")
        print("4. Actualizar datos desde API")
        print("5. Salir")
        
        choice = input("\nSelecciona opción (1-5): ").strip()
        
        if choice == "1":
            option_1_prediction_jornada()
        elif choice == "2":
            option_2_prediction_detallada()
        elif choice == "3":
            option_3_clasificacion()
        elif choice == "4":
            option_4_actualizar()
        elif choice == "5":
            print("\n¡Hasta luego!")
            break
        else:
            print("Opción inválida.")


if __name__ == "__main__":
    main()
