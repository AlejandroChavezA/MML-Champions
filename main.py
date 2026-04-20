#!/usr/bin/env python3
"""
Champions League Predictor - Interactive Menu System
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime
import csv
import pandas as pd
import glob

from prediction.champions_auto_update import auto_update_champions_current_season
from prediction.feature_engineering import (
    load_historical_matches,
    extract_features_for_match,
    features_to_numeric
)

BASE_YEAR = "2025-26"
CLEAN_ROOT = Path("data/cleaned")
CHAMPIONS_DIR = CLEAN_ROOT / "champions"
TEAMS_FILE = CHAMPIONS_DIR / f"teams_{BASE_YEAR.replace('-', '_')}.csv"
MATCHES_FILE = CHAMPIONS_DIR / f"matches_{BASE_YEAR.replace('-', '_')}.csv"

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
    # New knockout features
    'phase_type', 'home_knockout_exp', 'away_knockout_exp',
    'home_best_round', 'away_best_round',
    'h2h_champions_w', 'h2h_champions_away_w', 'h2h_champions_draw'
]

LABEL_MAP = {0: 'EMPATE', 1: 'LOCAL', 2: 'VISITANTE'}
LABEL_MAP_REVERSE = {'EMPATE': 0, 'LOCAL': 1, 'VISITANTE': 2}

# Configuración de competiciones
LEAGUES = {
    'champions': {
        'name': 'Champions League',
        'dir': 'champions',
        'model': 'models/champions_model.pkl'
    },
    'europa': {
        'name': 'Europa League',
        'dir': 'europa_league',
        'model': 'models/europa_league_model.pkl'
    },
    'conference': {
        'name': 'Conference League',
        'dir': 'conference',
        'model': 'models/conference_model.pkl'
    }
}

current_league = 'champions'
current_model = "random_forest"
_model = None
_history = None


def load_model():
    """Load trained model from file."""
    global _model
    if _model is None:
        league_config = LEAGUES[current_league]
        with open(league_config['model'], 'rb')as f:
            _model = pickle.load(f)
    return _model


def load_history():
    """Load all historical matches for feature extraction."""
    global _history
    if _history is None:
        league_config = LEAGUES[current_league]
        files = sorted(glob.glob(f'data/cleaned/{league_config["dir"]}/matches_*.csv'))
        _history = load_historical_matches(files)
        _history['date'] = pd.to_datetime(_history['date'], utc=True, errors='coerce').dt.tz_localize(None)
    return _history


def load_cleaned_matches(year: str = BASE_YEAR):
    league_config = LEAGUES[current_league]
    path = CLEAN_ROOT / league_config['dir'] / f"matches_{year.replace('-', '_')}.csv"
    if not path.exists():
        return []
    matches = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            matches.append({
                'id': row.get('id'),
                'date': row.get('date'),
                'matchday': int(row.get('matchday') or 0),
                'home_team': row.get('home_team'),
                'away_team': row.get('away_team'),
                'home_score': float(row.get('home_score') or 0),
                'away_score': float(row.get('away_score') or 0),
                'status': row.get('status'),
            })
    return matches


def load_teams(year: str = BASE_YEAR):
    league_config = LEAGUES[current_league]
    path = CLEAN_ROOT / league_config['dir'] / f"teams_{year.replace('-', '_')}.csv"
    teams = []
    if not path.exists():
        return teams
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and len(row) > 1:
                teams.append(row[1])
    return sorted(list(dict.fromkeys(teams)))


def compute_rankings(matches):
    standings = {}
    for m in matches:
        h = m.get('home_team')
        a = m.get('away_team')
        sh = m.get('home_score')
        sa = m.get('away_score')
        if not h or not a or sh is None or sa is None:
            continue
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


def print_rankings(ranking):
    print("\n" + "=" * 80)
    print("TABLA DE POSICIONES - CHAMPIONS LEAGUE")
    print("=" * 80)
    print(f"{'Pos':<4} {'Equipo':<25} {'Pts':>4} {'PJ':>4} {'V':>4} {'E':>4} {'D':>4} {'GD':>5} {'GF':>4} {'GA':>4}")
    print("-" * 80)
    for i, row in enumerate(ranking[:20], 1):
        t, pts, played, wins, draws, losses, gd, gf, ga = row
        print(f"{i:<4} {t:<25} {pts:>4} {played:>4} {wins:>4} {draws:>4} {losses:>4} {int(gd):+5} {int(gf):>4} {int(ga):>4}")
    print("=" * 80)


def print_team_stats(teams, matches):
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE EQUIPOS")
    print("=" * 60)
    print(f"Total de equipos disponibles: {len(teams)}")
    print(f"Total de partidos registrados: {len(matches)}")
    print("\nEquipos:")
    for i, team in enumerate(teams[:20], 1):
        print(f"  {i:2}. {team}")
    if len(teams) > 20:
        print(f"  ... y {len(teams) - 20} más")


def get_team_by_number(teams, num):
    if 1 <= num <= len(teams):
        return teams[num - 1]
    return None


def predict_match_ml(home_team, away_team, match_date=None, matchday=None):
    """Predict using trained ML model."""
    try:
        history = load_history()
        model = load_model()
        
        if match_date is None:
            match_date = datetime.now().isoformat()
        else:
            # Convert to tz-naive for comparison with history
            match_date = pd.to_datetime(match_date, utc=True).tz_localize(None).isoformat()
        
        features = extract_features_for_match(
            home_team, away_team, match_date, history,
            standings=None, n_form=5, matchday=matchday
        )
        
        numeric = features_to_numeric(features)
        X = pd.DataFrame([numeric])[FEATURE_COLS].fillna(0)
        
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        
        pred_label = LABEL_MAP[pred]
        confidence = max(proba) * 100
        
        home_prob = proba[LABEL_MAP_REVERSE['LOCAL']] * 100
        away_prob = proba[LABEL_MAP_REVERSE['VISITANTE']] * 100
        draw_prob = proba[LABEL_MAP_REVERSE['EMPATE']] * 100
        
        return {
            'home': home_team,
            'away': away_team,
            'prediction': pred_label,
            'confidence': confidence,
            'home_prob': home_prob,
            'away_prob': away_prob,
            'draw_prob': draw_prob,
            'features': features,
            'numeric_features': numeric
        }
    except Exception as e:
        # Fallback to history-based prediction
        return None


def predict_match(home_team, away_team, matches):
    """Fallback predict using history if ML model fails."""
    home_wins = 0
    away_wins = 0
    draws = 0

    for m in matches:
        if m['home_team'] == home_team and m['away_team'] == away_team:
            if m['home_score'] > m['away_score']:
                home_wins += 1
            elif m['home_score'] < m['away_score']:
                away_wins += 1
            else:
                draws += 1
        elif m['home_team'] == away_team and m['away_team'] == home_team:
            if m['home_score'] > m['away_score']:
                away_wins += 1
            elif m['home_score'] < m['away_score']:
                home_wins += 1
            else:
                draws += 1

    total = home_wins + away_wins + draws
    if total == 0:
        return None

    home_prob = home_wins / total
    away_prob = away_wins / total
    draw_prob = draws / total

    if home_prob > away_prob and home_prob > draw_prob:
        prediction = "LOCAL"
        confidence = home_prob * 100
    elif away_prob > home_prob and away_prob > draw_prob:
        prediction = "VISITANTE"
        confidence = away_prob * 100
    else:
        prediction = "EMPATE"
        confidence = draw_prob * 100

    return {
        'home': home_team,
        'away': away_team,
        'prediction': prediction,
        'confidence': confidence,
        'home_prob': home_prob * 100,
        'away_prob': away_prob * 100,
        'draw_prob': draw_prob * 100,
        'head_to_head': {
            'home_wins': home_wins,
            'away_wins': away_wins,
            'draws': draws
        }
    }


def option_1_predict_round(matches, teams):
    print("\n" + "=" * 60)
    print("PREDICCIÓN DE JORNADA COMPLETA")
    print("=" * 60)

    matchdays = sorted(set(m.get('matchday', 0) for m in matches if m.get('matchday')))
    if not matchdays:
        print("No hay jornadas disponibles.")
        return

    print(f"Jornadas disponibles: {min(matchdays)} - {max(matchdays)}")
    print(f"Jornadas completadas: {len([m for m in matches if m.get('status') == 'finished'])}")

    try:
        round_num = int(input("\nIngresa el número de jornada: "))
    except ValueError:
        print("Número inválido.")
        return

    round_matches = [m for m in matches if m.get('matchday') == round_num]
    if not round_matches:
        print(f"No hay partidos en la jornada {round_num}.")
        return

    # Check if jornada is finished
    finished_matches = [m for m in round_matches if m.get('status') == 'FINISHED']
    is_finished = len(finished_matches) == len(round_matches) and len(round_matches) > 0

    print(f"\n{'=' * 80}")
    if is_finished:
        print(f"RESULTADOS JORNADA {round_num}")
    else:
        print(f"PREDICCIONES JORNADA {round_num}")
    print("=" * 80)
    
    if is_finished:
        correct = 0
        total = 0
        
        for m in round_matches:
            home = m.get('home_team', 'N/A')
            away = m.get('away_team', 'N/A')
            home_score = m.get('home_score', 0)
            away_score = m.get('away_score', 0)
            
            # Determine actual result
            if home_score > away_score:
                actual = "LOCAL"
            elif home_score < away_score:
                actual = "VISITANTE"
            else:
                actual = "EMPATE"
            
            result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
            if not result:
                result = predict_match(home, away, matches)
            
            if result:
                pred = result['prediction']
                conf = result['confidence']
                
                # Check if prediction was correct
                is_correct = pred == actual
                if is_correct:
                    correct += 1
                    status = "OK"
                else:
                    status = "ERROR"
                total += 1
                
                # Format scores
                score_str = f"{int(home_score)}-{int(away_score)}"
                print(f"{home:<25} {away:<25} {pred:<12} {conf:.1f}%      {actual:<10} {score_str:<6} {status}")
            else:
                score_str = f"{int(home_score)}-{int(away_score)}"
                print(f"{home:<25} {away:<25} {'N/A':<12} {'--':<10}      {actual:<10} {score_str:<6}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print("=" * 80)
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    else:
        # Regular prediction mode
        total_confidence = 0
        for m in round_matches:
            home = m.get('home_team', 'N/A')
            away = m.get('away_team', 'N/A')
            result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
            if not result:
                result = predict_match(home, away, matches)
            if result:
                pred = result['prediction']
                conf = result['confidence']
                total_confidence += conf
                print(f"{home:<25} {away:<25} {pred:<12} {conf:.1f}%")
            else:
                print(f"{home:<25} {away:<25} {'N/A':<12} {'--':<10}")

        print("=" * 80)
        print(f"Confianza promedio: {total_confidence / len(round_matches):.1f}%")
    
    print(f"Modelo utilizado: {current_model}")
    input("\nPresiona Enter para continuar...")


def option_2_predict_round_detailed(matches, teams):
    print("\n" + "=" * 60)
    print("PREDICCIÓN POR JORNADA (DETALLADA)")
    print("=" * 60)

    print("\nSelecciona tipo de predicción:")
    print("1. Por jornada (fase de grupos)")
    print("2. Por fase eliminatoria")
    
    try:
        type_choice = input("\nSelecciona opción (1-2): ")
    except ValueError:
        return
    
    if type_choice == "2":
        # Phase-based prediction
        option_8_predict_by_phase_detailed(matches, teams)
        return
    
    # Continue with jornada-based prediction (original)
    matchdays = sorted(set(m.get('matchday', 0) for m in matches if m.get('matchday')))
    if not matchdays:
        print("No hay jornadas disponibles.")
        return

    completed = len([m for m in matches if m.get('status') == 'finished'])
    next_round = max([m.get('matchday', 0) for m in matches if m.get('status') != 'finished'] or [1])

    print(f"Jornadas disponibles: {min(matchdays)} - {max(matchdays)}")
    print(f"Jornadas completadas: {completed}")
    print(f"Próxima jornada: {next_round}")

    print("\nOpciones:")
    print("1. Seleccionar jornada específica")
    print("2. Siguiente jornada no completada")
    print("3. Ver última jornada completada")

    try:
        choice = input("\nSelecciona opción (1-3): ")
    except ValueError:
        return

    if choice == "1":
        try:
            round_num = int(input("Ingresa el número de jornada: "))
        except ValueError:
            return
    elif choice == "2":
        round_num = next_round
    elif choice == "3":
        round_num = max(matchdays)
    else:
        return

    round_matches = [m for m in matches if m.get('matchday') == round_num]
    if not round_matches:
        print(f"No hay partidos en la jornada {round_num}.")
        return

    print(f"\nAnalizando jornada {round_num}...")

    local_wins = 0
    away_wins = 0
    draw_count = 0

    for m in round_matches:
        home = m.get('home_team', 'N/A')
        away = m.get('away_team', 'N/A')
        date = m.get('date', 'N/A')

        print(f"\n{'─' * 60}")
        print(f"⚽ PARTIDO: {home[:3].upper()} @ {away[:3].upper()}")
        print(f"📅 {date}")
        print("─" * 60)

        result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
        if not result:
            result = predict_match(home, away, matches)
        if result:
            pred = result['prediction']
            conf = result['confidence']

            if pred == "LOCAL":
                icon = "🏠"
                local_wins += 1
            elif pred == "VISITANTE":
                icon = "✈️"
                away_wins += 1
            else:
                icon = "🤝"
                draw_count += 1

            print(f"\n🎯 EL MODELO DICE: {icon} {pred}")
            print(f"   Confianza: {conf:.1f}%")
            print(f"   {home}: {result['home_prob']:.1f}% chance | {away}: {result['away_prob']:.1f}% chance | Empate: {result['draw_prob']:.1f}% chance")

            print(f"\n✅ HISTORIAL ({home} vs {away}):")
            h2h = result['head_to_head']
            print(f"   Victorias {home}: {h2h['home_wins']}")
            print(f"   Victorias {away}: {h2h['away_wins']}")
            print(f"   Empates: {h2h['draws']}")
        else:
            print("   Sin datos históricos para este encuentro.")

    print(f"\n{'─' * 60}")
    print(f"📊 RESUMEN JORNADA {round_num}")
    print("─" * 60)
    print(f"Victorias locales: {local_wins}")
    print(f"Victorias visitantes: {away_wins}")
    print(f"Empates: {draw_count}")
    print(f"Total partidos: {len(round_matches)}")

    total_conf = 0
    count = 0
    for m in round_matches:
        result = predict_match(m.get('home_team'), m.get('away_team'), matches)
        if result:
            total_conf += result['confidence']
            count += 1
    avg_conf = total_conf / count if count > 0 else 0
    print(f"Confianza promedio: {avg_conf:.1f}%")
    print(f"Modelo utilizado: {current_model}")
    print("=" * 60)
    input("\nPresiona Enter para continuar...")


def option_3_predict_match(teams, matches):
    print("\n" + "=" * 60)
    print("PREDICCIÓN PARTIDO POR PARTIDO")
    print("=" * 60)

    print("\nSelecciona dos equipos:")
    for i, team in enumerate(teams, 1):
        print(f"  {i:2}. {team}")

    try:
        home_num = int(input("\nIngresa número de equipo LOCAL: "))
        away_num = int(input("Ingresa número de equipo VISITANTE: "))
    except ValueError:
        print("Número inválido.")
        return

    home_team = get_team_by_number(teams, home_num)
    away_team = get_team_by_number(teams, away_num)

    if not home_team or not away_team:
        print("Equipo(s) inválido(s).")
        return

    result = predict_match(home_team, away_team, matches)

    print(f"\n{'=' * 60}")
    print(f"🏆 {home_team} vs {away_team}")
    print("=" * 60)

    if result:
        pred = result['prediction']
        conf = result['confidence']

        if pred == "LOCAL":
            icon = "🏠"
        elif pred == "VISITANTE":
            icon = "✈️"
        else:
            icon = "🤝"

        print(f"\n🎯 PREDICCIÓN: {icon} GANA {pred}")
        print(f"   Confianza: {conf:.1f}%")
        print(f"\n   Probabilidades:")
        print(f"   {home_team}: {result['home_prob']:.1f}%")
        print(f"   {away_team}: {result['away_prob']:.1f}%")
        print(f"   Empate: {result['draw_prob']:.1f}%")

        print(f"\n📊 Historial directo:")
        h2h = result['head_to_head']
        print(f"   {home_team}: {h2h['home_wins']} victorias")
        print(f"   {away_team}: {h2h['away_wins']} victorias")
        print(f"   Empates: {h2h['draws']}")
    else:
        print("No hay suficientes datos históricos para hacer una predicción.")

    input("\nPresiona Enter para continuar...")


def option_4_team_stats(teams, matches):
    print("\n" + "=" * 60)
    print("ESTADÍSTICAS DE EQUIPOS")
    print("=" * 60)

    if not teams:
        print("No hay equipos disponibles.")
        return

    try:
        team_num = int(input(f"Ingresa número de equipo (1-{len(teams)}): "))
    except ValueError:
        return

    team = get_team_by_number(teams, team_num)
    if not team:
        print("Equipo inválido.")
        return

    home_matches = [m for m in matches if m.get('home_team') == team]
    away_matches = [m for m in matches if m.get('away_team') == team]

    wins = sum(1 for m in home_matches if m.get('home_score') > m.get('away_score'))
    wins += sum(1 for m in away_matches if m.get('away_score') > m.get('home_score'))

    draws = sum(1 for m in home_matches if m.get('home_score') == m.get('away_score'))
    draws += sum(1 for m in away_matches if m.get('away_score') == m.get('home_score'))

    losses = sum(1 for m in home_matches if m.get('home_score') < m.get('away_score'))
    losses += sum(1 for m in away_matches if m.get('away_score') < m.get('home_score'))

    gf = sum(m.get('home_score', 0) for m in home_matches) + sum(m.get('away_score', 0) for m in away_matches)
    ga = sum(m.get('away_score', 0) for m in home_matches) + sum(m.get('home_score', 0) for m in away_matches)

    played = len(home_matches) + len(away_matches)
    pts = wins * 3 + draws

    print(f"\n{'=' * 50}")
    print(f"📊 {team}")
    print("=" * 50)
    print(f"Partidos jugados: {played}")
    print(f"Victorias: {wins} | Empates: {draws} | Derrotas: {losses}")
    print(f"Goles a favor: {int(gf)} | Goles en contra: {int(ga)} | Diferencia: {int(gf - ga)}")
    print(f"Puntos: {pts}")
    print(f"Rendimiento: {(pts / (played * 3) * 100):.1f}%" if played > 0 else "N/A")

    input("\nPresiona Enter para continuar...")


def option_5_rankings(matches):
    ranking = compute_rankings(matches)
    print_rankings(ranking)
    input("\nPresiona Enter para continuar...")


def get_phase_from_match(m):
    date_str = m.get('date', '')
    if not date_str:
        return 'SCHEDULED'
    
    try:
        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        month = date.month
        
        if month in [9, 10, 11, 12]:
            return 'GRUPO'
        elif month in [1, 2]:
            return '16AVOS'
        elif month == 3:
            return 'OCTAVOS'
        elif month == 4:
            if date.day <= 20:
                return 'CUARTOS'
            return 'SEMIS'
        elif month == 5:
            return 'FINAL'
        return 'UNKNOWN'
    except:
        return 'UNKNOWN'


def option_8_predict_by_phase_detailed(matches, teams):
    """Predicción detallada por fase con formato de análisis completo"""
    print("\n" + "=" * 60)
    print("PREDICCIÓN POR FASE (DETALLADA)")
    print("=" * 60)
    
    phases = {
        'GRUPO': 'Fase de Grupos',
        '16AVOS': 'Dieciseisavos (Playoff)',
        'OCTAVOS': 'Octavos de Final',
        'CUARTOS': 'Cuartos de Final',
        'SEMIS': 'Semifinales',
        'FINAL': 'Final'
    }
    
    print("\nSelecciona una fase:")
    print("1. Fase de Grupos")
    print("2. Dieciseisavos (Playoff)")
    print("3. Octavos de Final")
    print("4. Cuartos de Final")
    print("5. Semifinales")
    print("6. Final")
    
    choice = input("\nSelecciona opción (1-6): ")
    
    phase_map = {
        '1': 'GRUPO',
        '2': '16AVOS',
        '3': 'OCTAVOS',
        '4': 'CUARTOS',
        '5': 'SEMIS',
        '6': 'FINAL'
    }
    
    if choice not in phase_map:
        return
    
    phase = phase_map[choice]
    phase_matches = [m for m in matches if get_phase_from_match(m) == phase]
    
    if not phase_matches:
        print(f"\nNo hay partidos en la fase de {phases[phase]}.")
        input("\nPresiona Enter para continuar...")
        return
    
    # For knockout phases, ask about ida/vuelta
    leg_name = ""
    if phase in ['16AVOS', 'OCTAVOS', 'CUARTOS', 'SEMIS']:
        print(f"\nFase {phases[phase]}:")
        print("1. Todos los partidos")
        print("2. Solo IDA")
        print("3. Solo VUELTA")
        
        leg_choice = input("\nSelecciona opción (1-3): ")
        
        if leg_choice == "2":
            phase_matches = sorted(phase_matches, key=lambda x: x.get('date', ''))[:len(phase_matches)//2]
            leg_name = "(IDA)"
        elif leg_choice == "3":
            phase_matches = sorted(phase_matches, key=lambda x: x.get('date', ''))[len(phase_matches)//2:]
            leg_name = "(VUELTA)"
    
    # Header
    print("\n" + "=" * 60)
    print("PREDICCIÓN POR JORNADA (DETALLADA)")
    print("=" * 60)
    
    phase_display = f"UCL 2025/26 — {phases[phase]} {leg_name}"
    print(f"\n{phase_display}")
    
    completed = len([m for m in matches if m.get('status') == 'FINISHED'])
    print(f"Fases completadas: {completed}")
    print(f"Próxima fase: {phases[phase]} {leg_name}")
    print("=" * 60)
    
    # Stats
    local_wins = 0
    away_wins = 0
    draw_count = 0
    total_conf = 0
    
    # Process each match
    for m in phase_matches:
        home = m.get('home_team', 'N/A')
        away = m.get('away_team', 'N/A')
        date = m.get('date', 'N/A')
        status = m.get('status', 'SCHEDULED')
        
        # Format date
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(date.replace('Z', '+00:00'))
            date_formatted = dt.strftime("%d %b %Y · %H:%M CET")
        except:
            date_formatted = date[:10]
        
        print(f"\n{'─' * 60}")
        print(f"⚽ PARTIDO: {home[:3].upper()} @ {away[:3].upper()}")
        print(f"📅 {date_formatted}")
        print("─" * 60)
        
        result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
        if not result:
            result = predict_match(home, away, matches)
        
        if result:
            pred = result['prediction']
            conf = result['confidence']
            total_conf += conf
            
            if pred == "LOCAL":
                icon = "🏠"
                local_wins += 1
                winner = home
            elif pred == "VISITANTE":
                icon = "✈️"
                away_wins += 1
                winner = away
            else:
                icon = "🤝"
                draw_count += 1
                winner = "EMPATE"
            
            print(f"\n🎯 EL MODELO DICE: {icon} GANA {winner}")
            print(f"   Confianza: {conf:.1f}%")
            print(f"   {home}: {result['home_prob']:.1f}% chance | {away}: {result['away_prob']:.1f}% chance | Empate: {result['draw_prob']:.1f}% chance")
            
            # Generate REAL factors based on features
            print(f"\n✅ ¿POR QUÉ FAVORECE A {winner}?")
            print("─" * 60)
            
            factors_for = []
            numeric = result.get('numeric_features', {})
            
            if winner == home:
                if numeric.get('home_last5_w', 0) >= 3:
                    factors_for.append(f"Ganó {int(numeric.get('home_last5_w', 0))} de sus últimos 5")
                if numeric.get('home_uefa_coef', 0) > numeric.get('away_uefa_coef', 0):
                    diff = numeric.get('uefa_coef_diff', 0)
                    factors_for.append(f"Mejor coeficiente UEFA (+{diff:.0f})")
                if numeric.get('home_goals_for', 0) > numeric.get('away_goals_for', 0):
                    factors_for.append(f"Mejor ataque: {numeric.get('home_goals_for', 0):.1f} goles")
                if numeric.get('home_days_rest', 0) > numeric.get('away_days_rest', 0) + 1:
                    factors_for.append(f"Más días de descanso ({int(numeric.get('home_days_rest', 0))} vs {int(numeric.get('away_days_rest', 0))})")
                if numeric.get('h2h_home_w', 0) > numeric.get('h2h_away_w', 0):
                    factors_for.append(f"H2H favorable: {int(numeric.get('h2h_home_w', 0))}V vs {int(numeric.get('h2h_away_w', 0))}V")
            else:  # winner is away
                if numeric.get('away_last5_w', 0) >= 3:
                    factors_for.append(f"Ganó {int(numeric.get('away_last5_w', 0))} de sus últimos 5")
                if numeric.get('away_uefa_coef', 0) > numeric.get('home_uefa_coef', 0):
                    diff = numeric.get('uefa_coef_diff', 0)
                    factors_for.append(f"Mejor coeficiente UEFA (+{abs(diff):.0f})")
                if numeric.get('away_goals_for', 0) > numeric.get('home_goals_for', 0):
                    factors_for.append(f"Mejor ataque: {numeric.get('away_goals_for', 0):.1f} goles")
                if numeric.get('away_days_rest', 0) > numeric.get('home_days_rest', 0) + 1:
                    factors_for.append(f"Más días de descanso ({int(numeric.get('away_days_rest', 0))} vs {int(numeric.get('home_days_rest', 0))})")
                if numeric.get('h2h_away_w', 0) > numeric.get('h2h_home_w', 0):
                    factors_for.append(f"H2H favorable: {int(numeric.get('h2h_away_w', 0))}V vs {int(numeric.get('h2h_home_w', 0))}V")
            
            # Add some generic positive factors if not enough
            if len(factors_for) < 4:
                if conf > 55:
                    factors_for.append(f"Confianza del modelo: {conf:.0f}%")
                factors_for.append(f"Partido de eliminatoria - todo puede pasar")
            
            for i, f in enumerate(factors_for[:4], 1):
                print(f"  {i}. {f} ⭐")
            
            # Factors against
            print(f"\n❌ ¿QUÉ FAVORECE A {away if winner == home else home}?")
            print("─" * 60)
            
            factors_against = []
            loser = away if winner == home else home
            
            if loser == home:
                if numeric.get('home_last5_l', 0) >= 2:
                    factors_against.append(f"Perdió {int(numeric.get('home_last5_l', 0))} de sus últimos 5")
                if numeric.get('away_goals_for', 0) > numeric.get('home_goals_for', 0):
                    factors_against.append(f"Peor ataque que el rival: {numeric.get('home_goals_for', 0):.1f} vs {numeric.get('away_goals_for', 0):.1f}")
                if numeric.get('away_uefa_coef', 0) > numeric.get('home_uefa_coef', 0):
                    factors_against.append(f"Coeficiente UEFA inferior")
                if result['draw_prob'] > 20:
                    factors_against.append(f"Alta probabilidad de empate ({result['draw_prob']:.0f}%)")
            else:  # loser is away
                if numeric.get('away_last5_l', 0) >= 2:
                    factors_against.append(f"Perdió {int(numeric.get('away_last5_l', 0))} de sus últimos 5")
                if numeric.get('home_goals_for', 0) > numeric.get('away_goals_for', 0):
                    factors_against.append(f"Peor ataque que el rival")
                if numeric.get('home_uefa_coef', 0) > numeric.get('away_uefa_coef', 0):
                    factors_against.append(f"Coeficiente UEFA inferior")
                if result['draw_prob'] > 20:
                    factors_against.append(f"Alta probabilidad de empate ({result['draw_prob']:.0f}%)")
            
            # Add generic negative factors if not enough
            if len(factors_against) < 4:
                factors_against.append(f"Juega como visitante")
                factors_against.append(f"Partido de vuelta puede revertir todo")
            
            for i, f in enumerate(factors_against[:4], 1):
                print(f"  {i}. {f} ⭐")
    
    # Summary
    avg_conf = total_conf / len(phase_matches) if phase_matches else 0
    print(f"\n{'─' * 60}")
    print(f"📊 RESUMEN — UCL {phases[phase].upper()} {leg_name}")
    print("─" * 60)
    print(f"Victorias locales pred.:    {local_wins}")
    print(f"Victorias visitantes pred.: {away_wins}")
    print(f"Empates pred.:              {draw_count}")
    print(f"Total partidos:             {len(phase_matches)}")
    print(f"Confianza promedio:         {avg_conf:.1f}%")
    print(f"Modelo utilizado:           random_forest")
    print("─" * 60)
    
    input("\nPresiona Enter para continuar...")


def option_8_predict_by_phase(matches, teams):
    print("\n" + "=" * 60)
    print("PREDICCIÓN POR FASE")
    print("=" * 60)
    
    phases = {
        'GRUPO': 'Fase de Grupos',
        '16AVOS': 'Dieciseisavos (Playoff)',
        'OCTAVOS': 'Octavos de Final',
        'CUARTOS': 'Cuartos de Final',
        'SEMIS': 'Semifinales',
        'FINAL': 'Final'
    }
    
    print("\nFases disponibles:")
    for key, name in phases.items():
        count = len([m for m in matches if get_phase_from_match(m) == key])
        status = "✓" if count > 0 else " "
        print(f"  {status} {key}: {name} ({count} partidos)")
    
    print("\nSelecciona una fase:")
    print("1. Fase de Grupos")
    print("2. Dieciseisavos (Playoff)")
    print("3. Octavos de Final")
    print("4. Cuartos de Final")
    print("5. Semifinales")
    print("6. Final")
    print("7. Volver al menú principal")
    
    choice = input("\nSelecciona opción (1-7): ")
    
    phase_map = {
        '1': 'GRUPO',
        '2': '16AVOS',
        '3': 'OCTAVOS',
        '4': 'CUARTOS',
        '5': 'SEMIS',
        '6': 'FINAL'
    }
    
    if choice not in phase_map:
        return
    
    phase = phase_map[choice]
    phase_matches = [m for m in matches if get_phase_from_match(m) == phase]
    
    if not phase_matches:
        print(f"\nNo hay partidos en la fase de {phases[phase]}.")
        input("\nPresiona Enter para continuar...")
        return
    
    # For knockout phases, ask about ida/vuelta
    if phase in ['16AVOS', 'OCTAVOS']:
        print(f"\nFase {phases[phase]}:")
        print("1. Todos los partidos")
        print("2. Solo partidos de IDA")
        print("3. Solo partidos de VUELTA")
        
        leg_choice = input("\nSelecciona opción (1-3): ")
        
        # Separate by first leg vs second leg based on date
        if leg_choice == "2":
            # Ida: earlier dates (first leg)
            phase_matches = sorted(phase_matches, key=lambda x: x.get('date', ''))[:len(phase_matches)//2]
            leg_name = "IDA"
        elif leg_choice == "3":
            # Vuelta: later dates (second leg)
            phase_matches = sorted(phase_matches, key=lambda x: x.get('date', ''))[len(phase_matches)//2:]
            leg_name = "VUELTA"
        else:
            leg_name = "TODOS"
        
        if leg_name and leg_name != "TODOS":
            phase_title = f"{phases[phase]} ({leg_name})"
        else:
            phase_title = phases[phase]
    else:
        leg_name = None
        phase_title = phases[phase]
    
    # Check if phase is finished
    finished_matches = [m for m in phase_matches if m.get('status') == 'FINISHED']
    is_finished = len(finished_matches) == len(phase_matches) and len(phase_matches) > 0
    
    print(f"\n{'=' * 80}")
    if is_finished:
        print(f"RESULTADOS - {phase_title.upper()}")
        print(f"RESULTADOS - {phases[phase].upper()}")
    else:
        print(f"PREDICCIONES - {phase_title.upper()}")
    print("=" * 80)
    print(f"{'Local':<25} {'Visitante':<25} {'Predicción':<12} {'Confianza':<10} {'Resultado':<12}")
    print("-" * 80)
    
    if is_finished:
        correct = 0
        total = 0
        
        for m in phase_matches:
            home = m.get('home_team', 'N/A')
            away = m.get('away_team', 'N/A')
            home_score = m.get('home_score', 0)
            away_score = m.get('away_score', 0)
            
            # Determine actual result
            if home_score > away_score:
                actual = "LOCAL"
            elif home_score < away_score:
                actual = "VISITANTE"
            else:
                actual = "EMPATE"
            
            if home and away:
                result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
                if not result:
                    result = predict_match(home, away, matches)
                if result:
                    pred = result['prediction']
                    conf = result['confidence']
                    
                    # Check if prediction was correct
                    is_correct = pred == actual
                    if is_correct:
                        correct += 1
                        status = "OK"
                    else:
                        status = "ERROR"
                    total += 1
                    
                    score_str = f"{int(home_score)}-{int(away_score)}"
                    print(f"{home:<25} {away:<25} {pred:<12} {conf:.1f}%      {actual:<10} {score_str:<6} {status}")
                else:
                    score_str = f"{int(home_score)}-{int(away_score)}"
                    print(f"{home:<25} {away:<25} {'N/A':<12} {'--':<10}      {actual:<10} {score_str:<6}")
        
        accuracy = (correct / total * 100) if total > 0 else 0
        print("=" * 80)
        print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    else:
        # Regular prediction mode
        for m in phase_matches:
            home = m.get('home_team', 'N/A')
            away = m.get('away_team', 'N/A')
            status = m.get('status', 'N/A')
            
            if home and away:
                result = predict_match_ml(home, away, match_date=m.get('date'), matchday=m.get('matchday'))
                if not result:
                    result = predict_match(home, away, matches)
                if result:
                    pred = result['prediction']
                    conf = result['confidence']
                    print(f"{home:<25} {away:<25} {pred:<12} {conf:.1f}%      {status:<10}")
                else:
                    print(f"{home:<25} {away:<25} {'N/A':<12} {'--':<10}      {status:<10}")
        
        print("=" * 80)
    
    input("\nPresiona Enter para continuar...")


def option_6_change_model():
    global current_model
    print("\n" + "=" * 60)
    print("CAMBIAR MODO DE PREDICCIÓN")
    print("=" * 60)
    print("1. Modo Fase de Grupos")
    print("2. Modo Eliminatorias (16avos en adelante)")
    print("3. Cambiar tipo de modelo (RF, XGBoost, etc.)")
    
    choice = input("\nSelecciona opción (1-3): ")
    
    if choice == "1":
        current_model = "groups"
        print("Modo cambiado a: Fase de Grupos")
    elif choice == "2":
        current_model = "knockout"
        print("Modo cambiado a: Eliminatorias")
    elif choice == "3":
        print("\nTipo de modelo:")
        print("1. Random Forest")
        print("2. XGBoost")
        print("3. Gradient Boosting")
        
        model_choice = input("\nSelecciona modelo (1-3): ")
        
        model_types = {
            "1": "random_forest",
            "2": "xgboost",
            "3": "gradient_boosting"
        }
        
        if model_choice in model_types:
            print(f"Modelo cambiado a: {model_types[model_choice]}")
        else:
            print("Opción inválida.")
    else:
        print("Opción inválida.")

    input("\nPresiona Enter para continuar...")


def option_7_model_performance():
    import pickle
    import pandas as pd
    import numpy as np
    
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DE RENDIMIENTO DEL MODELO")
    print("=" * 70)
    
    # Cargar datos de entrenamiento
    league_config = LEAGUES[current_league]
    training_file = f'data/training/{league_config["dir"]}_training_full.csv'
    
    try:
        df = pd.read_csv(training_file)
    except:
        print("No hay datos de entrenamiento disponibles.")
        input("\nPresiona Enter para continuar...")
        return
    
    # Cargar modelo
    try:
        with open(league_config['model'], 'rb') as f:
            model = pickle.load(f)
    except:
        print("No hay modelo entrenado.")
        input("\nPresiona Enter para continuar...")
        return
    
    X = df[FEATURE_COLS].fillna(0)
    y = df['result']
    
    # Accuracy general
    preds = model.predict(X)
    overall_acc = (preds == y).mean() * 100
    
    print(f"\n📈 RENDIMIENTO GENERAL")
    print("-" * 40)
    print(f"Accuracy total: {overall_acc:.1f}%")
    print(f"Partidos analizados: {len(df)}")
    
    # Accuracy por fase
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
    
    print(f"\n📊 ACCURACY POR FASE")
    print("-" * 40)
    
    phases_order = ['GRUPO', '16AVOS', 'OCTAVOS', 'CUARTOS/SEMIS', 'FINAL']
    phase_results = {}
    
    for phase in phases_order:
        phase_df = df[df['phase'] == phase]
        if len(phase_df) > 0:
            X_phase = phase_df[FEATURE_COLS].fillna(0)
            y_phase = phase_df['result']
            preds_phase = model.predict(X_phase)
            acc = (preds_phase == y_phase).mean() * 100
            phase_results[phase] = acc
            print(f"  {phase:<15}: {acc:5.1f}% ({len(phase_df):3d} partidos)")
    
    # Accuracy por temporada
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
    
    print(f"\n📅 ACCURACY POR TEMPORADA")
    print("-" * 40)
    
    seasons = sorted(df['season'].unique())
    for season in seasons:
        season_df = df[df['season'] == season]
        if len(season_df) > 5:
            X_season = season_df[FEATURE_COLS].fillna(0)
            y_season = season_df['result']
            preds_season = model.predict(X_season)
            acc = (preds_season == y_season).mean() * 100
            print(f"  {season:<12}: {acc:5.1f}% ({len(season_df):3d} partidos)")
    
    # Análisis de tendencias (ventanas de 20 partidos)
    print(f"\n📉 TENDENCIAS (últimos 100 partidos)")
    print("-" * 40)
    
    df_sorted = df.sort_values('match_date').reset_index(drop=True)
    window = 20
    
    if len(df_sorted) >= window:
        # Calcular accuracy en ventanas rodantes
        rolling_accs = []
        dates = []
        
        for i in range(window, len(df_sorted), 5):  # Cada 5 partidos
            window_df = df_sorted.iloc[i-window:i]
            X_w = window_df[FEATURE_COLS].fillna(0)
            y_w = window_df['result']
            preds_w = model.predict(X_w)
            acc = (preds_w == y_w).mean() * 100
            rolling_accs.append(acc)
            dates.append(str(window_df.iloc[-1]['match_date'])[:10])
        
        # Mostrar últimos 5 puntos de tendencia
        print("  Últimos registros de tendencia:")
        for i in range(max(0, len(rolling_accs)-5), len(rolling_accs)):
            trend_indicator = "↑" if rolling_accs[i] > rolling_accs[i-1] else "↓" if rolling_accs[i] < rolling_accs[i-1] else "→"
            print(f"    {dates[i]}: {rolling_accs[i]:.1f}% {trend_indicator}")
        
        # Detectar picos y valles
        if len(rolling_accs) > 5:
            max_acc = max(rolling_accs[-10:])
            min_acc = min(rolling_accs[-10:])
            avg_acc = np.mean(rolling_accs[-10:])
            
            print(f"\n  📌 Análisis de los últimos 100 partidos:")
            print(f"     Mejor racha: {max_acc:.1f}%")
            print(f"     Peor racha: {min_acc:.1f}%")
            print(f"     Promedio: {avg_acc:.1f}%")
            
            # Detectar posibles caídas
            if rolling_accs[-1] < avg_acc - 10:
                print(f"     ⚠️  ALERTA: El modelo está rindiendo {avg_acc - rolling_accs[-1]:.1f}% por debajo del promedio")
            elif rolling_accs[-1] > avg_acc + 10:
                print(f"     🌟 PICO: El modelo está rindiendo {rolling_accs[-1] - avg_acc:.1f}% por encima del promedio")
    
    # Predicciones recientes
    print(f"\n🎯 ÚLTIMAS PREDICCIONES")
    print("-" * 40)
    
    recent_df = df_sorted.tail(20)
    correct_recent = (model.predict(recent_df[FEATURE_COLS].fillna(0)) == recent_df['result']).mean() * 100
    print(f"  Accuracy últimos 20 partidos: {correct_recent:.1f}%")
    
    print(f"\n📋 DATOS DEL MODELO")
    print("-" * 40)
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Dataset: {len(df)} partidos")
    print(f"  Modo actual: {'ELIMINATORIAS' if current_model == 'knockout' else 'GRUPO'}")
    
    input("\nPresiona Enter para continuar...")


def change_league():
    """Cambiar de competición."""
    global current_league, _model, _history
    
    print("\n" + "=" * 50)
    print("SELECCIONAR COMPETICIÓN")
    print("=" * 50)
    print("1. Champions League")
    print("2. Europa League")
    print("3. Conference League")
    
    choice = input("\nSelecciona competición (1-3): ")
    
    league_map = {
        "1": "champions",
        "2": "europa",
        "3": "conference"
    }
    
    if choice in league_map:
        new_league = league_map[choice]
        if new_league != current_league:
            current_league = new_league
            _model = None
            _history = None
            league_name = LEAGUES[current_league]['name']
            print(f"\n✓ Cambiado a {league_name}")
    else:
        print("\nOpción inválida.")


def main():
    global current_league
    
    print("=" * 50)
    print("PREDICTOR UEFA")
    print("=" * 50)
    
    # Seleccionar competición al inicio
    change_league()
    
    league_name = LEAGUES[current_league]['name']
    print(f"\nCompetición actual: {league_name}")
    
    if current_league == 'champions':
        auto_update_champions_current_season()

    matches = load_cleaned_matches()
    teams = load_teams()

    if not matches:
        print("No hay datos disponibles.")
        return

    while True:
        print("\n" + "=" * 50)
        print(f"PREDICTOR {league_name.upper()}")
        print("=" * 50)
        print("1.  Predicción de jornada completa")
        print("2.  Predicción por jornada (detalles)")
        print("3.  Predicción por fase")
        print("4.  Estadísticas de equipos")
        print("5.  Cambiar modelo de predicción")
        print("6.  Rendimiento de modelos")
        print("7.  Cambiar competición")
        print("8.  Salir")
        print("=" * 50)

        option = input("Selecciona una opción (1-8): ")

        if option == "1":
            option_1_predict_round(matches, teams)
        elif option == "2":
            option_2_predict_round_detailed(matches, teams)
        elif option == "3":
            option_8_predict_by_phase(matches, teams)
        elif option == "4":
            option_4_team_stats(teams, matches)
        elif option == "5":
            option_6_change_model()
        elif option == "6":
            option_7_model_performance()
        elif option == "7":
            change_league()
            league_name = LEAGUES[current_league]['name']
            matches = load_cleaned_matches()
            teams = load_teams()
        elif option == "8":
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción inválida.")


if __name__ == '__main__':
    main()
