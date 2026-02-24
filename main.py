#!/usr/bin/env python3
"""
Champions League Predictor - Cleaned UI (data/cleaned)
====================================================
Minimal, safe UI using a canonical data layer in data/cleaned.
"""

import json
from pathlib import Path
from datetime import datetime
import csv

BASE_YEAR = "2025-26"
CLEAN_ROOT = Path("data/cleaned")
CURRENT_DIR = CLEAN_ROOT / BASE_YEAR
CURRENT_MATCHES_FILE = CURRENT_DIR / "champions_matches.json"
TEAMS_FILE = CURRENT_DIR / "champions_teams.csv"

# Helpers

def load_cleaned_matches(year: str = BASE_YEAR):
    p = CLEAN_ROOT / year / "champions_matches.json"
    if not p.exists():
        return []
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_teams(year: str = BASE_YEAR):
    path = CLEAN_ROOT / year / "champions_teams.csv"
    teams = []
    if not path.exists():
        return teams
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row:
                teams.append(row[0])
    return sorted(list(dict.fromkeys(teams)))


def compute_rankings(matches):
    standings = {}
    for m in matches:
        h = m.get('home_team')
        a = m.get('away_team')
        sh = m.get('home_score')
        sa = m.get('away_score')
        if h is None or a is None or sh is None or sa is None:
            continue
        if h not in standings:
            standings[h] = {'played':0,'wins':0,'draws':0,'losses':0,'gf':0,'ga':0,'pts':0}
        if a not in standings:
            standings[a] = {'played':0,'wins':0,'draws':0,'losses':0,'gf':0,'ga':0,'pts':0}
        standings[h]['played'] += 1
        standings[a]['played'] += 1
        standings[h]['gf'] += sh; standings[h]['ga'] += sa
        standings[a]['gf'] += sa; standings[a]['ga'] += sh
        if sh > sa:
            standings[h]['wins'] += 1; standings[h]['pts'] += 3
            standings[a]['losses'] += 1
        elif sh < sa:
            standings[a]['wins'] += 1; standings[a]['pts'] += 3
            standings[h]['losses'] += 1
        else:
            standings[h]['draws'] += 1; standings[a]['draws'] += 1
            standings[h]['pts'] += 1; standings[a]['pts'] += 1

    # Build ranking list
    ranking = []
    for t, s in standings.items():
        gd = s['gf'] - s['ga']
        ranking.append((t, s['pts'], s['played'], s['wins'], s['draws'], s['losses'], gd, s['gf'], s['ga']))
    ranking.sort(key=lambda x: (x[1], x[6]), reverse=True)
    return ranking


def print_rankings(ranking):
    print("\nTop 20 equipos (ranking):")
    print("Pos  Equipo                 Pts  PJ  V  E  D  GD  GF  GA")
    print("----" * 5)
    for i, row in enumerate(ranking[:20], 1):
        t, pts, played, wins, draws, losses, gd, gf, ga = row
        print(f"{i:<4} {t:<22} {pts:<4} {played:<4} {wins:<4} {draws:<4} {losses:<4} {gd:+d} {gf:<4} {ga:<4}")


def main():
    print("=" * 40)
    print("PREDICTOR CHAMPIONS LEAGUE (data/cleaned)")
    print("=" * 40)

    # Cargar datos limpios actuales
    matches = load_cleaned_matches()
    teams = load_teams()

    if not matches:
        print("No hay datos limpios disponibles en data/cleaned/2025-26.")
        print("Por favor genera data limpia con el pipeline y ejecuta de nuevo.")
        return

    # 1) Predicción partido por partido (esbozo)
    print("\n3. Prediccion partido por partido")
    print("Requiere selección de dos equipos (Mostrando lista corta)...")
    print("Ejemplo: seleccionar 1 y 5 para equipos 1 y 5 de la lista")

    # 2) Estadísticas de equipos
    trend = compute_rankings(matches)
    print_rankings(trend)

    print("\n8. Salir")

if __name__ == '__main__':
    main()
