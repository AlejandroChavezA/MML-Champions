#!/usr/bin/env python3
"""
Champions League Data Clean Pipeline Minimal (Canónico en data/cleaned)
===========================================
Este pipeline genera data/cleaned/2025-26 a partir de data/dataReal
Mantiene intactos los datos originales.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import re

YEAR = "2025-26"
CLEAN_ROOT = Path("data/cleaned")
CLEAN_DIR = CLEAN_ROOT / YEAR
DATA_REAL = Path("data/dataReal")
TEAMS_2025_26 = Path("data/teams_2025_26.csv")

MONTHS = {
    'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
    'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12
}

def load_teams(path: Path):
    teams = []
    if not path.exists():
        return teams
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                name = line.split(',')[0].strip()
            else:
                name = line.strip()
            if name:
                teams.append(name)
    seen = set(); uniq = []
    for t in teams:
        if t not in seen:
            uniq.append(t); seen.add(t)
    return uniq


def parse_date_from_line(line: str):
    m = re.search(r'([A-Za-z]{3})/(\d{1,2})\s+(\d{4})', line)
    if not m:
        return None
    mon = m.group(1); day = int(m.group(2)); year = int(m.group(3))
    mon_num = MONTHS.get(mon[:3], 1)
    return f"{year}-{mon_num:02d}-{day:02d}"


def parse_cl(year_dir: Path):
    cl_file = year_dir / 'cl.txt'
    if not cl_file.exists():
        return []
    matches = []
    current_date = None
    current_stage = 'League'
    current_matchday = None
    with open(cl_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for raw in lines:
        line = raw.rstrip('\n')
        d = parse_date_from_line(line)
        if d:
            current_date = d
            continue
        if 'Matchday' in line:
            m = re.search(r'Matchday\s+(\d+)', line, re.IGNORECASE)
            if m:
                current_matchday = int(m.group(1))
        if '»' in line:
            seg = line.replace('»','').strip()
            current_stage = seg
            continue
        m = re.match(r'^\s*([0-9]{2}\.[0-9]{2})\s+([^()]+)\s*\([A-Z]{3}\)\s*v\s+([^()]+)\s*\([A-Z]{3}\)\s*([0-9]+)-([0-9]+)', line)
        if m:
            time = m.group(1)
            home_raw = m.group(2).strip()
            away_raw = m.group(3).strip()
            home_score = int(m.group(4)); away_score = int(m.group(5))
            home = home_raw.strip(); away = away_raw.strip()
            home = re.sub(r'\s*\([A-Z]{3}\)$','', home).strip()
            away = re.sub(r'\s*\([A-Z]{3}\)$','', away).strip()
            matches.append({'date': current_date, 'time': time, 'home_team': home, 'away_team': away, 'home_score': home_score, 'away_score': away_score, 'stage': current_stage})
    return matches


def to_canonic(m, season=YEAR):
    h = int(m.get('home_score',0)); a = int(m.get('away_score',0))
    date_iso = f"{m['date']} {m['time']}:00+00:00" if m['date'] and m['time'] else ""
    total_goals = h + a
    gd = h - a
    result = 'LOCAL' if h>a else ('VISITANTE' if h<a else 'EMPATE')
    return {
        'id': f"{season}-{(hash((m['home_team'], m['away_team'], m['date'], m['time']))%1000000):06d}",
        'date': date_iso,
        'matchday': 1,
        'home_team': m['home_team'],
        'away_team': m['away_team'],
        'home_score': h,
        'away_score': a,
        'status': 'FINISHED' if (m['home_score'] is not None and m['away_score'] is not None) else 'SCHEDULED',
        'total_goals': total_goals,
        'goal_difference': gd,
        'result': result
    }


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_YEAR_DIR = CLEAN_DIR / YEAR
    CLEAN_YEAR_DIR.mkdir(parents=True, exist_ok=True)

    year_dir = DATA_REAL / YEAR
    teams = load_teams(TEAMS_2025_26)

    raw = parse_cl(year_dir)
    canonic = []
    idx = 1
    for m in raw:
        if m.get('home_team') and m.get('away_team'):
            if teams and (m['home_team'] not in teams or m['away_team'] not in teams):
                continue
            cm = {
                'id': f"{YEAR}-{idx:06d}",
                'date': f"{m['date']} {m['time']}:00+00:00" if m['date'] and m['time'] else '',
                'matchday': 1,
                'home_team': m['home_team'], 'away_team': m['away_team'],
                'home_score': m['home_score'], 'away_score': m['away_score'],
                'status': 'FINISHED',
                'total_goals': int(m['home_score']) + int(m['away_score']),
                'goal_difference': int(m['home_score']) - int(m['away_score']),
                'result': 'LOCAL' if m['home_score'] > m['away_score'] else ('VISITANTE' if m['home_score'] < m['away_score'] else 'EMPATE')
            }
            canonic.append(cm)
            idx += 1

    with open(CLEAN_YEAR_DIR / 'champions_matches.json','w',encoding='utf-8') as f:
        json.dump(canonic, f, indent=2, ensure_ascii=False)

    unique = sorted(set([m['home_team'] for m in canonic] + [m['away_team'] for m in canonic]))
    with open(CLEAN_YEAR_DIR / 'champions_teams.csv','w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['team_name','season'])
        for t in unique:
            w.writerow([t, YEAR])

    summary = {
        'season': YEAR,
        'partidos': len(canonic),
        'equipos': len(unique),
        'generated_at': datetime.utcnow().isoformat() + 'Z'
    }
    with open(CLEAN_YEAR_DIR / 'summary.json','w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("OK! Clean data generated.")

if __name__ == '__main__':
    main()
