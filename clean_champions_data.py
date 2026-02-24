#!/usr/bin/env python3
"""
Script para limpiar y convertir los datos de Champions League de data/dataReal
al formato CSV estándar de matches_2025.csv
"""

import os
import re
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

try:
    from config.team_names import normalize_team_name
except ImportError:
    def normalize_team_name(name: str) -> str:
        s = name.strip()
        return s[2:].strip() if s.startswith("v ") else s

def parse_date(date_str: str, year: int, time_str: str = "20:45") -> Optional[str]:
    """
    Parsea una fecha del formato del archivo a ISO format.
    Ejemplos:
    - "Tue Sep/13 2011" -> "2011-09-13T20:45:00Z"
    - "Tue Sep/17" -> "2024-09-17T20:45:00Z" (usa el año proporcionado)
    """
    # Patrón para fechas con año completo
    pattern_with_year = r'(\w+)\s+(\w+)/(\d+)\s+(\d{4})'
    match = re.search(pattern_with_year, date_str)
    
    if match:
        _, month, day, year_str = match.groups()
        year = int(year_str)
    else:
        # Patrón para fechas sin año
        pattern_no_year = r'(\w+)\s+(\w+)/(\d+)'
        match = re.search(pattern_no_year, date_str)
        if not match:
            return None
        _, month, day = match.groups()
    
    # Mapeo de meses abreviados
    month_map = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }
    
    month_num = month_map.get(month[:3])
    if not month_num:
        return None
    
    # Formato: YYYY-MM-DDTHH:MM:SSZ
    return f"{year}-{month_num}-{day.zfill(2)}T{time_str}:00Z"


def parse_time(time_str: str) -> str:
    """Convierte hora del formato '20.45' a '20:45'"""
    return time_str.replace('.', ':')


def extract_teams(match_line: str) -> Optional[Tuple[str, str]]:
    """
    Extrae los nombres de los equipos de una línea de partido.
    Ejemplo: "Manchester City (ENG)   v SSC Napoli (ITA)" -> ("Manchester City", "SSC Napoli")
    """
    # Patrón para encontrar equipos con países
    pattern = r'(.+?)\s+\([A-Z]{3}\)\s+v\s+(.+?)\s+\([A-Z]{3}\)'
    match = re.search(pattern, match_line)
    
    if match:
        home_team = match.group(1).strip()
        away_team = match.group(2).strip()
        
        # Eliminar hora del inicio si existe (formato "20.45 " o "18.00 ")
        home_team = re.sub(r'^\d{1,2}\.\d{2}\s+', '', home_team)
        away_team = re.sub(r'^\d{1,2}\.\d{2}\s+', '', away_team)
        
        return (home_team, away_team)
    
    return None


def extract_score(score_str: str) -> Optional[Tuple[float, float]]:
    """
    Extrae el resultado del partido.
    Ejemplos:
    - "1-1 (0-0)" -> (1.0, 1.0)
    - "2-0" -> (2.0, 0.0)
    - "4-3 pen. 1-0 a.e.t. (1-0, 1-0)" -> (4.0, 3.0) (resultado final después de penales)
    """
    # Primero buscar resultado después de "pen." (penales) - este es el resultado final
    pen_pattern = r'pen\.\s+(\d+)-(\d+)'
    pen_match = re.search(pen_pattern, score_str)
    if pen_match:
        home_score = float(pen_match.group(1))
        away_score = float(pen_match.group(2))
        return (home_score, away_score)
    
    # Si no hay penales, buscar el primer resultado (antes del paréntesis si existe)
    pattern = r'(\d+)-(\d+)'
    match = re.search(pattern, score_str)
    
    if match:
        home_score = float(match.group(1))
        away_score = float(match.group(2))
        return (home_score, away_score)
    
    return None


def extract_matchday(line: str, current_matchday: Optional[int] = None) -> Optional[int]:
    """
    Extrae el número de matchday de una línea.
    Ejemplos:
    - "» Group A" -> None (grupos no tienen número, mantiene el anterior)
    - "» League, Matchday 1" -> 1
    - "» Matchday 5" -> 5
    - "» Round of 16" -> None (mantiene el anterior)
    """
    # Buscar "Matchday X"
    pattern = r'Matchday\s+(\d+)'
    match = re.search(pattern, line, re.IGNORECASE)
    
    if match:
        return int(match.group(1))
    
    # Si no hay matchday explícito, mantener el anterior
    return current_matchday


def parse_champions_file(file_path: Path) -> List[Dict]:
    """
    Parsea un archivo de Champions League y retorna una lista de partidos.
    """
    matches = []
    
    # Extraer año de la temporada del nombre del directorio
    # Ejemplo: "data/dataReal/2011-12/cl.txt" -> 2011
    dir_name = file_path.parent.name
    year_match = re.search(r'(\d{4})', dir_name)
    if year_match:
        base_year = int(year_match.group(1))
    else:
        # Intentar extraer del contenido del archivo
        base_year = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    current_date = None
    current_time = "20:45"
    current_matchday = None
    match_id_counter = 1
    last_year = base_year
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Buscar matchday
        current_matchday = extract_matchday(line, current_matchday)
        
        # Buscar fecha
        date_pattern = r'(\w+)\s+(\w+)/(\d+)(?:\s+(\d{4}))?'
        date_match = re.search(date_pattern, line)
        if date_match:
            date_str = line
            if date_match.group(4):  # Tiene año
                year = int(date_match.group(4))
                last_year = year
            elif last_year:
                year = last_year
            else:
                year = None
            
            if year:
                # Buscar hora en la misma línea o siguiente
                time_pattern = r'(\d{1,2})\.(\d{2})'
                time_match = re.search(time_pattern, line)
                if time_match:
                    current_time = parse_time(time_match.group(0))
                else:
                    # Buscar en la siguiente línea
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        time_match = re.search(time_pattern, next_line)
                        if time_match:
                            current_time = parse_time(time_match.group(0))
                
                current_date = parse_date(date_str, year, current_time)
        
        # Buscar hora (puede estar en línea separada)
        time_pattern = r'(\d{1,2})\.(\d{2})'
        time_match = re.search(time_pattern, line)
        if time_match:
            current_time = parse_time(time_match.group(0))
            if current_date:
                # Actualizar la hora en la fecha
                date_part = current_date.split('T')[0]
                current_date = f"{date_part}T{current_time}:00Z"
        
        # Buscar línea de partido
        if 'v' in line and '(' in line and ')' in line:
            teams = extract_teams(line)
            if teams:
                home_team, away_team = teams
                
                # Buscar resultado en la misma línea o siguiente
                score_line = line
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if re.search(r'\d+-\d+', next_line) and not re.search(r'v', next_line):
                        score_line = next_line
                
                score = extract_score(score_line)
                
                if score and current_date:
                    home_score, away_score = score
                    # Normalizar nombres a canónico (para correlación con all_teams y teams)
                    home_canonical = normalize_team_name(home_team)
                    away_canonical = normalize_team_name(away_team)

                    match_id = int(datetime.now().timestamp() * 1000) + match_id_counter
                    match_id_counter += 1

                    match_data = {
                        'id': match_id,
                        'date': current_date,
                        'matchday': current_matchday if current_matchday else 1,
                        'home_team': home_canonical,
                        'away_team': away_canonical,
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': 'FINISHED'
                    }
                    
                    matches.append(match_data)
        
        i += 1
    
    return matches


def get_competition_from_file(file_path: Path) -> str:
    """
    Determina la competición según el nombre del archivo.
    cl, clq -> champions
    el, elq -> europa_league
    conf, confq -> conference
    """
    name = file_path.stem.lower()  # cl, el, conf, clq, elq, confq
    if name in ('cl', 'clq'):
        return 'champions'
    if name in ('el', 'elq'):
        return 'europa_league'
    if name in ('conf', 'confq'):
        return 'conference'
    return 'champions'  # por defecto


def main():
    """
    Función principal que procesa todos los archivos de data/dataReal
    y genera un archivo CSV por temporada y competición.
    Carpetas: champions/, europa_league/, conference/
    """
    data_real_path = Path('data/dataReal')
    base_output = Path('data/cleaned')
    
    # Carpetas por competición
    competitions = {
        'champions': base_output / 'champions',
        'europa_league': base_output / 'europa_league',
        'conference': base_output / 'conference',
    }
    for folder in competitions.values():
        folder.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos .txt en data/dataReal
    txt_files = list(data_real_path.rglob('*.txt'))
    
    print(f"Encontrados {len(txt_files)} archivos .txt para procesar")
    
    # Agrupar por competición y temporada: {competition: {season: [matches]}}
    matches_by_competition: Dict[str, Dict[str, List[Dict]]] = {
        'champions': {},
        'europa_league': {},
        'conference': {},
    }
    
    for txt_file in txt_files:
        season = txt_file.parent.name
        competition = get_competition_from_file(txt_file)
        
        if season not in matches_by_competition[competition]:
            matches_by_competition[competition][season] = []
        
        print(f"Procesando: {txt_file} -> {competition}/")
        try:
            matches = parse_champions_file(txt_file)
            matches_by_competition[competition][season].extend(matches)
            print(f"  -> {len(matches)} partidos")
        except Exception as e:
            print(f"  -> Error: {e}")
            continue
    
    fieldnames = ['id', 'date', 'matchday', 'home_team', 'away_team', 
                 'home_score', 'away_score', 'status']
    
    total_matches = 0
    for competition, output_folder in competitions.items():
        seasons = matches_by_competition[competition]
        for season in sorted(seasons.keys()):
            matches = seasons[season]
            matches.sort(key=lambda x: x['date'])
            
            filename = f"matches_{season.replace('-', '_')}.csv"
            output_file = output_folder / filename
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(matches)
            
            total_matches += len(matches)
            print(f"✓ {competition}/{filename}: {len(matches)} partidos")
    
    if total_matches > 0:
        print(f"\n✓ Total: {total_matches} partidos")
        print(f"✓ Carpetas: {list(competitions.keys())}")
    else:
        print("\n✗ No se encontraron partidos para procesar")


if __name__ == '__main__':
    main()
