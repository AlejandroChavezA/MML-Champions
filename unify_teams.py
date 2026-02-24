#!/usr/bin/env python3
"""
Unifica nombres de equipos en all_teams.csv y genera:
- data/all_teams.csv (limpiado y unificado)
- data/teams_2025.csv (equipos 2024-25 y 2025-26 que podamos sacar)
- data/cleaned/{champions,europa_league,conference}/teams_YYYY_YY.csv por temporada

El mapeo alias -> canónico se carga desde config/teams_mapping.csv.
Edita ese archivo para añadir nuevas correlaciones.
"""

import csv
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

try:
    from config.team_names import normalize_team_name
except ImportError:
    def normalize_team_name(name: str) -> str:
        s = name.strip()
        return s[2:].strip() if s.startswith("v ") else s


def load_and_unify_all_teams(path: Path) -> Dict[str, Set[str]]:
    """Carga all_teams.csv, unifica nombres y devuelve {canonical_name: set(seasons)}."""
    by_canonical: Dict[str, Set[str]] = defaultdict(set)
    seen_raw = set()

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["team_name", "seasons"]:
            raise ValueError("all_teams.csv debe tener columnas team_name, seasons")
        for row in reader:
            raw = row["team_name"].strip()
            if not raw or raw in seen_raw:
                continue
            seen_raw.add(raw)
            canonical = normalize_team_name(raw)
            # Ignorar nombres que parecen truncados (muy cortos)
            if len(canonical) < 4:
                continue
            seasons_str = row.get("seasons", "").strip()
            for s in re.split(r",\s*", seasons_str):
                s = s.strip().strip('"')
                if s:
                    by_canonical[canonical].add(s)

    return dict(by_canonical)


def write_unified_all_teams(by_canonical: Dict[str, Set[str]], path: Path) -> None:
    """Escribe all_teams unificado: team_name, seasons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(by_canonical.keys()):
        seasons = sorted(by_canonical[name])
        rows.append({"team_name": name, "seasons": ", ".join(seasons)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team_name", "seasons"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ all_teams unificado: {path} ({len(rows)} equipos)")


def extract_teams_from_matches_csv(matches_path: Path) -> Set[str]:
    """Extrae conjunto de nombres de equipos (home + away) de un CSV de partidos."""
    teams: Set[str] = set()
    with open(matches_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = row.get("home_team", "").strip()
            a = row.get("away_team", "").strip()
            if h:
                teams.add(h)
            if a:
                teams.add(a)
    return teams


def season_from_filename(filename: str) -> str:
    """matches_2024_25.csv -> 2024-25"""
    m = re.match(r"matches_(\d{4})_(\d{2})\.csv", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return ""


def generate_teams_per_competition(cleaned_base: Path) -> None:
    """
    Un CSV de equipos por temporada dentro de cada Liga (champions, europa_league, conference).
    Estructura paralela a matches: id, team_name, season.
    """
    for comp in ("champions", "europa_league", "conference"):
        folder = cleaned_base / comp
        if not folder.is_dir():
            continue
        for csv_path in sorted(folder.glob("matches_*.csv")):
            season = season_from_filename(csv_path.name)
            if not season:
                continue
            teams = sorted(extract_teams_from_matches_csv(csv_path))
            out_name = f"teams_{season.replace('-', '_')}.csv"
            out_path = folder / out_name
            rows = [
                {"id": idx + 1, "team_name": t, "season": season}
                for idx, t in enumerate(teams)
            ]
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["id", "team_name", "season"])
                w.writeheader()
                w.writerows(rows)
            print(f"  {comp}/teams_{season.replace('-', '_')}.csv: {len(rows)} equipos")


def build_teams_2025(cleaned_base: Path, data_path: Path) -> None:
    """Construye data/teams_2025.csv con equipos de 2024-25 y 2025-26 (todas competiciones)."""
    teams_seasons: List[Tuple[str, str, str]] = []  # (team_name, season, competition)
    for comp in ("champions", "europa_league", "conference"):
        folder = cleaned_base / comp
        if not folder.is_dir():
            continue
        for season in ("2024_25", "2025_26"):
            path = folder / f"matches_{season}.csv"
            if not path.exists():
                continue
            season_dash = season.replace("_", "-")
            for t in extract_teams_from_matches_csv(path):
                teams_seasons.append((t, season_dash, comp))

    # Unificar por (team_name, season) y añadir competition (podemos concatenar si aparece en varias)
    by_team_season: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for name, season, comp in teams_seasons:
        by_team_season[(name, season)].add(comp)

    rows = []
    for (name, season), comps in sorted(by_team_season.items()):
        rows.append({
            "team_name": name,
            "season": season,
            "competitions": ", ".join(sorted(comps)),
        })

    out_path = data_path / "teams_2025.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["team_name", "season", "competitions"])
        w.writeheader()
        w.writerows(rows)
    print(f"✓ teams_2025.csv: {out_path} ({len(rows)} filas)")


def main() -> None:
    data_path = Path("data")
    all_teams_path = data_path / "all_teams.csv"
    cleaned_base = data_path / "cleaned"

    if not all_teams_path.exists():
        print("No existe data/all_teams.csv")
        return

    # 1) Unificar y guardar all_teams
    by_canonical = load_and_unify_all_teams(all_teams_path)
    write_unified_all_teams(by_canonical, all_teams_path)

    # 2) Generar teams por temporada y competición (desde partidos)
    print("Generando teams por temporada y competición...")
    generate_teams_per_competition(cleaned_base)

    # 3) teams_2025.csv
    build_teams_2025(cleaned_base, data_path)


if __name__ == "__main__":
    main()
