#!/usr/bin/env python3
"""
Lista todos los nombres únicos de equipos que aparecen en los matches.
Sirve para detectar duplicados (mismo club, distinto nombre) y añadirlos a config/teams_mapping.csv.

Uso:
  python3 discover_team_aliases.py          # Lista todos
  python3 discover_team_aliases.py --similar # Sugiere pares similares (requiere python-Levenshtein o difflib)
"""

import csv
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set


def extract_all_team_names(cleaned_base: Path) -> Dict[str, Set[str]]:
    """Extrae {team_name: set(archivos donde aparece)}."""
    by_name: dict[str, set[str]] = defaultdict(set)
    for comp in ("champions", "europa_league", "conference"):
        folder = cleaned_base / comp
        if not folder.is_dir():
            continue
        for csv_path in folder.glob("matches_*.csv"):
            rel = f"{comp}/{csv_path.name}"
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = row.get("home_team", "").strip()
                    a = row.get("away_team", "").strip()
                    if h:
                        by_name[h].add(rel)
                    if a:
                        by_name[a].add(rel)
    return dict(by_name)


def similar_pairs(names: list, threshold: float = 0.7) -> list:
    """Encuentra pares de nombres similares usando difflib (sin dependencias extra)."""
    import difflib
    pairs = []
    sorted_names = sorted(names)
    for i, a in enumerate(sorted_names):
        for b in sorted_names[i + 1 :]:
            r = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if r >= threshold and r < 1.0:
                pairs.append((a, b, round(r, 2)))
    return sorted(pairs, key=lambda x: -x[2])


def main():
    cleaned = Path("data/cleaned")
    if not cleaned.exists():
        print("No existe data/cleaned/")
        return

    by_name = extract_all_team_names(cleaned)
    names = sorted(by_name.keys())
    print(f"Total de nombres únicos: {len(names)}\n")

    if "--similar" in sys.argv or "-s" in sys.argv:
        print("Pares similares (posibles duplicados):")
        print("-" * 60)
        pairs = similar_pairs(names)
        for a, b, score in pairs[:50]:
            print(f"  {score:.2f}  {a}")
            print(f"       {b}")
            print()
    else:
        print("Lista de equipos (para revisar duplicados manualmente):")
        print("-" * 60)
        for n in names:
            files = ", ".join(sorted(by_name[n])[:3])
            if len(by_name[n]) > 3:
                files += f" (+{len(by_name[n])-3} más)"
            print(f"  {n}")
        print()
        print("Para sugerencias de pares similares: python3 discover_team_aliases.py --similar")
        print("Para añadir correlaciones: edita config/teams_mapping.csv (alias,canonical)")


if __name__ == "__main__":
    main()
