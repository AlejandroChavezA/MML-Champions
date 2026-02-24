"""
Módulo compartido para normalizar nombres de equipos.
Carga el mapeo desde config/teams_mapping.csv (alias -> canonical).
Edita ese CSV para añadir nuevas correlaciones.
"""

import csv
from pathlib import Path
from typing import Dict

_MAPPING: Dict[str, str] = {}
_LOADED = False


def _load_mapping() -> Dict[str, str]:
    global _MAPPING, _LOADED
    if _LOADED:
        return _MAPPING
    path = Path(__file__).parent / "teams_mapping.csv"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                alias = row.get("alias", "").strip()
                canonical = row.get("canonical", "").strip()
                if alias and canonical:
                    _MAPPING[alias] = canonical
    _LOADED = True
    return _MAPPING


def normalize_team_name(name: str) -> str:
    """
    Convierte un nombre de equipo a su forma canónica.
    Si no hay mapeo, devuelve el nombre tal cual (sin 'v ' al inicio).
    """
    s = name.strip()
    if s.startswith("v "):
        s = s[2:].strip()
    mapping = _load_mapping()
    return mapping.get(s, s)


def get_mapping() -> Dict[str, str]:
    """Devuelve el diccionario alias -> canonical (para scripts que lo necesiten)."""
    return _load_mapping().copy()
