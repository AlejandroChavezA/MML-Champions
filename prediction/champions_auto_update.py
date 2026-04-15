#!/usr/bin/env python3
"""
Actualización automática de datos de Champions League (temporada actual).

Objetivo:
- Al iniciar el programa, refrescar los fixtures de Champions desde la API
  y escribirlos en el mismo formato que usamos en data/cleaned/champions/matches_YYYY_YY.csv

IMPORTANTE:
- Usa Football-Data.org API (https://api.football-data.org/v4)
- Configura tu API key y base URL en config/api_config.py
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List

import requests

try:
    from config.api_config import API_FOOTBALL_KEY, API_FOOTBALL_BASE_URL, HEADERS
except ImportError:
    API_FOOTBALL_KEY = "fd9ecc768e3644dfa9b30e9536031700"
    API_FOOTBALL_BASE_URL = "https://api.football-data.org/v4"
    HEADERS = {'X-Auth-Token': API_FOOTBALL_KEY}


def _can_call_api() -> bool:
    token = HEADERS.get('X-Auth-Token', '')
    return bool(token and "TU_API" not in token)




def fetch_champions_fixtures_api(season_start_year: int) -> List[Dict[str, Any]]:
    """
    Llama a football-data.org API para Champions League.
    """
    if not _can_call_api():
        print("⚠️  API token no configurada; se omite actualización automática de Champions.")
        return []

    url = f"{API_FOOTBALL_BASE_URL}/competitions/CL/matches"
    params = {"season": season_start_year}
    print(f"🌐 Llamando a Football-Data API Champions: {url} (season={season_start_year})")
    resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return data.get("matches", [])


def fixtures_to_matches_rows(fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from config.team_names import normalize_team_name
    
    rows: List[Dict[str, Any]] = []
    for item in fixtures:
        match_id = item.get("id")
        date_iso = item.get("utcDate")
        stage = item.get("stage", "")
        matchday = item.get("matchday")
        home_team_raw = (item.get("homeTeam") or {}).get("name")
        away_team_raw = (item.get("awayTeam") or {}).get("name")
        home_team = normalize_team_name(home_team_raw) if home_team_raw else ""
        away_team = normalize_team_name(away_team_raw) if away_team_raw else ""
        score = item.get("score") or {}
        home_score = score.get("fullTime", {}).get("home")
        away_score = score.get("fullTime", {}).get("away")
        status = item.get("status", "")

        if date_iso:
            try:
                dt = datetime.fromisoformat(date_iso.replace("Z", "+00:00")).astimezone(timezone.utc)
                date_iso = dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                pass

        rows.append(
            {
                "id": match_id,
                "date": date_iso,
                "matchday": matchday,
                "home_team": home_team,
                "away_team": away_team,
                "home_score": float(home_score) if home_score is not None else "",
                "away_score": float(away_score) if away_score is not None else "",
                "status": status,
            }
        )
    rows.sort(key=lambda r: (r.get("date") or "", r.get("id") or 0))
    return rows


def write_matches_csv(rows: List[Dict[str, Any]], season_label: str) -> Path:
    """
    Escribe data/cleaned/champions/matches_YYYY_YY.csv con nuestro formato estándar.
    """
    out_dir = Path("data/cleaned/champions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"matches_{season_label.replace('-', '_')}.csv"
    fieldnames = ["id", "date", "matchday", "home_team", "away_team", "home_score", "away_score", "status"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"✅ Champions matches actualizados: {out_path} ({len(rows)} filas)")
    return out_path


def write_teams_from_matches(rows: List[Dict[str, Any]], season_label: str) -> Path:
    """
    Extrae equipos únicos de los rows (home/away) y genera teams_YYYY_YY.csv en champions/.
    """
    from config.team_names import normalize_team_name  # reutilizamos mapeo canónico

    teams = set()
    for r in rows:
        h = r.get("home_team")
        a = r.get("away_team")
        if h:
            teams.add(normalize_team_name(str(h)))
        if a:
            teams.add(normalize_team_name(str(a)))

    out_dir = Path("data/cleaned/champions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"teams_{season_label.replace('-', '_')}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "team_name", "season"])
        w.writeheader()
        for idx, name in enumerate(sorted(teams), 1):
            w.writerow({"id": idx, "team_name": name, "season": season_label})
    print(f"✅ Champions teams actualizados: {out_path} ({len(teams)} equipos)")
    return out_path


def _last_update_path() -> Path:
    return Path("data/cleaned/champions_last_update.json")


def _should_update(max_age_hours: int = 6) -> bool:
    p = _last_update_path()
    if not p.exists():
        return True
    try:
        with open(p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        ts = meta.get("timestamp")
        if not ts:
            return True
        dt = datetime.fromisoformat(ts)
        return datetime.now(timezone.utc) - dt > timedelta(hours=max_age_hours)
    except Exception:
        return True


def _write_last_update_meta(season_start_year: int, season_label: str) -> None:
    p = _last_update_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "season_start_year": season_start_year,
        "season_label": season_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def auto_update_champions_current_season(
    *,
    season_start_year: int = 2025,
    season_label: str = "2025-26",
    force: bool = False,
) -> None:
    """
    Punto de entrada para usar desde main.py:

        from prediction.champions_auto_update import auto_update_champions_current_season
        auto_update_champions_current_season()
    """
    if not _can_call_api():
        # No romper ejecución si falta la key; solo avisar.
        print("⚠️  API-FOOTBALL key no configurada; se omite auto-update de Champions.")
        return

    if not force and not _should_update():
        print("ℹ️  Champions ya está actualizado recientemente; no se vuelve a llamar a la API.")
        return

    try:
        fixtures = fetch_champions_fixtures_api(season_start_year)
        if not fixtures:
            print("⚠️  No se recibieron fixtures de Champions desde la API.")
            return
        rows = fixtures_to_matches_rows(fixtures)
        write_matches_csv(rows, season_label)
        write_teams_from_matches(rows, season_label)
        _write_last_update_meta(season_start_year, season_label)
    except Exception as e:
        print(f"❌ Error actualizando datos de Champions desde la API: {e}")


if __name__ == "__main__":
    auto_update_champions_current_season(force=True)

