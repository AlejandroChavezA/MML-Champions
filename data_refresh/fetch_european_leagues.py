#!/usr/bin/env python3
"""Data fetch utility for Europa League and Conference League using TheSportsDB API."""

import json
import requests
from datetime import datetime
from pathlib import Path

API_KEY = "123"
BASE_URL = "https://www.thesportsdb.com/api/v1/json/123"

LEAGUES = {
    "4481": "UEFA Europa League",
    "5071": "UEFA Conference League"
}

def fetch_events(league_id, next_or_past="next"):
    """Fetch events (past or next) for a specific league."""
    endpoint = f"events{next_or_past}league.php"
    url = f"{BASE_URL}/{endpoint}?id={league_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("events", []) or []
    except Exception as e:
        print(f"Error fetching {next_or_past} events for league {league_id}: {e}")
        return []

def transform_match(event, league_name):
    """Transform TheSportsDB event to our format."""
    date_str = event.get("dateEvent", "")
    time_str = event.get("strTime", "00:00:00")
    
    if time_str and len(time_str.split(":")) == 2:
        time_str = time_str + ":00"
    
    status = event.get("strStatus", "Unknown")
    if status == "FT":
        status = "completed"
    elif status == "Not Started":
        status = "scheduled"
    elif status == "HT":
        status = "halftime"
    elif status in ["1H", "2H"]:
        status = "in_progress"
    
    round_num = event.get("intRound", "")
    try:
        round_num = int(round_num) if round_num else None
    except:
        round_num = None
    
    return {
        "date": date_str,
        "time": time_str,
        "home_team": event.get("strHomeTeam", ""),
        "away_team": event.get("strAwayTeam", ""),
        "stage": f"Round {round_num}" if round_num else "Unknown",
        "jornada": round_num,
        "home_score": int(event.get("intHomeScore", 0)) if event.get("intHomeScore") else None,
        "away_score": int(event.get("intAwayScore", 0)) if event.get("intAwayScore") else None,
        "status": status,
        "league": league_name,
        "season": event.get("strSeason", ""),
        "venue": event.get("strVenue", ""),
        "home_team_badge": event.get("strHomeTeamBadge", ""),
        "away_team_badge": event.get("strAwayTeamBadge", "")
    }

def fetch_all_league_data(league_id, league_name):
    """Fetch all data (past and next) for a league."""
    print(f"Fetching {league_name}...")
    
    past_events = fetch_events(league_id, "past")
    print(f"  - Past events: {len(past_events)}")
    
    next_events = fetch_events(league_id, "next")
    print(f"  - Next events: {len(next_events)}")
    
    all_events = past_events + next_events
    
    matches = []
    for event in all_events:
        match = transform_match(event, league_name)
        matches.append(match)
    
    matches.sort(key=lambda x: (x["date"] or "", x["time"] or ""))
    
    return matches

def group_by_jornada(matches):
    """Group matches by jornada/round."""
    jornadas = {}
    for match in matches:
        jornada = match.get("jornada")
        if jornada is None:
            jornada = "Unknown"
        
        if jornada not in jornadas:
            jornadas[jornada] = []
        jornadas[jornada].append(match)
    
    return jornadas

def main():
    print("=" * 60)
    print("Fetching Europa League and Conference League data")
    print("=" * 60)
    
    all_data = {
        "metadata": {
            "competition": "UEFA Europa League & Conference League",
            "data_source": "TheSportsDB API",
            "export_date": datetime.now().strftime("%Y-%m-%d"),
            "api_version": "v1"
        },
        "matches": [],
        "by_league": {}
    }
    
    for league_id, league_name in LEAGUES.items():
        matches = fetch_all_league_data(league_id, league_name)
        
        all_data["by_league"][league_name] = {
            "league_id": league_id,
            "total_matches": len(matches),
            "matches": matches,
            "jornadas": group_by_jornada(matches)
        }
        
        all_data["matches"].extend(matches)
    
    all_data["metadata"]["total_matches"] = len(all_data["matches"])
    
    output_file = Path(__file__).parent.parent / "european_leagues_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Data saved to: {output_file}")
    print(f"Total matches: {all_data['metadata']['total_matches']}")
    print("=" * 60)
    
    print("\nSummary by league:")
    for league_name, data in all_data["by_league"].items():
        print(f"  {league_name}: {data['total_matches']} matches, {len(data['jornadas'])} jornadas")
    
    jornada_sample = {}
    for league_name, data in all_data["by_league"].items():
        if data["jornadas"]:
            first_jornada = sorted(data["jornadas"].keys())[0]
            jornada_sample[league_name] = first_jornada
    
    print("\nSample - First jornada per league:")
    for league, jornada in jornada_sample.items():
        print(f"  {league}: Jornada {jornada}")

if __name__ == "__main__":
    main()
