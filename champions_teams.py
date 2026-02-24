#!/usr/bin/env python3
"""
Champions League Teams Manager - Final Version
============================================
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

class ChampionsTeamsManager:
    def __init__(self, data_dir="data/dataReal", output_dir="data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.teams_by_season = {}
    
    def extract_teams(self):
        """Extraer equipos de cada temporada"""
        print("Extrayendo equipos...")
        
        for season_dir in sorted(self.data_dir.iterdir(), reverse=True):
            if not (season_dir.is_dir() and re.match(r'\d{4}-\d{2}', season_dir.name)):
                continue
            
            season = season_dir.name
            cl_file = season_dir / "cl.txt"
            if not cl_file.exists():
                continue
            
            teams = self._parse_teams(cl_file)
            self.teams_by_season[season] = teams
            print(f"  {season}: {len(teams)} equipos")
        
        return self.teams_by_season
    
    def _parse_teams(self, filepath):
        """Parsear equipos de archivo"""
        teams = set()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip headers and empty lines
                if not line or line.startswith('=') or line.startswith('#') or line.startswith('»'):
                    continue
                
                # Remove time prefix like "18.45  " or "21.00  "
                line = re.sub(r'^\d{2}\.\d{2}\s+', '', line)
                
                # Skip lines without " v "
                if ' v ' not in line:
                    continue
                
                # Split by " v "
                parts = line.split(' v ')
                if len(parts) != 2:
                    continue
                
                # Extract home team (before v)
                home = parts[0].strip()
                home_team = self._get_team_name(home)
                if home_team:
                    teams.add(home_team)
                
                # Extract away team (after v, before score)
                away = parts[1].strip()
                # Remove score if present
                away = re.sub(r'\s+\d+-\d+.*$', '', away)
                away_team = self._get_team_name(away)
                if away_team:
                    teams.add(away_team)
        
        return teams
    
    def _get_team_name(self, text):
        """Extraer nombre de equipo"""
        if not text:
            return None
        
        # Remove "v " prefix if present
        text = re.sub(r'^v\s+', '', text)
        
        # Look for pattern: "Team Name (COUNTRY)"
        # At the END of the string (before any score)
        match = re.search(r'^(.+?)\s*\(([A-Z]{3})\)\s*$', text)
        if match:
            name = match.group(1).strip()
            
            # Filter valid names
            if len(name) >= 3 and not name.isdigit():
                # Skip if it looks like a broken fragment
                bad_starts = ['v', 'dam', 'ben', 'nch', 'tic']
                if any(name.lower().startswith(b) for b in bad_starts):
                    return None
                return name
        
        return None
    
    def get_current_teams(self):
        """Get current season teams"""
        if self.teams_by_season:
            return sorted(self.teams_by_season[list(self.teams_by_season.keys())[0]])
        return []
    
    def export_to_csv(self):
        """Exportar a CSV"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nExportando...")
        
        for season, teams in sorted(self.teams_by_season.items()):
            filename = f"teams_{season.replace('-', '_')}.csv"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['team_name', 'season'])
                for team in sorted(teams):
                    writer.writerow([team, season])
            
            print(f"  {filename}")
        
        print("OK!")

def main():
    print("=" * 50)
    print("TEAMS EXTRACTOR")
    print("=" * 50)
    
    mgr = ChampionsTeamsManager()
    mgr.extract_teams()
    
    teams = mgr.get_current_teams()
    print(f"\nTemporada actual: {list(mgr.teams_by_season.keys())[0]}")
    print(f"Equipos: {len(teams)}")
    
    for i, t in enumerate(teams, 1):
        print(f"  {i:2d}. {t}")
    
    mgr.export_to_csv()

if __name__ == "__main__":
    main()