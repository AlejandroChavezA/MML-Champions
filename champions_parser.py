#!/usr/bin/env python3
"""
Champions League Data Parser - Simplificado
========================================
Parser simplificado para datos reales de Champions League
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import defaultdict

class ChampionsLeagueParser:
    """Parser simplificado para archivos de datos de Champions League"""
    
    def __init__(self, data_dir: str = "data/dataReal"):
        self.data_dir = Path(data_dir)
        self.competitions = {
            'cl': 'Champions League',
            'el': 'Europa League', 
            'conf': 'Conference League'
        }
    
    def get_available_seasons(self) -> Dict[str, List[str]]:
        """Obtener temporadas disponibles por competición"""
        seasons = defaultdict(list)
        
        for season_dir in self.data_dir.iterdir():
            if season_dir.is_dir() and re.match(r'\d{4}-\d{2}', season_dir.name):
                season_name = season_dir.name
                
                for file_path in season_dir.glob('*.txt'):
                    comp_code = file_path.stem
                    if comp_code in self.competitions:
                        seasons[comp_code].append(season_name)
        
        # Ordenar temporadas
        for comp in seasons:
            seasons[comp].sort(reverse=True)
        
        return dict(seasons)
    
    def parse_file_simple(self, comp_code: str, season: str) -> List[Dict]:
        """Parsear un archivo específico de forma simplificada"""
        file_path = self.data_dir / season / f"{comp_code}.txt"
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        matches = []
        lines = content.split('\n')
        current_date = None
        current_stage = 'League'
        
        for line in lines:
            line = line.strip()
            
            # Ignorar líneas vacías y comentarios
            if not line or line.startswith('#') or line.startswith('='):
                continue
            
            # Detectar secciones
            if line.startswith('»'):
                if 'League' in line:
                    current_stage = 'League'
                elif 'Group' in line:
                    current_stage = 'Group Stage'
                elif 'Quarter' in line:
                    current_stage = 'Quarter-Finals'
                elif 'Semi' in line:
                    current_stage = 'Semi-Finals'
                elif 'Final' in line:
                    current_stage = 'Final'
                continue
            
            # Detectar fechas
            if re.match(r'^[A-Z][a-z]{2}/\d{1,2}/\d{4}', line):
                current_date = line.strip()
                continue
            
            # Detectar partidos
            if re.match(r'^\d{2}\.\d{2}', line):
                match_data = self._parse_match_line_simple(line, current_date, current_stage)
                if match_data:
                    matches.append(match_data)
        
        return matches
    
    def _parse_match_line_simple(self, line: str, date: Optional[str], stage: str) -> Optional[Dict]:
        """Parsear línea de partido de forma simplificada"""
        try:
            # Partido no jugado
            if 'N.N.' in line or 'v N.N.' in line:
                return None
            
            # Extraer tiempo
            time_match = re.match(r'^(\d{2}\.\d{2})\s+', line)
            if not time_match:
                return None
            
            time = time_match.group(1)
            rest_line = line[time_match.end():]
            
            # Dividir por 'v'
            if ' v ' not in rest_line:
                return None
            
            parts = rest_line.split(' v ')
            if len(parts) != 2:
                return None
            
            # Extraer equipos y resultado
            home_part = parts[0].strip()
            away_part = parts[1].strip()
            
            # Extraer nombre de equipo y país del home
            home_match = re.match(r'([A-Za-z\s\-\.\']+)\s+\([A-Z]{3}\)', home_part)
            if not home_match:
                return None
            
            home_team = home_match.group(1).strip()
            
            # Extraer visitante y resultado
            away_score_match = re.search(r'([A-Za-z\s\-\.\']+)\s+\([A-Z]{3}\)\s*(\d+-\d+)', away_part)
            if not away_score_match:
                return None
            
            away_team = away_score_match.group(1).strip()
            score_str = away_score_match.group(2)
            
            # Parsear resultado
            score_parts = score_str.split('-')
            if len(score_parts) != 2:
                return None
            
            home_score = int(score_parts[0])
            away_score = int(score_parts[1])
            
            return {
                'date': date or '',
                'time': time,
                'home_team': home_team,
                'away_team': away_team,
                'stage': stage,
                'home_score': home_score,
                'away_score': away_score,
                'status': 'completed'
            }
            
        except Exception as e:
            # Si hay error en el parseo, ignorar esta línea
            return None
    
    def load_all_champions_data(self) -> List[Dict]:
        """Cargar todos los datos de Champions League disponibles"""
        print("📊 Cargando datos de Champions League...")
        
        all_matches = []
        seasons = self.get_available_seasons()
        
        # Cargar solo Champions League
        for season in seasons.get('cl', []):
            print(f"  📋 Cargando Champions League {season}...")
            matches = self.parse_file_simple('cl', season)
            all_matches.extend(matches)
        
        print(f"✅ Champions League: {len(all_matches)} partidos cargados")
        return all_matches
    
    def export_to_json(self, matches: List[Dict], output_file: str = "champions_data.json"):
        """Exportar datos a JSON"""
        export_data = {
            'metadata': {
                'competition': 'UEFA Champions League',
                'total_matches': len(matches),
                'data_source': 'dataReal directory',
                'export_date': '2025-02-11'
            },
            'matches': matches
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Datos exportados a: {output_file}")
        return export_data

def main():
    """Función de prueba del parser"""
    parser = ChampionsLeagueParser()
    
    # Cargar datos
    matches = parser.load_all_champions_data()
    
    if matches:
        # Exportar a JSON
        parser.export_to_json(matches)
        
        # Mostrar algunos ejemplos
        print(f"\n📋 Ejemplos de partidos:")
        for i, match in enumerate(matches[:5], 1):
            print(f"   {i}. {match.get('date', 'N/A')}: {match.get('home_team', 'N/A')} {match.get('home_score', 0)}-{match.get('away_score', 0)} {match.get('away_team', 'N/A')}")
    
    else:
        print("❌ No se encontraron partidos")

if __name__ == "__main__":
    main()