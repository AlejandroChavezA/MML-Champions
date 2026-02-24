#!/usr/bin/env python3
"""
Champions League Data Analyzer
===========================
Analizador de datos para generar estadísticas y predicciones
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

class ChampionsLeagueAnalyzer:
    """Analizador de datos de Champions League"""
    
    def __init__(self, data_file: str = "champions_data.json"):
        self.data_file = Path(data_file)
        self.matches = []
        self.team_stats = defaultdict(dict)
        self.teams = set()
    
    def load_data(self, data: Union[Dict, None] = None) -> bool:
        """Cargar datos desde archivo o diccionario"""
        if data is not None:
            all_matches = data.get('champions_league', [])
        else:
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                all_matches = loaded_data.get('champions_league', [])
            except FileNotFoundError:
                print(f"❌ No se encontró archivo: {self.data_file}")
                return False
        
        # Filtrar solo partidos completados
        self.matches = [m for m in all_matches if m.get('home_score') is not None]
        self.teams = self._extract_teams(self.matches)
        
        print(f"✅ Datos cargados:")
        print(f"   📊 Total partidos: {len(self.matches)}")
        print(f"   🏟️ Equipos únicos: {len(self.teams)}")
        
        return True
    
    def _extract_teams(self, matches: List[Dict]) -> set:
        """Extraer equipos únicos de los partidos"""
        teams = set()
        for match in matches:
            if match.get('home_team'):
                teams.add(match['home_team'])
            if match.get('away_team'):
                teams.add(match['away_team'])
        return teams
    
    def calculate_team_statistics(self) -> Dict:
        """Calcular estadísticas completas para todos los equipos"""
        print("🔧 Calculando estadísticas de equipos...")
        
        # Inicializar estadísticas
        for team in self.teams:
            self.team_stats[team] = {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0,
                'goals_for': 0, 'goals_against': 0, 'points': 0,
                'home_matches': 0, 'away_matches': 0,
                'home_wins': 0, 'away_wins': 0,
                'clean_sheets': 0, 'failed_to_score': 0,
                'form': [], 'recent_form': []
            }
        
        # Calcular estadísticas partido por partido
        for match in self.matches:
            home_team = match.get('home_team')
            away_team = match.get('away_team')
            home_score = match.get('home_score', 0)
            away_score = match.get('away_score', 0)
            
            if not home_team or not away_team:
                continue
            
            # Actualizar partidos jugados
            self.team_stats[home_team]['matches'] += 1
            self.team_stats[away_team]['matches'] += 1
            self.team_stats[home_team]['home_matches'] += 1
            self.team_stats[away_team]['away_matches'] += 1
            
            # Actualizar goles
            self.team_stats[home_team]['goals_for'] += home_score
            self.team_stats[home_team]['goals_against'] += away_score
            self.team_stats[away_team]['goals_for'] += away_score
            self.team_stats[away_team]['goals_against'] += home_score
            
            # Clean sheets y partidos sin marcar
            if home_score == 0:
                self.team_stats[away_team]['clean_sheets'] += 1
                self.team_stats[home_team]['failed_to_score'] += 1
            
            if away_score == 0:
                self.team_stats[home_team]['clean_sheets'] += 1
                self.team_stats[away_team]['failed_to_score'] += 1
            
            # Determinar resultado y actualizar puntos
            result_home = self._get_result(home_score, away_score)
            result_away = self._get_result(away_score, home_score)
            
            # Actualizar estadísticas de resultados
            if result_home == 'W':
                self.team_stats[home_team]['wins'] += 1
            elif result_home == 'D':
                self.team_stats[home_team]['draws'] += 1
            else:
                self.team_stats[home_team]['losses'] += 1
            
            if result_away == 'W':
                self.team_stats[away_team]['wins'] += 1
            elif result_away == 'D':
                self.team_stats[away_team]['draws'] += 1
            else:
                self.team_stats[away_team]['losses'] += 1
            
            # Actualizar victorias local/visitante
            self.team_stats[home_team]['home_wins'] += 1 if result_home == 'W' else 0
            self.team_stats[away_team]['away_wins'] += 1 if result_away == 'W' else 0
            
            # Actualizar puntos
            points_home = 3 if result_home == 'W' else 1 if result_home == 'D' else 0
            points_away = 3 if result_away == 'W' else 1 if result_away == 'D' else 0
            
            self.team_stats[home_team]['points'] += points_home
            self.team_stats[away_team]['points'] += points_away
            
            # Agregar a forma (para análisis de forma reciente)
            self.team_stats[home_team]['form'].append(result_home)
            self.team_stats[away_team]['form'].append(result_away)
        
        # Calcular forma reciente (últimos 5 partidos)
        for team in self.team_stats:
            form = self.team_stats[team]['form']
            self.team_stats[team]['recent_form'] = form[-5:] if len(form) >= 5 else form
            
            # Calcular promedios
            matches = max(self.team_stats[team]['matches'], 1)
            self.team_stats[team]['points_per_game'] = self.team_stats[team]['points'] / matches
            self.team_stats[team]['goals_per_game'] = self.team_stats[team]['goals_for'] / matches
            self.team_stats[team]['goals_conceded_per_game'] = self.team_stats[team]['goals_against'] / matches
            self.team_stats[team]['goal_difference'] = self.team_stats[team]['goals_for'] - self.team_stats[team]['goals_against']
            self.team_stats[team]['win_rate'] = self.team_stats[team]['wins'] / matches
            self.team_stats[team]['draw_rate'] = self.team_stats[team]['draws'] / matches
            self.team_stats[team]['loss_rate'] = self.team_stats[team]['losses'] / matches
        
        print(f"✅ Estadísticas calculadas para {len(self.team_stats)} equipos")
        return self.team_stats
    
    def _get_result(self, home_score: int, away_score: int) -> str:
        """Determinar resultado (W/D/L)"""
        if home_score > away_score:
            return 'W'
        elif home_score < away_score:
            return 'L'
        else:
            return 'D'
    
    def get_team_rankings(self, limit: int = 20) -> List[Tuple]:
        """Obtener rankings de equipos"""
        rankings = []
        
        for team, stats in self.team_stats.items():
            rankings.append((
                team,
                stats['points'],
                stats['matches'],
                stats['wins'],
                stats['draws'],
                stats['losses'],
                stats['goal_difference'],
                stats['goals_for'],
                stats['goals_against'],
                stats['points_per_game']
            ))
        
        # Ordenar por puntos, luego por diferencia de goles
        rankings.sort(key=lambda x: (x[1], x[5]), reverse=True)
        
        return rankings[:limit]
    
    def get_head_to_head(self, team1: str, team2: str, limit: int = 10) -> List[Dict]:
        """Obtener historial de enfrentamientos directos"""
        h2h = []
        
        for match in self.matches:
            home = match.get('home_team')
            away = match.get('away_team')
            
            if (home == team1 and away == team2) or (home == team2 and away == team1):
                h2h.append(match.copy())
        
        # Ordenar por fecha (más recientes primero)
        h2h.sort(key=lambda x: x.get('date', ''), reverse=True)
        
        return h2h[:limit]
    
    def get_team_form(self, team: str, last_n: int = 5) -> List[str]:
        """Obtener forma reciente de un equipo"""
        if team not in self.team_stats:
            return []
        
        return self.team_stats[team]['recent_form'][-last_n:]
    
    def analyze_trends(self) -> Dict:
        """Analizar tendencias generales de los datos"""
        trends = {}
        
        # Distribución de resultados
        total_matches = len(self.matches)
        home_wins = sum(1 for m in self.matches if m.get('home_score', 0) > m.get('away_score', 0))
        away_wins = sum(1 for m in self.matches if m.get('away_score', 0) > m.get('home_score', 0))
        draws = sum(1 for m in self.matches if m.get('home_score', 0) == m.get('away_score', 0))
        
        trends['result_distribution'] = {
            'home_wins': {'count': home_wins, 'percentage': home_wins / total_matches * 100},
            'away_wins': {'count': away_wins, 'percentage': away_wins / total_matches * 100},
            'draws': {'count': draws, 'percentage': draws / total_matches * 100}
        }
        
        # Estadísticas de goles
        total_goals = sum(m.get('home_score', 0) + m.get('away_score', 0) for m in self.matches)
        trends['goal_statistics'] = {
            'total_goals': total_goals,
            'average_goals_per_match': total_goals / total_matches,
            'home_team_average': sum(m.get('home_score', 0) for m in self.matches) / total_matches,
            'away_team_average': sum(m.get('away_score', 0) for m in self.matches) / total_matches
        }
        
        # Rango de fechas
        dates = [m.get('date') for m in self.matches if m.get('date')]
        if dates:
            trends['date_range'] = {
                'earliest': min(dates),
                'latest': max(dates)
            }
        
        # Equipos con mejor/worst rendimiento
        if self.team_stats:
            best_team = max(self.team_stats.items(), key=lambda x: x[1]['points_per_game'])
            worst_team = min(self.team_stats.items(), key=lambda x: x[1]['points_per_game'])
            
            trends['team_extremes'] = {
                'best_team': {'name': best_team[0], 'ppg': best_team[1]['points_per_game']},
                'worst_team': {'name': worst_team[0], 'ppg': worst_team[1]['points_per_game']}
            }
        
        return trends
    
    def export_data(self, output_file: str = "champions_analysis.json"):
        """Exportar datos analizados a JSON"""
        output_data = {
            'metadata': {
                'total_matches': len(self.matches),
                'total_teams': len(self.teams),
                'analysis_date': datetime.now().isoformat(),
                'data_source': 'dataReal directory'
            },
            'team_statistics': dict(self.team_stats),
            'trends': self.analyze_trends()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Datos exportados a: {output_file}")
        return output_data