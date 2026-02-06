import requests
import pandas as pd
import json
import time
import os
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Optional

class ChampionsLeagueDataCollector:
    """Colector de datos especializado para UEFA Champions League"""
    
    def __init__(self):
        # API-Football (mejor para Champions League)
        self.api_football_url = "https://v3.football.api-sports.io"
        self.api_football_headers = {
            "x-apisports-key": "TU_API_FOOTBALL_KEY"  # Necesitarás tu API key
        }
        
        # Football-Data.org (backup)
        self.football_data_url = "https://api.football-data.org/v4"
        self.football_data_headers = {
            'X-Auth-Token': 'fd9ecc768e3644dfa9b30e9536031700'
        }
        
        self.data_dir = "../data"
        self.cache_dir = "../data/cache"
        
        # Coeficientes UEFA y presupuestos (desde config)
        self.uefa_coefficients = {
            'Real Madrid': 132.638, 'Bayern Munich': 128.928, 'Manchester City': 123.478,
            'Liverpool': 117.000, 'Chelsea': 106.841, 'Barcelona': 98.478,
            'Paris Saint Germain': 92.641, 'Juventus': 89.641, 'Manchester United': 87.641,
            'Arsenal': 84.641, 'Atlético Madrid': 82.641, 'Tottenham': 80.641,
            'Borussia Dortmund': 78.641, 'Inter': 76.641, 'AC Milan': 74.641,
            'Napoli': 72.641, 'RB Leipzig': 70.641, 'Porto': 68.641, 'Ajax': 66.641
        }
        
        self.club_budgets = {
            'Real Madrid': 1.4, 'Manchester City': 1.25, 'Paris Saint Germain': 1.1,
            'Bayern Munich': 1.05, 'Barcelona': 0.95, 'Liverpool': 0.85,
            'Chelsea': 0.8, 'Juventus': 0.75, 'Manchester United': 0.7, 'Arsenal': 0.65
        }
        
    def create_data_directory(self):
        """Crear estructura de directorios para Champions League"""
        directories = [
            self.data_dir,
            f"{self.data_dir}/raw",
            f"{self.data_dir}/cleaned", 
            f"{self.data_dir}/processed",
            self.cache_dir
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"📁 Creado directorio: {directory}")
    
    async def collect_champions_data_async(self, seasons: List[int] = [2023, 2024]):
        """Colectar datos de Champions League de forma asíncrona"""
        print("🏆 INICIANDO COLECTA ASÍNCRONA CHAMPIONS LEAGUE")
        print("=" * 60)
        
        self.create_data_directory()
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for season in seasons:
                task = self._collect_season_data(session, season)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                season = seasons[i]
                if isinstance(result, Exception):
                    print(f"❌ Error temporada {season}: {result}")
                else:
                    print(f"✅ Temporada {season} completada")
        
        print("\n🎉 COLECTA CHAMPIONS LEAGUE COMPLETADA")
    
    async def _collect_season_data(self, session: aiohttp.ClientSession, season: int):
        """Colectar datos de una temporada específica"""
        try:
            # 1. Obtener equipos
            teams = await self._get_champions_teams_async(session, season)
            
            # 2. Obtener partidos
            matches = await self._get_champions_matches_async(session, season)
            
            # 3. Enriquecer datos
            enriched_matches = await self._enrich_matches_data(matches, teams)
            
            # 4. Guardar datos
            await self._save_season_data(season, teams, enriched_matches)
            
            return True
            
        except Exception as e:
            print(f"Error en temporada {season}: {e}")
            return False
    
    async def _get_champions_teams_async(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener equipos de Champions League de forma asíncrona"""
        print(f"📋 Obteniendo equipos temporada {season}...")
        
        # Intentar con API-Football primero
        teams = await self._fetch_teams_api_football(session, season)
        
        if not teams:
            # Fallback a Football-Data.org
            teams = await self._fetch_teams_football_data(session, season)
        
        if not teams:
            # Fallback a datos simulados
            teams = self._generate_mock_teams()
        
        print(f"✅ {len(teams)} equipos obtenidos")
        return teams
    
    async def _fetch_teams_api_football(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener equipos desde API-Football"""
        try:
            url = f"{self.api_football_url}/teams"
            params = {
                "league": 2,  # Champions League ID
                "season": season
            }
            
            async with session.get(url, headers=self.api_football_headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    print(f"⚠️ API-Football error: {response.status}")
                    return []
        except Exception as e:
            print(f"⚠️ Error API-Football: {e}")
            return []
    
    async def _fetch_teams_football_data(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener equipos desde Football-Data.org"""
        try:
            url = f"{self.football_data_url}/competitions/CL/teams"
            params = {"season": season}
            
            async with session.get(url, headers=self.football_data_headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('teams', [])
                else:
                    print(f"⚠️ Football-Data error: {response.status}")
                    return []
        except Exception as e:
            print(f"⚠️ Error Football-Data: {e}")
            return []
    
    def _generate_mock_teams(self) -> List[Dict]:
        """Generar equipos simulados para Champions League"""
        teams_data = [
            {"team": {"id": 541, "name": "Real Madrid"}, "venue": "Santiago Bernabéu"},
            {"team": {"id": 529, "name": "Arsenal"}, "venue": "Emirates Stadium"},
            {"team": {"id": 542, "name": "Bayern Munich"}, "venue": "Allianz Arena"},
            {"team": {"id": 545, "name": "Manchester City"}, "venue": "Etihad Stadium"},
            {"team": {"id": 546, "name": "Paris Saint Germain"}, "venue": "Parc des Princes"},
            {"team": {"id": 548, "name": "Barcelona"}, "venue": "Camp Nou"},
            {"team": {"id": 550, "name": "Liverpool"}, "venue": "Anfield"},
            {"team": {"id": 552, "name": "Juventus"}, "venue": "Allianz Stadium"},
            {"team": {"id": 557, "name": "Inter"}, "venue": "San Siro"},
            {"team": {"id": 558, "name": "AC Milan"}, "venue": "San Siro"},
            {"team": {"id": 563, "name": "Atlético Madrid"}, "venue": "Metropolitano"},
            {"team": {"id": 566, "name": "Tottenham"}, "venue": "Tottenham Stadium"},
            {"team": {"id": 574, "name": "Borussia Dortmund"}, "venue": "Signal Iduna Park"},
            {"team": {"id": 583, "name": "Napoli"}, "venue": "Stadio San Paolo"},
            {"team": {"id": 586, "name": "RB Leipzig"}, "venue": "Red Bull Arena"},
            {"team": {"id": 603, "name": "Porto"}, "venue": "Estádio do Dragão"},
            {"team": {"id": 610, "name": "Ajax"}, "venue": "Johan Cruyff Arena"},
            {"team": {"id": 628, "name": "Chelsea"}, "venue": "Stamford Bridge"},
            {"team": {"id": 657, "name": "Manchester United"}, "venue": "Old Trafford"}
        ]
        return teams_data
    
    async def _get_champions_matches_async(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener partidos de Champions League de forma asíncrona"""
        print(f"⚽ Obteniendo partidos temporada {season}...")
        
        # Intentar con API-Football primero
        matches = await self._fetch_matches_api_football(session, season)
        
        if not matches:
            # Fallback a Football-Data.org
            matches = await self._fetch_matches_football_data(session, season)
        
        if not matches:
            # Fallback a datos simulados
            matches = self._generate_mock_matches(season)
        
        print(f"✅ {len(matches)} partidos obtenidos")
        return matches
    
    async def _fetch_matches_api_football(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener partidos desde API-Football"""
        try:
            url = f"{self.api_football_url}/fixtures"
            params = {
                "league": 2,  # Champions League ID
                "season": season
            }
            
            async with session.get(url, headers=self.api_football_headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('response', [])
                else:
                    print(f"⚠️ API-Football matches error: {response.status}")
                    return []
        except Exception as e:
            print(f"⚠️ Error API-Football matches: {e}")
            return []
    
    async def _fetch_matches_football_data(self, session: aiohttp.ClientSession, season: int) -> List[Dict]:
        """Obtener partidos desde Football-Data.org"""
        try:
            url = f"{self.football_data_url}/competitions/CL/matches"
            params = {"season": season}
            
            async with session.get(url, headers=self.football_data_headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('matches', [])
                else:
                    print(f"⚠️ Football-Data matches error: {response.status}")
                    return []
        except Exception as e:
            print(f"⚠️ Error Football-Data matches: {e}")
            return []
    
    def _generate_mock_matches(self, season: int) -> List[Dict]:
        """Generar partidos simulados para Champions League"""
        teams = list(self.uefa_coefficients.keys())
        matches = []
        
        # Simular fase de grupos (6 fechas)
        groups = [teams[i:i+4] for i in range(0, len(teams), 4)]
        
        for group_idx, group in enumerate(groups):
            for matchday in range(1, 7):  # 6 fechas de grupos
                for i in range(0, len(group), 2):
                    if i + 1 < len(group):
                        home_team = group[i]
                        away_team = group[i + 1]
                        
                        # Alternar localía
                        if matchday % 2 == 0:
                            home_team, away_team = away_team, home_team
                        
                        match_date = datetime(2024, 9, 1) + timedelta(weeks=matchday)
                        
                        match = {
                            "fixture": {
                                "id": f"{season}_group_{group_idx}_{matchday}_{i}",
                                "date": match_date.isoformat(),
                                "status": {"long": "FINISHED" if match_date < datetime.now() else "SCHEDULED"}
                            },
                            "teams": {
                                "home": {"name": home_team},
                                "away": {"name": away_team}
                            },
                            "goals": {
                                "home": None if match_date > datetime.now() else self._simulate_score(home_team, away_team)[0],
                                "away": None if match_date > datetime.now() else self._simulate_score(home_team, away_team)[1]
                            },
                            "league": {
                                "round": f"Group Stage - Matchday {matchday}",
                                "name": "UEFA Champions League"
                            }
                        }
                        matches.append(match)
        
        # Simular eliminatorias
        knockout_rounds = ["Round of 16", "Quarter-finals", "Semi-finals", "Final"]
        for round_name in knockout_rounds:
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    home_team = teams[i]
                    away_team = teams[i + 1]
                    
                    match_date = datetime(2024, 2, 1) + timedelta(weeks=len(knockout_rounds))
                    
                    match = {
                        "fixture": {
                            "id": f"{season}_{round_name}_{i}",
                            "date": match_date.isoformat(),
                            "status": {"long": "SCHEDULED"}
                        },
                        "teams": {
                            "home": {"name": home_team},
                            "away": {"name": away_team}
                        },
                        "goals": {"home": None, "away": None},
                        "league": {
                            "round": round_name,
                            "name": "UEFA Champions League"
                        }
                    }
                    matches.append(match)
        
        return matches
    
    def _simulate_score(self, home_team: str, away_team: str) -> Tuple[int, int]:
        """Simular resultado basado en coeficientes UEFA"""
        import random
        
        home_coeff = self.uefa_coefficients.get(home_team, 80)
        away_coeff = self.uefa_coefficients.get(away_team, 80)
        
        # Ventaja local
        home_advantage = 0.3
        
        # Calcular probabilidades
        home_strength = (home_coeff - away_coeff) / 100 + home_advantage
        
        # Simular goles
        if home_strength > 0.5:
            home_goals = random.choices([0, 1, 2, 3], weights=[20, 30, 35, 15])[0]
            away_goals = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
        else:
            home_goals = random.choices([0, 1, 2], weights=[40, 40, 20])[0]
            away_goals = random.choices([0, 1, 2, 3], weights=[20, 30, 35, 15])[0]
        
        return home_goals, away_goals
    
    async def _enrich_matches_data(self, matches: List[Dict], teams: List[Dict]) -> List[Dict]:
        """ Enriquecer partidos con características Champions League"""
        print("🔄 Enriqueciendo datos con características Champions...")
        
        enriched_matches = []
        
        for match in matches:
            # Extraer información básica
            if 'fixture' in match:  # API-Football format
                home_team = match['teams']['home']['name']
                away_team = match['teams']['away']['name']
                match_date = match['fixture']['date']
                status = match['fixture']['status']['long']
                round_info = match.get('league', {}).get('round', 'Unknown')
                home_score = match.get('goals', {}).get('home')
                away_score = match.get('goals', {}).get('away')
            else:  # Football-Data format
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                match_date = match['utcDate']
                status = match['status']
                round_info = match.get('stage', 'Unknown')
                home_score = match['score']['fullTime']['home'] if status == 'FINISHED' else None
                away_score = match['score']['fullTime']['away'] if status == 'FINISHED' else None
            
            # Características Champions League
            home_uefa_coeff = self.uefa_coefficients.get(home_team, 80)
            away_uefa_coeff = self.uefa_coefficients.get(away_team, 80)
            home_budget = self.club_budgets.get(home_team, 0.5)
            away_budget = self.club_budgets.get(away_team, 0.5)
            
            # Calcular distancia de viaje (simulada)
            travel_distance = self._calculate_travel_distance(home_team, away_team)
            
            # Determinar etapa y presión
            stage = self._determine_stage(round_info)
            pressure_factor = self._get_pressure_factor(stage)
            
            # Experiencia europea (títulos históricos)
            home_experience = self._get_champions_experience(home_team)
            away_experience = self._get_champions_experience(away_team)
            
            enriched_match = {
                # Datos básicos
                'match_id': match['fixture']['id'] if 'fixture' in match else match['id'],
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'status': status,
                'stage': stage,
                'round': round_info,
                
                # Características Champions League
                'home_uefa_coefficient': home_uefa_coeff,
                'away_uefa_coefficient': away_uefa_coeff,
                'uefa_coefficient_diff': home_uefa_coeff - away_uefa_coeff,
                'home_budget_billions': home_budget,
                'away_budget_billions': away_budget,
                'budget_diff': home_budget - away_budget,
                'travel_distance_km': travel_distance,
                'travel_fatigue_factor': self._get_travel_fatigue(travel_distance),
                'pressure_factor': pressure_factor,
                'home_champions_experience': home_experience,
                'away_champions_experience': away_experience,
                'experience_diff': home_experience - away_experience,
                
                # Formato específico
                'is_knockout': stage != 'GROUP_STAGE',
                'is_first_leg': self._is_first_leg(match, stage),
                'home_advantage_champions': 0.15 if stage == 'GROUP_STAGE' else 0.10
            }
            
            enriched_matches.append(enriched_match)
        
        print(f"✅ {len(enriched_matches)} partidos enriquecidos")
        return enriched_matches
    
    def _calculate_travel_distance(self, home_team: str, away_team: str) -> float:
        """Calcular distancia de viaje (simulada)"""
        # Distancias estimadas entre ciudades importantes
        city_distances = {
            'Madrid': {'Manchester': 1200, 'Munich': 1850, 'Paris': 1050, 'London': 1250, 'Milan': 1500},
            'Manchester': {'Madrid': 1200, 'Munich': 950, 'Paris': 600, 'London': 260, 'Milan': 1200},
            'Munich': {'Madrid': 1850, 'Manchester': 950, 'Paris': 750, 'London': 900, 'Milan': 550},
            'Paris': {'Madrid': 1050, 'Manchester': 600, 'Munich': 750, 'London': 340, 'Milan': 850},
            'London': {'Madrid': 1250, 'Manchester': 260, 'Munich': 900, 'Paris': 340, 'Milan': 1200},
            'Milan': {'Madrid': 1500, 'Manchester': 1200, 'Munich': 550, 'Paris': 850, 'London': 1200}
        }
        
        # Mapear equipos a ciudades
        team_cities = {
            'Real Madrid': 'Madrid', 'Atlético Madrid': 'Madrid',
            'Manchester City': 'Manchester', 'Manchester United': 'Manchester',
            'Bayern Munich': 'Munich',
            'Paris Saint Germain': 'Paris',
            'Arsenal': 'London', 'Chelsea': 'London', 'Tottenham': 'London',
            'Inter': 'Milan', 'AC Milan': 'Milan',
            'Liverpool': 'Manchester',  # Simplificación
            'Barcelona': 'Madrid',  # Simplificación
            'Juventus': 'Milan',  # Simplificación
            'Borussia Dortmund': 'Munich',  # Simplificación
            'Napoli': 'Milan',  # Simplificación
            'RB Leipzig': 'Munich',  # Simplificación
            'Porto': 'Madrid',  # Simplificación
            'Ajax': 'Amsterdam'  # No está en la tabla pero lo añadimos
        }
        
        home_city = team_cities.get(home_team, 'London')
        away_city = team_cities.get(away_team, 'London')
        
        # Obtener distancia
        if home_city in city_distances and away_city in city_distances[home_city]:
            return city_distances[home_city][away_city]
        else:
            return 500  # Distancia por defecto
    
    def _get_travel_fatigue(self, distance: float) -> float:
        """Calcular factor de fatiga por viaje"""
        if distance < 300:
            return 1.0  # Sin fatiga
        elif distance < 800:
            return 1.05  # Fatiga ligera
        elif distance < 1500:
            return 1.10  # Fatiga moderada
        else:
            return 1.15  # Fatiga alta
    
    def _determine_stage(self, round_info: str) -> str:
        """Determinar etapa de la competición"""
        round_info = round_info.lower()
        
        if 'group' in round_info:
            return 'GROUP_STAGE'
        elif 'round of 16' in round_info or '16' in round_info:
            return 'ROUND_OF_16'
        elif 'quarter' in round_info:
            return 'QUARTER_FINALS'
        elif 'semi' in round_info:
            return 'SEMI_FINALS'
        elif 'final' in round_info:
            return 'FINAL'
        else:
            return 'UNKNOWN'
    
    def _get_pressure_factor(self, stage: str) -> float:
        """Obtener factor de presión por etapa"""
        pressure_factors = {
            'GROUP_STAGE': 0.05,
            'ROUND_OF_16': 0.10,
            'QUARTER_FINALS': 0.15,
            'SEMI_FINALS': 0.20,
            'FINAL': 0.25,
            'UNKNOWN': 0.10
        }
        return pressure_factors.get(stage, 0.10)
    
    def _get_champions_experience(self, team: str) -> int:
        """Obtener experiencia en Champions League (títulos)"""
        champions_titles = {
            'Real Madrid': 14, 'AC Milan': 7, 'Liverpool': 6, 'Bayern Munich': 6,
            'Barcelona': 5, 'Manchester United': 3, 'Juventus': 2, 'Chelsea': 2,
            'Manchester City': 1, 'Inter': 1, 'Ajax': 4, 'Porto': 2,
            'Arsenal': 0, 'Tottenham': 0, 'Paris Saint Germain': 0, 'Atlético Madrid': 0,
            'Borussia Dortmund': 1, 'Napoli': 0, 'RB Leipzig': 0
        }
        return champions_titles.get(team, 0)
    
    def _is_first_leg(self, match: Dict, stage: str) -> bool:
        """Determinar si es partido de ida (simplificado)"""
        if stage == 'GROUP_STAGE':
            return False
        # Lógica simplificada - en implementación real se analizarían las fechas
        return True
    
    async def _save_season_data(self, season: int, teams: List[Dict], matches: List[Dict]):
        """Guardar datos de la temporada"""
        # Guardar equipos
        teams_df = pd.DataFrame(teams)
        teams_file = f"{self.data_dir}/raw/champions_teams_{season}.csv"
        teams_df.to_csv(teams_file, index=False)
        print(f"💾 Equipos guardados: {teams_file}")
        
        # Guardar partidos enriquecidos
        matches_df = pd.DataFrame(matches)
        matches_file = f"{self.data_dir}/raw/champions_matches_{season}.csv"
        matches_df.to_csv(matches_file, index=False)
        print(f"💾 Partidos guardados: {matches_file}")
        
        # Guardar cache para acceso rápido
        cache_file = f"{self.cache_dir}/champions_{season}_cache.json"
        cache_data = {
            'teams': teams,
            'matches': matches,
            'last_updated': datetime.now().isoformat(),
            'total_matches': len(matches),
            'total_teams': len(teams)
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Cache guardado: {cache_file}")
    
    def get_cached_data(self, season: int = 2024) -> Optional[Dict]:
        """Obtener datos cacheados"""
        cache_file = f"{self.cache_dir}/champions_{season}_cache.json"
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error leyendo cache: {e}")
        
        return None
    
    async def update_data(self, force_refresh: bool = False):
        """Actualizar datos con opción de forzar refresh"""
        if not force_refresh:
            # Verificar cache reciente
            cache = self.get_cached_data()
            if cache:
                last_updated = datetime.fromisoformat(cache['last_updated'])
                if datetime.now() - last_updated < timedelta(hours=24):
                    print("📁 Usando cache reciente (menos de 24h)")
                    return cache
        
        print("🔄 Actualizando datos desde APIs...")
        await self.collect_champions_data_async([2024])
        return self.get_cached_data()

# Función principal para testing
async def main():
    """Función principal para testing del colector"""
    collector = ChampionsLeagueDataCollector()
    
    print("🏆 CHAMPIONS LEAGUE DATA COLLECTOR")
    print("=" * 50)
    
    # Colectar datos
    await collector.collect_champions_data_async([2023, 2024])
    
    # Mostrar resumen
    cache = collector.get_cached_data(2024)
    if cache:
        print(f"\n📊 RESUMEN TEMPORADA 2024:")
        print(f"   Equipos: {cache['total_teams']}")
        print(f"   Partidos: {cache['total_matches']}")
        print(f"   Actualizado: {cache['last_updated']}")

if __name__ == "__main__":
    asyncio.run(main())