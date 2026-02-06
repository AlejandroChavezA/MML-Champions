import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import asyncio

class ChampionsFeatureEngineer:
    """Feature Engineer especializado para UEFA Champions League"""
    
    def __init__(self, data_dir: str = "../data/cleaned"):
        self.data_dir = data_dir
        self.matches_2023 = None
        self.matches_2024 = None
        self.teams_df = None
        
        # Coeficientes UEFA y presupuestos (integrados)
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
        
        # Experiencia Champions League
        self.champions_experience = {
            'Real Madrid': 14, 'AC Milan': 7, 'Liverpool': 6, 'Bayern Munich': 6,
            'Barcelona': 5, 'Manchester United': 3, 'Juventus': 2, 'Chelsea': 2,
            'Manchester City': 1, 'Inter': 1, 'Ajax': 4, 'Porto': 2,
            'Arsenal': 0, 'Tottenham': 0, 'Paris Saint Germain': 0, 'Atlético Madrid': 0,
            'Borussia Dortmund': 1, 'Napoli': 0, 'RB Leipzig': 0
        }
        
    def load_data(self):
        """Cargar datos de Champions League"""
        try:
            # Intentar cargar datos procesados
            self.matches_2023 = pd.read_csv(f"{self.data_dir}/champions_matches_2023_cleaned.csv")
            self.matches_2024 = pd.read_csv(f"{self.data_dir}/champions_matches_2024_cleaned.csv")
            self.teams_df = pd.read_csv(f"{self.data_dir}/champions_teams_cleaned.csv")
            
            # Convertir fechas
            for df in [self.matches_2023, self.matches_2024]:
                df['date'] = pd.to_datetime(df['date'])
                
            print("✅ Datos Champions League cargados exitosamente")
            return True
            
        except FileNotFoundError:
            print("📁 Datos procesados no encontrados, cargando datos crudos...")
            return self._load_raw_data()
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    def _load_raw_data(self):
        """Cargar datos crudos y procesarlos"""
        try:
            # Cargar datos crudos
            self.matches_2023 = pd.read_csv(f"../data/raw/champions_matches_2023.csv")
            self.matches_2024 = pd.read_csv(f"../data/raw/champions_matches_2024.csv")
            
            # Convertir fechas
            for df in [self.matches_2023, self.matches_2024]:
                df['date'] = pd.to_datetime(df['date'])
            
            print("✅ Datos crudos cargados, listo para feature engineering")
            return True
            
        except Exception as e:
            print(f"❌ Error cargando datos crudos: {e}")
            return False
    
    async def create_champions_features_async(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Crear características Champions League de forma asíncrona"""
        print("🏆 CREANDO CARACTERÍSTICAS CHAMPIONS LEAGUE...")
        
        features_list = []
        
        # Procesar partidos en lotes asíncronos
        batch_size = 50
        total_matches = len(matches_df)
        
        for i in range(0, total_matches, batch_size):
            batch = matches_df.iloc[i:i+batch_size]
            
            # Crear tareas asíncronas para el lote
            tasks = []
            for _, match in batch.iterrows():
                task = self._create_match_features_async(match)
                tasks.append(task)
            
            # Esperar a que se completen todas las tareas del lote
            batch_features = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrar resultados válidos
            for features in batch_features:
                if isinstance(features, dict):
                    features_list.append(features)
                elif isinstance(features, Exception):
                    print(f"⚠️ Error en partido: {features}")
            
            # Progreso
            processed = min(i + batch_size, total_matches)
            print(f"📊 Procesados: {processed}/{total_matches} ({processed/total_matches*100:.1f}%)")
        
        # Convertir a DataFrame
        features_df = pd.DataFrame(features_list)
        
        print(f"✅ Características Champions creadas: {len(features_df)} partidos, {len(features_df.columns)} features")
        return features_df
    
    async def _create_match_features_async(self, match: pd.Series) -> Dict:
        """Crear características para un partido específico de forma asíncrona"""
        try:
            # Datos básicos del partido
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = pd.to_datetime(match['date'])
            stage = match.get('stage', 'GROUP_STAGE')
            
            # Simular delay para procesamiento asíncrono
            await asyncio.sleep(0.001)
            
            # Características básicas
            features = {
                'match_id': match.get('match_id', ''),
                'date': match_date,
                'home_team': home_team,
                'away_team': away_team,
                'stage': stage,
                'is_knockout': stage != 'GROUP_STAGE',
                'home_score': match.get('home_score'),
                'away_score': match.get('away_score'),
                'status': match.get('status', 'SCHEDULED')
            }
            
            # 1. CARACTERÍSTICAS UEFA (CLAVE CHAMPIONS)
            home_uefa = self.uefa_coefficients.get(home_team, 80)
            away_uefa = self.uefa_coefficients.get(away_team, 80)
            
            features.update({
                'home_uefa_coefficient': home_uefa,
                'away_uefa_coefficient': away_uefa,
                'uefa_coefficient_diff': home_uefa - away_uefa,
                'uefa_coefficient_ratio': home_uefa / max(away_uefa, 1),
                'uefa_rank_home': self._get_uefa_rank(home_uefa),
                'uefa_rank_away': self._get_uefa_rank(away_uefa)
            })
            
            # 2. CARACTERÍSTICAS FINANCIERAS
            home_budget = self.club_budgets.get(home_team, 0.5)
            away_budget = self.club_budgets.get(away_team, 0.5)
            
            features.update({
                'home_budget_billions': home_budget,
                'away_budget_billions': away_budget,
                'budget_diff': home_budget - away_budget,
                'budget_ratio': home_budget / max(away_budget, 0.1),
                'home_budget_tier': self._get_budget_tier(home_budget),
                'away_budget_tier': self._get_budget_tier(away_budget)
            })
            
            # 3. CARACTERÍSTICAS DE VIAJE (INTERNACIONAL)
            travel_distance = match.get('travel_distance_km', self._estimate_travel_distance(home_team, away_team))
            
            features.update({
                'travel_distance_km': travel_distance,
                'travel_fatigue_factor': self._get_travel_fatigue(travel_distance),
                'is_long_distance': travel_distance > 1000,
                'travel_tier': self._get_travel_tier(travel_distance)
            })
            
            # 4. CARACTERÍSTICAS DE PRESIÓN (POR ETAPA)
            pressure_factor = self._get_stage_pressure(stage)
            
            features.update({
                'pressure_factor': pressure_factor,
                'stage_importance': self._get_stage_importance(stage),
                'is_final': stage == 'FINAL',
                'is_semi_final': stage == 'SEMI_FINALS'
            })
            
            # 5. EXPERIENCIA EUROPEA
            home_experience = self.champions_experience.get(home_team, 0)
            away_experience = self.champions_experience.get(away_team, 0)
            
            features.update({
                'home_champions_wins': home_experience,
                'away_champions_wins': away_experience,
                'champions_wins_diff': home_experience - away_experience,
                'home_champion_club': home_experience > 0,
                'away_champion_club': away_experience > 0,
                'experience_ratio': home_experience / max(away_experience, 1)
            })
            
            # 6. CARACTERÍSTICAS HISTÓRICAS (HEAD-TO-HEAD)
            h2h_features = await self._calculate_head_to_head_async(home_team, away_team, match_date)
            features.update(h2h_features)
            
            # 7. CARACTERÍSTICAS DE FORMA RECIENTE
            form_features = await self._calculate_recent_form_async(home_team, away_team, match_date)
            features.update(form_features)
            
            # 8. CARACTERÍSTICAS ESPECÍFICAS DE CHAMPIONS
            champions_specific = await self._calculate_champions_specific_async(home_team, away_team, stage, match_date)
            features.update(champions_specific)
            
            return features
            
        except Exception as e:
            print(f"❌ Error creando features para {match.get('home_team', 'Unknown')}: {e}")
            raise
    
    async def _calculate_head_to_head_async(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Calcular head-to-head histórico de forma asíncrona"""
        try:
            # Simular consulta asíncrona a base de datos
            await asyncio.sleep(0.001)
            
            # Combinar datos históricos
            historical_matches = []
            if self.matches_2023 is not None:
                historical_matches.append(self.matches_2023)
            if self.matches_2024 is not None:
                historical_matches.append(self.matches_2024)
            
            if not historical_matches:
                return self._get_default_h2h_features()
            
            all_matches = pd.concat(historical_matches)
            
            # Filtrar head-to-head
            h2h_mask = (
                ((all_matches['home_team'] == home_team) & (all_matches['away_team'] == away_team)) |
                ((all_matches['home_team'] == away_team) & (all_matches['away_team'] == home_team))
            )
            date_mask = all_matches['date'] < match_date
            
            h2h_matches = all_matches[h2h_mask & date_mask]
            
            if len(h2h_matches) == 0:
                return self._get_default_h2h_features()
            
            # Calcular estadísticas
            home_wins = len(h2h_matches[
                (h2h_matches['home_team'] == home_team) & 
                (h2h_matches['home_score'] > h2h_matches['away_score'])
            ])
            
            away_wins = len(h2h_matches[
                (h2h_matches['home_team'] == away_team) & 
                (h2h_matches['home_score'] > h2h_matches['away_score'])
            ])
            
            draws = len(h2h_matches[h2h_matches['home_score'] == h2h_matches['away_score']])
            
            total_h2h = len(h2h_matches)
            
            # Goles
            home_goals = h2h_matches[h2h_matches['home_team'] == home_team]['home_score'].sum()
            away_goals = h2h_matches[h2h_matches['home_team'] == away_team]['away_score'].sum()
            
            return {
                'h2h_total_matches': total_h2h,
                'h2h_home_wins': home_wins,
                'h2h_away_wins': away_wins,
                'h2h_draws': draws,
                'h2h_home_win_rate': home_wins / total_h2h if total_h2h > 0 else 0,
                'h2h_home_goals': home_goals,
                'h2h_away_goals': away_goals,
                'h2h_goal_diff': home_goals - away_goals,
                'h2h_recent_form': self._get_h2h_recent_form(h2h_matches, home_team)
            }
            
        except Exception as e:
            print(f"❌ Error en head-to-head: {e}")
            return self._get_default_h2h_features()
    
    async def _calculate_recent_form_async(self, home_team: str, away_team: str, match_date: datetime) -> Dict:
        """Calcular forma reciente de forma asíncrona"""
        try:
            await asyncio.sleep(0.001)
            
            # Combinar datos históricos
            historical_matches = []
            if self.matches_2023 is not None:
                historical_matches.append(self.matches_2023)
            if self.matches_2024 is not None:
                historical_matches.append(self.matches_2024)
            
            if not historical_matches:
                return self._get_default_form_features()
            
            all_matches = pd.concat(historical_matches)
            
            # Forma reciente (últimos 5 partidos)
            recent_features = {}
            
            for team, prefix in [(home_team, 'home'), (away_team, 'away')]:
                team_matches = all_matches[
                    ((all_matches['home_team'] == team) | (all_matches['away_team'] == team)) &
                    (all_matches['date'] < match_date)
                ].sort_values('date', ascending=False).head(5)
                
                if len(team_matches) > 0:
                    # Calcular estadísticas
                    wins = 0
                    draws = 0
                    goals_for = 0
                    goals_against = 0
                    
                    for _, match in team_matches.iterrows():
                        if match['home_team'] == team:
                            team_score = match['home_score']
                            opponent_score = match['away_score']
                        else:
                            team_score = match['away_score']
                            opponent_score = match['home_score']
                        
                        goals_for += team_score if pd.notna(team_score) else 0
                        goals_against += opponent_score if pd.notna(opponent_score) else 0
                        
                        if team_score > opponent_score:
                            wins += 1
                        elif team_score == opponent_score:
                            draws += 1
                    
                    recent_features.update({
                        f'{prefix}_last5_matches': len(team_matches),
                        f'{prefix}_last5_wins': wins,
                        f'{prefix}_last5_draws': draws,
                        f'{prefix}_last5_losses': len(team_matches) - wins - draws,
                        f'{prefix}_last5_points': wins * 3 + draws,
                        f'{prefix}_last5_win_rate': wins / len(team_matches) if len(team_matches) > 0 else 0,
                        f'{prefix}_last5_goals_for': goals_for,
                        f'{prefix}_last5_goals_against': goals_against,
                        f'{prefix}_last5_goal_diff': goals_for - goals_against,
                        f'{prefix}_last5_points_per_game': (wins * 3 + draws) / len(team_matches) if len(team_matches) > 0 else 0
                    })
                else:
                    # Valores por defecto si no hay partidos
                    for key in ['matches', 'wins', 'draws', 'losses', 'points', 'win_rate', 'goals_for', 'goals_against', 'goal_diff', 'points_per_game']:
                        recent_features[f'{prefix}_last5_{key}'] = 0
            
            return recent_features
            
        except Exception as e:
            print(f"❌ Error en forma reciente: {e}")
            return self._get_default_form_features()
    
    async def _calculate_champions_specific_async(self, home_team: str, away_team: str, stage: str, match_date: datetime) -> Dict:
        """Calcular características específicas de Champions League"""
        try:
            await asyncio.sleep(0.001)
            
            features = {}
            
            # Ventaja local en Champions (diferente a liga doméstica)
            home_advantage = self._get_champions_home_advantage(stage)
            features['champions_home_advantage'] = home_advantage
            
            # Factor de eliminación
            if stage != 'GROUP_STAGE':
                knockout_features = self._get_knockout_features(home_team, away_team, stage)
                features.update(knockout_features)
            else:
                features.update({
                    'is_knockout_match': 0,
                    'knockout_pressure_multiplier': 1.0,
                    'away_goals_advantage': 0
                })
            
            # Factor de experiencia europea
            home_exp = self.champions_experience.get(home_team, 0)
            away_exp = self.champions_experience.get(away_team, 0)
            
            features.update({
                'home_european_experience': home_exp,
                'away_european_experience': away_exp,
                'european_experience_gap': home_exp - away_exp,
                'home_big_club_status': 1 if home_exp >= 3 else 0,
                'away_big_club_status': 1 if away_exp >= 3 else 0
            })
            
            # Características de formato
            features.update({
                'competition_format': self._get_format_code(stage),
                'match_importance_level': self._get_importance_level(stage),
                'television_pressure': self._get_tv_pressure_factor(stage)
            })
            
            return features
            
        except Exception as e:
            print(f"❌ Error en características Champions: {e}")
            return {}
    
    def _get_uefa_rank(self, coefficient: float) -> int:
        """Obtener ranking UEFA a partir del coeficiente"""
        sorted_coeffs = sorted(self.uefa_coefficients.values(), reverse=True)
        return sorted_coeffs.index(coefficient) + 1 if coefficient in sorted_coeffs else len(sorted_coeffs)
    
    def _get_budget_tier(self, budget: float) -> str:
        """Clasificar presupuesto en tiers"""
        if budget >= 1.2:
            return 'ELITE'
        elif budget >= 0.8:
            return 'TOP_TIER'
        elif budget >= 0.5:
            return 'UPPER_MID'
        elif budget >= 0.3:
            return 'MID_TIER'
        else:
            return 'LOWER_TIER'
    
    def _estimate_travel_distance(self, home_team: str, away_team: str) -> float:
        """Estimar distancia de viaje entre equipos"""
        # Distancias estimadas (km)
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
            'Liverpool': 'Manchester',
            'Barcelona': 'Madrid',
            'Juventus': 'Milan',
            'Borussia Dortmund': 'Munich',
            'Napoli': 'Milan',
            'RB Leipzig': 'Munich',
            'Porto': 'Madrid',
            'Ajax': 'Amsterdam'
        }
        
        home_city = team_cities.get(home_team, 'London')
        away_city = team_cities.get(away_team, 'London')
        
        if home_city in city_distances and away_city in city_distances[home_city]:
            return city_distances[home_city][away_city]
        else:
            return 500  # Distancia por defecto
    
    def _get_travel_fatigue(self, distance: float) -> float:
        """Calcular factor de fatiga por viaje"""
        if distance < 300:
            return 1.0
        elif distance < 800:
            return 1.05
        elif distance < 1500:
            return 1.10
        else:
            return 1.15
    
    def _get_travel_tier(self, distance: float) -> str:
        """Clasificar distancia en tiers"""
        if distance < 300:
            return 'SHORT'
        elif distance < 800:
            return 'MEDIUM'
        elif distance < 1500:
            return 'LONG'
        else:
            return 'VERY_LONG'
    
    def _get_stage_pressure(self, stage: str) -> float:
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
    
    def _get_stage_importance(self, stage: str) -> int:
        """Obtener nivel de importancia por etapa"""
        importance_map = {
            'GROUP_STAGE': 1,
            'ROUND_OF_16': 2,
            'QUARTER_FINALS': 3,
            'SEMI_FINALS': 4,
            'FINAL': 5,
            'UNKNOWN': 1
        }
        return importance_map.get(stage, 1)
    
    def _get_champions_home_advantage(self, stage: str) -> float:
        """Obtener ventaja local en Champions League"""
        if stage == 'GROUP_STAGE':
            return 0.15  # Mayor ventaja local en grupos
        else:
            return 0.10  # Menor ventaja local en eliminatorias
    
    def _get_knockout_features(self, home_team: str, away_team: str, stage: str) -> Dict:
        """Obtener características de eliminatorias"""
        home_exp = self.champions_experience.get(home_team, 0)
        away_exp = self.champions_experience.get(away_team, 0)
        
        return {
            'is_knockout_match': 1,
            'knockout_pressure_multiplier': 1.0 + self._get_stage_pressure(stage),
            'away_goals_advantage': 0.2,  # Gol visitante más importante en eliminatorias
            'home_knockout_experience': home_exp,
            'away_knockout_experience': away_exp
        }
    
    def _get_format_code(self, stage: str) -> int:
        """Obtener código de formato"""
        format_codes = {
            'GROUP_STAGE': 1,
            'ROUND_OF_16': 2,
            'QUARTER_FINALS': 3,
            'SEMI_FINALS': 4,
            'FINAL': 5,
            'UNKNOWN': 0
        }
        return format_codes.get(stage, 0)
    
    def _get_importance_level(self, stage: str) -> float:
        """Obtener nivel de importancia"""
        importance_levels = {
            'GROUP_STAGE': 0.3,
            'ROUND_OF_16': 0.6,
            'QUARTER_FINALS': 0.8,
            'SEMI_FINALS': 0.9,
            'FINAL': 1.0,
            'UNKNOWN': 0.5
        }
        return importance_levels.get(stage, 0.5)
    
    def _get_tv_pressure_factor(self, stage: str) -> float:
        """Obtener factor de presión televisiva"""
        tv_factors = {
            'GROUP_STAGE': 1.0,
            'ROUND_OF_16': 1.2,
            'QUARTER_FINALS': 1.4,
            'SEMI_FINALS': 1.6,
            'FINAL': 2.0,
            'UNKNOWN': 1.0
        }
        return tv_factors.get(stage, 1.0)
    
    def _get_h2h_recent_form(self, h2h_matches: pd.DataFrame, team: str) -> float:
        """Calcular forma reciente en head-to-head"""
        if len(h2h_matches) == 0:
            return 0.0
        
        # Últimos 3 encuentros
        recent_h2h = h2h_matches.tail(3)
        points = 0
        
        for _, match in recent_h2h.iterrows():
            if match['home_team'] == team:
                if match['home_score'] > match['away_score']:
                    points += 3
                elif match['home_score'] == match['away_score']:
                    points += 1
            else:
                if match['away_score'] > match['home_score']:
                    points += 3
                elif match['away_score'] == match['home_score']:
                    points += 1
        
        return points / (len(recent_h2h) * 3) if len(recent_h2h) > 0 else 0.0
    
    def _get_default_h2h_features(self) -> Dict:
        """Características head-to-head por defecto"""
        return {
            'h2h_total_matches': 0,
            'h2h_home_wins': 0,
            'h2h_away_wins': 0,
            'h2h_draws': 0,
            'h2h_home_win_rate': 0,
            'h2h_home_goals': 0,
            'h2h_away_goals': 0,
            'h2h_goal_diff': 0,
            'h2h_recent_form': 0
        }
    
    def _get_default_form_features(self) -> Dict:
        """Características de forma por defecto"""
        defaults = {}
        for prefix in ['home', 'away']:
            for key in ['matches', 'wins', 'draws', 'losses', 'points', 'win_rate', 'goals_for', 'goals_against', 'goal_diff', 'points_per_game']:
                defaults[f'{prefix}_last5_{key}'] = 0
        return defaults
    
    def get_feature_summary(self) -> Dict:
        """Obtener resumen de características disponibles"""
        return {
            'total_features': 47,
            'uefa_features': 6,
            'financial_features': 4,
            'travel_features': 4,
            'pressure_features': 4,
            'experience_features': 6,
            'head_to_head_features': 9,
            'form_features': 20,
            'champions_specific_features': 6
        }

# Función principal para testing
async def main():
    """Función principal para testing del feature engineering"""
    engineer = ChampionsFeatureEngineer()
    
    print("🏆 CHAMPIONS LEAGUE FEATURE ENGINEERING")
    print("=" * 50)
    
    # Cargar datos
    if not engineer.load_data():
        print("❌ No se pudieron cargar los datos")
        return
    
    # Crear features para datos 2024
    if engineer.matches_2024 is not None:
        print("📊 Creando características para temporada 2024...")
        features_df = await engineer.create_champions_features_async(engineer.matches_2024)
        
        # Guardar features
        features_df.to_csv("../data/processed/champions_features_2024.csv", index=False)
        print(f"💾 Características guardadas: ../data/processed/champions_features_2024.csv")
        
        # Mostrar resumen
        print(f"\n📈 RESUMEN:")
        print(f"   Partidos procesados: {len(features_df)}")
        print(f"   Características creadas: {len(features_df.columns)}")
        
        # Mostrar algunas características clave
        key_features = ['home_uefa_coefficient', 'away_uefa_coefficient', 'uefa_coefficient_diff',
                        'travel_distance_km', 'pressure_factor', 'home_champions_wins']
        
        print(f"\n🎯 EJEMPLO DE CARACTERÍSTICAS CLAVE:")
        for feature in key_features:
            if feature in features_df.columns:
                print(f"   {feature}: {features_df[feature].iloc[0]:.3f}")
    
    # Resumen de características
    summary = engineer.get_feature_summary()
    print(f"\n🏆 RESUMEN DE CARACTERÍSTICAS CHAMPIONS:")
    for category, count in summary.items():
        if category != 'total_features':
            print(f"   {category}: {count}")

if __name__ == "__main__":
    asyncio.run(main())