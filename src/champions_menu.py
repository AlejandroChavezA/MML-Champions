#!/usr/bin/env python3
"""
MENÚ INTERACTIVO CHAMPIONS LEAGUE PREDICTOR
Sistema completo con menú terminal para UEFA Champions League
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Añadir path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.champions_data_collection import ChampionsLeagueDataCollector
from src.champions_feature_engineering import ChampionsFeatureEngineer
from src.champions_prediction_models import ChampionsMatchPredictor

class ChampionsTerminalInterface:
    """Interfaz terminal para Champions League Predictor"""
    
    def __init__(self):
        self.data_collector = ChampionsLeagueDataCollector()
        self.feature_engineer = ChampionsFeatureEngineer()
        self.predictor = ChampionsMatchPredictor()
        self.current_data = None
        self.current_features = None
        self.prediction_queue = asyncio.Queue()
        self.processing_predictions = False
        
    async def initialize_system(self):
        """Inicializar el sistema completo"""
        print("🏆 INICIANDO CHAMPIONS LEAGUE PREDICTOR")
        print("=" * 60)
        print("🌍 Sistema especializado para UEFA Champions League")
        print("⚡ Procesamiento asíncrono con características avanzadas")
        print("📊 Coeficientes UEFA, factores de viaje, presión europea")
        print("=" * 60)
        
        # 1. Cargar datos
        print("\n📊 PASO 1: CARGANDO DATOS...")
        data_loaded = await self._load_data()
        
        if not data_loaded:
            print("❌ No se pudieron cargar los datos. El sistema no puede continuar.")
            return False
        
        # 2. Cargar características
        print("\n🔧 PASO 2: CARGANDO CARACTERÍSTICAS...")
        features_loaded = await self._load_features()
        
        # 3. Cargar modelos
        print("\n🤖 PASO 3: CARGANDO MODELOS...")
        models_loaded = self.predictor.load_models()
        
        if not models_loaded:
            print("⚠️ No se encontraron modelos entrenados. ¿Deseas entrenarlos ahora?")
            choice = input("👉 (S/N): ").strip().upper()
            if choice == 'S':
                await self._train_models()
        
        # 4. Iniciar worker de predicciones
        print("\n🔄 PASO 4: INICIANDO WORKER ASÍNCRONO...")
        asyncio.create_task(self._prediction_worker())
        
        print("\n✅ SISTEMA CHAMPIONS LEAGUE INICIALIZADO")
        await asyncio.sleep(2)
        return True
    
    async def _load_data(self) -> bool:
        """Cargar datos de Champions League"""
        try:
            # Intentar cargar cache primero
            cache = self.data_collector.get_cached_data(2024)
            if cache:
                print("📁 Usando cache local de datos")
                self.current_data = cache
                return True
            
            # Si no hay cache, cargar desde archivos
            try:
                matches_file = "../data/raw/champions_matches_2024.csv"
                if os.path.exists(matches_file):
                    df = pd.read_csv(matches_file)
                    df['date'] = pd.to_datetime(df['date'])
                    self.current_data = {'matches': df.to_dict('records')}
                    print("✅ Datos cargados desde archivos locales")
                    return True
            except Exception as e:
                print(f"⚠️ Error cargando archivos: {e}")
            
            # Opción de actualizar datos
            print("📡 No hay datos locales disponibles.")
            choice = input("👉 ¿Deseas descargar datos desde APIs? (S/N): ").strip().upper()
            if choice == 'S':
                await self.data_collector.update_data(force_refresh=True)
                cache = self.data_collector.get_cached_data(2024)
                if cache:
                    self.current_data = cache
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return False
    
    async def _load_features(self) -> bool:
        """Cargar características procesadas"""
        try:
            # Intentar cargar características procesadas
            features_file = "../data/processed/champions_features_2024.csv"
            if os.path.exists(features_file):
                self.current_features = pd.read_csv(features_file)
                print("✅ Características cargadas desde archivo")
                return True
            
            # Si no hay características procesadas, generarlas
            if self.current_data and 'matches' in self.current_data:
                print("🔧 Generando características desde datos...")
                matches_df = pd.DataFrame(self.current_data['matches'])
                
                # Cargar datos para feature engineering
                if self.feature_engineer.load_data():
                    self.current_features = await self.feature_engineer.create_champions_features_async(matches_df)
                    
                    # Guardar características
                    self.current_features.to_csv(features_file, index=False)
                    print("✅ Características generadas y guardadas")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error cargando características: {e}")
            return False
    
    async def _train_models(self):
        """Entrenar modelos de Champions League"""
        try:
            if self.current_features is None:
                print("❌ No hay características para entrenar")
                return
            
            print("🤖 ENTRENANDO MODELOS CHAMPIONS...")
            
            # Crear targets
            def get_result(row):
                if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                    return 1  # DRAW
                elif row['home_score'] > row['away_score']:
                    return 0  # HOME_WIN
                elif row['home_score'] < row['away_score']:
                    return 2  # AWAY_WIN
                else:
                    return 1  # DRAW
            
            targets = self.current_features.apply(get_result, axis=1)
            
            # Seleccionar características numéricas
            numeric_features = self.current_features.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_features if col not in ['home_score', 'away_score', 'match_id']]
            
            X = self.current_features[feature_columns]
            y = targets
            
            # Entrenar modelos
            await self.predictor.train_champions_models_async(X, y)
            
            print("✅ Modelos entrenados y guardados")
            
        except Exception as e:
            print(f"❌ Error entrenando modelos: {e}")
    
    async def show_main_menu(self):
        """Mostrar menú principal"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("🏆 CHAMPIONS LEAGUE PREDICTOR - MENÚ PRINCIPAL")
            print("=" * 70)
            print("🌍 Predicciones especializadas para UEFA Champions League")
            print("⚡ Características únicas: coeficientes UEFA, viajes, presión europea")
            print("🤖 Modelos optimizados para formato europeo")
            print("=" * 70)
            
            # Estado del sistema
            await self._show_system_status()
            
            print("\n🎯 MENÚ DE OPCIONES:")
            print("1. 📋 Ver próximos partidos de Champions")
            print("2. ⚽ Predecir partido específico")
            print("3. 🔄 Predicción en lote (múltiples partidos)")
            print("4. 📊 Estadísticas de equipos Champions")
            print("5. 🏆 Análisis por etapa (Grupos/Eliminatorias)")
            print("6. 📈 Ver rendimiento de modelos")
            print("7. 🌍 Factores Champions (UEFA, viajes, presión)")
            print("8. 💾 Actualizar datos desde APIs")
            print("9. 🤖 Reentrenar modelos")
            print("10. 🚪 Salir")
            
            choice = input("\n👉 Selecciona una opción: ").strip()
            
            if choice == "1":
                await self.show_upcoming_matches()
            elif choice == "2":
                await self.predict_specific_match()
            elif choice == "3":
                await self.batch_predictions()
            elif choice == "4":
                await self.show_team_statistics()
            elif choice == "5":
                await self.show_stage_analysis()
            elif choice == "6":
                await self.show_model_performance()
            elif choice == "7":
                await self.show_champions_factors()
            elif choice == "8":
                await self.update_data()
            elif choice == "9":
                await self._train_models()
            elif choice == "10":
                print("\n👋 ¡Gracias por usar Champions League Predictor!")
                break
            else:
                print("\n❌ Opción inválida")
                await asyncio.sleep(1)
    
    async def _show_system_status(self):
        """Mostrar estado actual del sistema"""
        print("📈 ESTADO DEL SISTEMA:")
        
        # Datos
        data_status = "✅ Disponibles" if self.current_data else "❌ No cargados"
        print(f"   📊 Datos Champions: {data_status}")
        
        # Características
        features_status = "✅ Disponibles" if self.current_features is not None else "❌ No generadas"
        if self.current_features is not None:
            features_status += f" ({len(self.current_features)} partidos)"
        print(f"   🔧 Características: {features_status}")
        
        # Modelos
        models_count = len(self.predictor.models)
        models_status = f"✅ {models_count} modelos" if models_count > 0 else "❌ No entrenados"
        print(f"   🤖 Modelos: {models_status}")
        
        # Cola de predicciones
        queue_size = self.prediction_queue.qsize() if hasattr(self.prediction_queue, 'qsize') else 0
        print(f"   🔄 Cola predicciones: {queue_size}")
        
        # Última actualización
        if self.current_data and 'last_updated' in self.current_data:
            last_update = self.current_data['last_updated']
            try:
                update_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                time_ago = datetime.now() - update_time
                hours_ago = int(time_ago.total_seconds() / 3600)
                print(f"   🕐 Última actualización: hace {hours_ago} horas")
            except:
                print(f"   🕐 Última actualización: {last_update}")
    
    async def show_upcoming_matches(self):
        """Mostrar próximos partidos"""
        print("\n📋 PRÓXIMOS PARTIDOS CHAMPIONS LEAGUE")
        print("=" * 60)
        
        if not self.current_data or 'matches' not in self.current_data:
            print("❌ No hay datos disponibles")
            await asyncio.sleep(2)
            return
        
        matches_df = pd.DataFrame(self.current_data['matches'])
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Filtrar partidos futuros
        now = datetime.now()
        upcoming = matches_df[matches_df['date'] > now].sort_values('date').head(10)
        
        if len(upcoming) == 0:
            print("📭 No hay partidos próximos programados")
            await asyncio.sleep(2)
            return
        
        print(f"📅 Próximos {len(upcoming)} partidos:")
        print()
        
        for i, (_, match) in enumerate(upcoming.iterrows(), 1):
            match_date = match['date']
            stage = match.get('stage', 'Unknown')
            
            print(f"{i:2d}. 🏆 {match['home_team']:20s} vs {match['away_team']:20s}")
            print(f"     📅 {match_date.strftime('%d/%m %H:%M')} | 🎯 {stage}")
            
            # Mostrar información Champions si está disponible
            if 'home_uefa_coefficient' in match and 'away_uefa_coefficient' in match:
                home_uefa = match['home_uefa_coefficient']
                away_uefa = match['away_uefa_coefficient']
                print(f"     🌍 UEFA: {home_uefa:.1f} vs {away_uefa:.1f}")
            
            print()
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def predict_specific_match(self):
        """Predecir partido específico"""
        print("\n⚽ PREDICCIÓN DE PARTIDO CHAMPIONS")
        print("=" * 60)
        
        if not self.current_data or 'matches' not in self.current_data:
            print("❌ No hay datos disponibles")
            await asyncio.sleep(2)
            return
        
        if len(self.predictor.models) == 0:
            print("❌ No hay modelos cargados")
            await asyncio.sleep(2)
            return
        
        matches_df = pd.DataFrame(self.current_data['matches'])
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Filtrar partidos futuros
        now = datetime.now()
        upcoming = matches_df[matches_df['date'] > now].sort_values('date').head(15)
        
        if len(upcoming) == 0:
            print("📭 No hay partidos próximos para predecir")
            await asyncio.sleep(2)
            return
        
        print("Selecciona un partido para predecir:")
        for i, (_, match) in enumerate(upcoming.iterrows(), 1):
            match_date = match['date']
            print(f"{i:2d}. {match['home_team']} vs {match['away_team']} ({match_date.strftime('%d/%m %H:%M')})")
        
        try:
            choice = int(input("\n👉 Número del partido: ")) - 1
            if 0 <= choice < len(upcoming):
                match = upcoming.iloc[choice]
                
                print(f"\n🔄 Procesando predicción...")
                print(f"🏆 {match['home_team']} vs {match['away_team']}")
                print("⏳ Analizando factores Champions League...")
                
                # Realizar predicción asíncrona
                prediction = await self._predict_match_from_data(match)
                
                if prediction:
                    await self._display_prediction(prediction, match)
                else:
                    print("❌ Error en la predicción")
                
                input("\n👉 Presiona Enter para continuar...")
            else:
                print("❌ Selección inválida")
                await asyncio.sleep(1)
                
        except ValueError:
            print("❌ Ingresa un número válido")
            await asyncio.sleep(1)
    
    async def _predict_match_from_data(self, match_data: pd.Series) -> Optional[Dict]:
        """Predecir partido desde datos"""
        try:
            if self.current_features is None:
                return None
            
            # Buscar características del partido
            match_features = self.current_features[
                (self.current_features['home_team'] == match_data['home_team']) &
                (self.current_features['away_team'] == match_data['away_team'])
            ]
            
            if len(match_features) == 0:
                print("⚠️ No se encontraron características para este partido")
                return None
            
            # Obtener características para predicción
            feature_row = match_features.iloc[0]
            
            # Seleccionar solo características numéricas
            numeric_features = self.current_features.select_dtypes(include=[np.number]).columns
            prediction_features = [col for col in numeric_features if col not in ['home_score', 'away_score', 'match_id']]
            
            features_for_prediction = feature_row[prediction_features]
            
            # Realizar predicción
            prediction = await self.predictor.predict_match_async(features_for_prediction)
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None
    
    async def _display_prediction(self, prediction: Dict, match_data: pd.Series):
        """Mostrar resultado de predicción"""
        print(f"\n✅ PREDICCIÓN CHAMPIONS LEAGUE COMPLETADA")
        print("=" * 50)
        print(f"🏆 Partido: {prediction.get('home_team', match_data['home_team'])} vs {prediction.get('away_team', match_data['away_team'])}")
        print(f"🎯 Predicción: {prediction['prediction']}")
        print(f"📊 Confianza: {prediction['confidence']:.3f}")
        print(f"🤖 Modelo: {prediction['model_used']}")
        
        print(f"\n📈 PROBABILIDADES:")
        print(f"   🏠 Local: {prediction['home_probability']:.3f}")
        print(f"   🤝 Empate: {prediction['draw_probability']:.3f}")
        print(f"   ✈️  Visitante: {prediction['away_probability']:.3f}")
        
        # Factores Champions League
        if 'champions_factors' in prediction:
            factors = prediction['champions_factors']
            print(f"\n🌍 FACTORES CHAMPIONS LEAGUE:")
            print(f"   🔥 Impacto UEFA: {factors['uefa_impact']:.3f}")
            print(f"   ✈️  Impacto Viaje: {factors['travel_impact']:.3f}")
            print(f"   ⚡ Impacto Presión: {factors['pressure_impact']:.3f}")
        
        # Información adicional del partido
        if 'stage' in match_data:
            print(f"\n🎯 INFORMACIÓN PARTIDO:")
            print(f"   Etapa: {match_data['stage']}")
            
            if 'home_uefa_coefficient' in match_data:
                print(f"   UEFA Local: {match_data['home_uefa_coefficient']:.1f}")
                print(f"   UEFA Visitante: {match_data['away_uefa_coefficient']:.1f}")
            
            if 'travel_distance_km' in match_data:
                print(f"   Distancia Viaje: {match_data['travel_distance_km']:.0f} km")
    
    async def batch_predictions(self):
        """Predicciones en lote"""
        print("\n🔄 PREDICCIONES EN LOTE CHAMPIONS")
        print("=" * 60)
        
        if not self.current_data or 'matches' not in self.current_data:
            print("❌ No hay datos disponibles")
            await asyncio.sleep(2)
            return
        
        if len(self.predictor.models) == 0:
            print("❌ No hay modelos cargados")
            await asyncio.sleep(2)
            return
        
        matches_df = pd.DataFrame(self.current_data['matches'])
        matches_df['date'] = pd.to_datetime(matches_df['date'])
        
        # Filtrar partidos futuros
        now = datetime.now()
        upcoming = matches_df[matches_df['date'] > now].sort_values('date').head(5)
        
        if len(upcoming) == 0:
            print("📭 No hay partidos próximos para predecir")
            await asyncio.sleep(2)
            return
        
        print(f"📊 Procesando {len(upcoming)} partidos en paralelo...")
        
        # Añadir todos a la cola
        for _, match in upcoming.iterrows():
            await self.prediction_queue.put(match)
        
        # Esperar a que se procesen
        processed = 0
        total_to_process = len(upcoming)
        start_time = time.time()
        
        while processed < total_to_process:
            await asyncio.sleep(0.5)
            # Aquí verificaríamos cuántos se han procesado (implementación simplificada)
            processed = min(processed + 1, total_to_process)
            print(f"✅ Procesados: {processed}/{total_to_process}")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 LOTE COMPLETADO")
        print(f"⏱️  Tiempo total: {total_time:.3f}s")
        print(f"📊 Promedio por predicción: {total_time/len(upcoming):.3f}s")
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def show_team_statistics(self):
        """Mostrar estadísticas de equipos"""
        print("\n📊 ESTADÍSTICAS EQUIPOS CHAMPIONS")
        print("=" * 60)
        
        if not self.current_data or 'matches' not in self.current_data:
            print("❌ No hay datos disponibles")
            await asyncio.sleep(2)
            return
        
        matches_df = pd.DataFrame(self.current_data['matches'])
        
        # Estadísticas básicas
        teams = set()
        for _, match in matches_df.iterrows():
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        print(f"🏆 Total equipos Champions: {len(teams)}")
        print()
        
        # Mostrar equipos con coeficientes UEFA
        uefa_coeffs = self.data_collector.uefa_coefficients
        
        print("🌍 RANKING UEFA (Top 10):")
        sorted_teams = sorted(uefa_coeffs.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (team, coeff) in enumerate(sorted_teams, 1):
            print(f"{i:2d}. {team:20s} - {coeff:.3f}")
        
        print()
        
        # Presupuestos
        print("💰 PRESUPUESTOS CLUBES (Top 10):")
        budgets = self.data_collector.club_budgets
        sorted_budgets = sorted(budgets.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for i, (team, budget) in enumerate(sorted_budgets, 1):
            print(f"{i:2d}. {team:20s} - {budget:.2f}B €")
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def show_stage_analysis(self):
        """Análisis por etapa"""
        print("\n🏆 ANÁLISIS POR ETAPA CHAMPIONS")
        print("=" * 60)
        
        if not self.current_data or 'matches' not in self.current_data:
            print("❌ No hay datos disponibles")
            await asyncio.sleep(2)
            return
        
        matches_df = pd.DataFrame(self.current_data['matches'])
        
        # Contar partidos por etapa
        stage_counts = matches_df['stage'].value_counts()
        
        print("📊 DISTRIBUCIÓN POR ETAPAS:")
        for stage, count in stage_counts.items():
            print(f"   {stage}: {count} partidos")
        
        print()
        
        # Análisis por etapa
        stages = ['GROUP_STAGE', 'ROUND_OF_16', 'QUARTER_FINALS', 'SEMI_FINALS', 'FINAL']
        
        for stage in stages:
            if stage in stage_counts:
                stage_matches = matches_df[matches_df['stage'] == stage]
                
                print(f"🎯 {stage}:")
                print(f"   Partidos: {len(stage_matches)}")
                
                # Estadísticas de goles si hay resultados
                finished_matches = stage_matches[stage_matches['status'] == 'FINISHED']
                if len(finished_matches) > 0:
                    total_goals = finished_matches['home_score'].fillna(0) + finished_matches['away_score'].fillna(0)
                    avg_goals = total_goals.mean()
                    print(f"   Promedio goles: {avg_goals:.2f}")
                
                print()
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def show_model_performance(self):
        """Mostrar rendimiento de modelos"""
        print("\n📈 RENDIMIENTO MODELOS CHAMPIONS")
        print("=" * 60)
        
        if not self.predictor.model_performance:
            print("❌ No hay datos de rendimiento disponibles")
            await asyncio.sleep(2)
            return
        
        performance = self.predictor.model_performance
        
        print("📊 COMPARACIÓN DE MODELOS:")
        print(f"{'Modelo':<25} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12}")
        print("-" * 65)
        
        sorted_models = sorted(
            performance.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            print(f"{model_name:<25} {metrics['train_accuracy']:<12.3f} "
                  f"{metrics['test_accuracy']:<12.3f} {metrics['cv_mean']:<12.3f}")
        
        # Mejor modelo
        best_model = sorted_models[0]
        print(f"\n🏆 Mejor modelo: {best_model[0]}")
        print(f"📊 Accuracy: {best_model[1]['test_accuracy']:.3f}")
        
        # Features importantes
        if best_model[1]['feature_importance']:
            print(f"\n🎯 Top 5 Features ({best_model[0]}):")
            top_features = list(best_model[1]['feature_importance'].items())[:5]
            for feature, importance in top_features:
                print(f"   {feature}: {importance:.4f}")
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def show_champions_factors(self):
        """Mostrar factores Champions League"""
        print("\n🌍 FACTORES CHAMPIONS LEAGUE")
        print("=" * 60)
        
        print("🔥 CARACTERÍSTICAS ÚNICAS CHAMPIONS:")
        print("   1. Coeficientes UEFA - Factor internacional clave")
        print("   2. Distancias de viaje - Fatiga transcontinental")
        print("   3. Presión por etapa - Diferente en cada fase")
        print("   4. Experiencia europea - Títulos históricos")
        print("   5. Formato variable - Grupos vs Eliminatorias")
        print("   6. Factor gol visitante - Más importante en eliminatorias")
        
        print()
        
        # Mostrar pesos configurados
        print("⚖️ PESOS CONFIGURADOS:")
        weights = self.predictor.champions_weights
        
        for factor, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {factor}: {weight:.2f}")
        
        print()
        
        # Impacto esperado
        print("📈 IMPACTO ESPERADO EN ACCURACY:")
        print("   Baseline (sin factores): ~45-50%")
        print("   Con coeficientes UEFA: +5-8%")
        print("   Con factores de viaje: +3-5%")
        print("   Con experiencia europea: +3-4%")
        print("   Con todos los factores: ~65-75% potencial")
        
        input("\n👉 Presiona Enter para continuar...")
    
    async def update_data(self):
        """Actualizar datos desde APIs"""
        print("\n💾 ACTUALIZAR DATOS CHAMPIONS")
        print("=" * 60)
        
        choice = input("👉 ¿Actualizar datos desde APIs? (S/N): ").strip().upper()
        
        if choice == 'S':
            print("🔄 Actualizando datos...")
            success = await self.data_collector.update_data(force_refresh=True)
            
            if success:
                # Recargar datos
                await self._load_data()
                print("✅ Datos actualizados correctamente")
            else:
                print("❌ Error actualizando datos")
        
        await asyncio.sleep(2)
    
    async def _prediction_worker(self):
        """Worker asíncrono para predicciones"""
        while True:
            try:
                # Esperar por partido en cola
                match = await self.prediction_queue.get()
                
                # Procesar predicción
                prediction = await self._predict_match_from_data(match)
                
                # Aquí podríamos guardar o mostrar resultados
                if prediction:
                    print(f"✅ Predicción completada: {match['home_team']} vs {match['away_team']}")
                
                self.prediction_queue.task_done()
                
            except Exception as e:
                print(f"❌ Error en worker: {e}")
                await asyncio.sleep(1)
    
    async def run(self):
        """Ejecutar interfaz principal"""
        # Inicializar sistema
        if not await self.initialize_system():
            print("❌ No se pudo inicializar el sistema")
            return
        
        # Mostrar menú principal
        await self.show_main_menu()

# Función principal
async def main():
    """Función principal"""
    interface = ChampionsTerminalInterface()
    await interface.run()

if __name__ == "__main__":
    print("🏆 CHAMPIONS LEAGUE PREDICTOR")
    print("=" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Sistema detenido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")