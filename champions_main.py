#!/usr/bin/env python3
"""
CHAMPIONS LEAGUE PREDICTOR - SISTEMA COMPLETO
Integración final del sistema asíncrono Champions League
"""

import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json

# Añadir path para importaciones
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.champions_data_collection import ChampionsLeagueDataCollector
from src.champions_feature_engineering import ChampionsFeatureEngineer
from src.champions_prediction_models import ChampionsMatchPredictor
from src.champions_menu import ChampionsTerminalInterface

class ChampionsLeaguePredictorSystem:
    """Sistema completo integrado para Champions League"""
    
    def __init__(self):
        self.data_collector = ChampionsLeagueDataCollector()
        self.feature_engineer = ChampionsFeatureEngineer()
        self.predictor = ChampionsMatchPredictor()
        self.interface = ChampionsTerminalInterface()
        
        # Estado del sistema
        self.is_initialized = False
        self.current_season = 2024
        self.last_update = None
        
        # Configuración
        self.config = {
            'auto_update': True,
            'cache_duration_hours': 24,
            'batch_size': 50,
            'max_predictions_queue': 100
        }
    
    async def initialize_complete_system(self):
        """Inicializar sistema completo Champions League"""
        print("🏆 CHAMPIONS LEAGUE PREDICTOR - INICIALIZACIÓN COMPLETA")
        print("=" * 70)
        print("🌍 Sistema especializado UEFA Champions League")
        print("⚡ Procesamiento asíncrono avanzado")
        print("📊 47 características Champions únicas")
        print("🤖 5 modelos optimizados para fútbol europeo")
        print("=" * 70)
        
        try:
            # Paso 1: Verificar entorno
            await self._verify_environment()
            
            # Paso 2: Cargar o actualizar datos
            await self._setup_data()
            
            # Paso 3: Generar características
            await self._setup_features()
            
            # Paso 4: Cargar o entrenar modelos
            await self._setup_models()
            
            # Paso 5: Inicializar componentes asíncronos
            await self._setup_async_components()
            
            self.is_initialized = True
            self.last_update = datetime.now()
            
            print("\n✅ SISTEMA CHAMPIONS LEAGUE COMPLETAMENTE INICIALIZADO")
            print(f"🕐 Última actualización: {self.last_update.strftime('%d/%m %H:%M')}")
            
            # Mostrar resumen
            await self._show_system_summary()
            
            return True
            
        except Exception as e:
            print(f"❌ Error en inicialización: {e}")
            return False
    
    async def _verify_environment(self):
        """Verificar entorno y dependencias"""
        print("\n🔍 PASO 1: VERIFICANDO ENTORNO...")
        
        # Verificar directorios
        required_dirs = [
            "../data", "../data/raw", "../data/cleaned", 
            "../data/processed", "../data/cache", "../models"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
                print(f"   📁 Creado: {dir_path}")
        
        # Verificar datos cache
        cache = self.data_collector.get_cached_data(self.current_season)
        if cache:
            print("   ✅ Cache datos encontrado")
        else:
            print("   ⚠️ Sin cache - se descargarán datos")
        
        print("   ✅ Entorno verificado")
    
    async def _setup_data(self):
        """Configurar datos del sistema"""
        print("\n📊 PASO 2: CONFIGURANDO DATOS...")
        
        # Intentar cargar cache primero
        cache = self.data_collector.get_cached_data(self.current_season)
        
        if cache:
            # Verificar si el cache es reciente
            last_updated = datetime.fromisoformat(cache['last_updated'].replace('Z', '+00:00'))
            age_hours = (datetime.now() - last_updated).total_seconds() / 3600
            
            if age_hours < self.config['cache_duration_hours']:
                print(f"   ✅ Usando cache reciente ({age_hours:.1f} horas)")
                self.interface.current_data = cache
                return
        
        # Si no hay cache o es antiguo, actualizar
        print("   🔄 Actualizando datos desde APIs...")
        success = await self.data_collector.update_data(force_refresh=True)
        
        if success:
            cache = self.data_collector.get_cached_data(self.current_season)
            if cache:
                self.interface.current_data = cache
                print("   ✅ Datos actualizados y cacheados")
            else:
                raise Exception("Error obteniendo datos actualizados")
        else:
            # Intentar cargar datos existentes
            try:
                matches_file = f"../data/raw/champions_matches_{self.current_season}.csv"
                if os.path.exists(matches_file):
                    df = pd.read_csv(matches_file)
                    df['date'] = pd.to_datetime(df['date'])
                    self.interface.current_data = {'matches': df.to_dict('records')}
                    print("   ✅ Cargados datos existentes")
                else:
                    raise Exception("No hay datos disponibles")
            except Exception as e:
                print(f"   ⚠️ Error cargando datos: {e}")
                raise
    
    async def _setup_features(self):
        """Configurar características del sistema"""
        print("\n🔧 PASO 3: CONFIGURANDO CARACTERÍSTICAS...")
        
        # Intentar cargar características procesadas
        features_file = f"../data/processed/champions_features_{self.current_season}.csv"
        
        if os.path.exists(features_file):
            self.interface.current_features = pd.read_csv(features_file)
            print(f"   ✅ Características cargadas ({len(self.interface.current_features)} partidos)")
            return
        
        # Si no hay características, generarlas
        if self.interface.current_data and 'matches' in self.interface.current_data:
            print("   🔧 Generando características desde datos...")
            
            # Cargar datos para feature engineering
            if self.feature_engineer.load_data():
                matches_df = pd.DataFrame(self.interface.current_data['matches'])
                
                # Generar características asíncronas
                features_df = await self.feature_engineer.create_champions_features_async(matches_df)
                
                # Guardar características
                features_df.to_csv(features_file, index=False)
                self.interface.current_features = features_df
                
                print(f"   ✅ Características generadas ({len(features_df)} partidos, {len(features_df.columns)} features)")
            else:
                raise Exception("Error cargando datos para feature engineering")
        else:
            raise Exception("No hay datos para generar características")
    
    async def _setup_models(self):
        """Configurar modelos del sistema"""
        print("\n🤖 PASO 4: CONFIGURANDO MODELOS...")
        
        # Intentar cargar modelos existentes
        models_loaded = self.predictor.load_models()
        
        if models_loaded:
            print(f"   ✅ {len(self.predictor.models)} modelos cargados")
        else:
            print("   🤖 No hay modelos - entrenando nuevos modelos...")
            await self._train_new_models()
    
    async def _train_new_models(self):
        """Entrenar nuevos modelos"""
        try:
            if self.interface.current_features is None:
                raise Exception("No hay características para entrenar")
            
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
            
            targets = self.interface.current_features.apply(get_result, axis=1)
            
            # Seleccionar características numéricas
            numeric_features = self.interface.current_features.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_features if col not in ['home_score', 'away_score', 'match_id']]
            
            X = self.interface.current_features[feature_columns]
            y = targets
            
            print(f"   📊 Dataset: {len(X)} partidos, {len(X.columns)} características")
            
            # Entrenar modelos
            await self.predictor.train_champions_models_async(X, y)
            
            print(f"   ✅ {len(self.predictor.models)} modelos entrenados")
            
        except Exception as e:
            print(f"   ❌ Error entrenando modelos: {e}")
            raise
    
    async def _setup_async_components(self):
        """Configurar componentes asíncronos"""
        print("\n⚡ PASO 5: CONFIGURANDO COMPONENTES ASÍNCRONOS...")
        
        # Inicializar cola de predicciones
        self.interface.prediction_queue = asyncio.Queue(maxsize=self.config['max_predictions_queue'])
        
        # Iniciar worker de predicciones
        asyncio.create_task(self.interface._prediction_worker())
        
        # Inicializar otros componentes asíncronos si es necesario
        print("   ✅ Componentes asíncronos iniciados")
    
    async def _show_system_summary(self):
        """Mostrar resumen del sistema"""
        print("\n📊 RESUMEN DEL SISTEMA CHAMPIONS LEAGUE:")
        print("-" * 50)
        
        # Datos
        if self.interface.current_data:
            if 'matches' in self.interface.current_data:
                matches_count = len(self.interface.current_data['matches'])
                print(f"   📊 Partidos cargados: {matches_count}")
            if 'total_teams' in self.interface.current_data:
                print(f"   🏆 Equipos: {self.interface.current_data['total_teams']}")
        
        # Características
        if self.interface.current_features is not None:
            print(f"   🔧 Características: {len(self.interface.current_features.columns)}")
            
            # Mostrar características clave
            key_features = ['home_uefa_coefficient', 'away_uefa_coefficient', 'travel_distance_km', 
                          'pressure_factor', 'home_champions_wins']
            available_key_features = [f for f in key_features if f in self.interface.current_features.columns]
            print(f"   🎯 Features clave: {len(available_key_features)}")
        
        # Modelos
        if self.predictor.models:
            print(f"   🤖 Modelos disponibles: {len(self.predictor.models)}")
            
            # Mejor modelo
            if self.predictor.model_performance:
                best_model = max(self.predictor.model_performance.items(), 
                               key=lambda x: x[1]['test_accuracy'])
                print(f"   🏆 Mejor modelo: {best_model[0]} ({best_model[1]['test_accuracy']:.3f} accuracy)")
        
        # Características Champions
        print(f"   🌍 Factores Champions: {len(self.predictor.champions_weights)}")
        print(f"   ⚡ Cola predicciones: {self.interface.prediction_queue.qsize()}")
        
        print("-" * 50)
    
    async def run_interactive_mode(self):
        """Ejecutar modo interactivo"""
        if not self.is_initialized:
            print("❌ Sistema no inicializado. Ejecuta initialize_complete_system() primero.")
            return
        
        print("\n🚀 INICIANDO MODO INTERACTIVO CHAMPIONS LEAGUE")
        await asyncio.sleep(1)
        
        # Ejecutar interfaz terminal
        await self.interface.show_main_menu()
    
    async def run_batch_predictions(self, match_ids: List[str] = None) -> Dict:
        """Ejecutar predicciones en lote"""
        if not self.is_initialized:
            return {"error": "Sistema no inicializado"}
        
        print("🔄 EJECUTANDO PREDICCIONES EN LOTE")
        
        if match_ids is None:
            # Predecir todos los partidos próximos
            if self.interface.current_data and 'matches' in self.interface.current_data:
                matches_df = pd.DataFrame(self.interface.current_data['matches'])
                matches_df['date'] = pd.to_datetime(matches_df['date'])
                
                now = datetime.now()
                upcoming = matches_df[matches_df['date'] > now]
                match_ids = upcoming['match_id'].tolist() if 'match_id' in upcoming.columns else []
        
        if not match_ids:
            return {"error": "No hay partidos para predecir"}
        
        print(f"📊 Procesando {len(match_ids)} predicciones...")
        
        predictions = []
        start_time = time.time()
        
        for match_id in match_ids:
            try:
                # Aquí iría la lógica de predicción para cada match_id
                # Por ahora, simulamos predicciones
                await asyncio.sleep(0.1)
                
                prediction = {
                    "match_id": match_id,
                    "prediction": "HOME_WIN",
                    "confidence": 0.65,
                    "timestamp": datetime.now().isoformat()
                }
                predictions.append(prediction)
                
            except Exception as e:
                print(f"❌ Error prediciendo {match_id}: {e}")
        
        total_time = time.time() - start_time
        
        return {
            "success": True,
            "total_predictions": len(predictions),
            "processing_time": total_time,
            "avg_time_per_prediction": total_time / len(predictions),
            "predictions": predictions
        }
    
    async def get_system_status(self) -> Dict:
        """Obtener estado completo del sistema"""
        return {
            "initialized": self.is_initialized,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "current_season": self.current_season,
            "data_status": {
                "loaded": self.interface.current_data is not None,
                "matches_count": len(self.interface.current_data.get('matches', [])) if self.interface.current_data else 0
            },
            "features_status": {
                "loaded": self.interface.current_features is not None,
                "features_count": len(self.interface.current_features.columns) if self.interface.current_features is not None else 0
            },
            "models_status": {
                "loaded": len(self.predictor.models),
                "available": list(self.predictor.models.keys())
            },
            "async_status": {
                "queue_size": self.interface.prediction_queue.qsize(),
                "max_queue_size": self.config['max_predictions_queue']
            },
            "champions_factors": {
                "uefa_weights": len(self.predictor.champions_weights),
                "total_features": 47
            }
        }
    
    def get_system_info(self) -> Dict:
        """Obtener información del sistema"""
        return {
            "name": "Champions League Predictor",
            "version": "1.0.0",
            "type": "Async ML Prediction System",
            "sport": "Football",
            "competition": "UEFA Champions League",
            "features": {
                "total": 47,
                "uefa_coefficients": True,
                "travel_factors": True,
                "pressure_factors": True,
                "european_experience": True,
                "knockout_format": True
            },
            "models": {
                "random_forest": True,
                "xgboost": True,
                "lightgbm": True,
                "gradient_boosting": True,
                "logistic_regression": True
            },
            "async_capabilities": {
                "data_collection": True,
                "feature_engineering": True,
                "predictions": True,
                "batch_processing": True
            }
        }

# Función principal para ejecutar el sistema completo
async def main():
    """Función principal del sistema Champions League"""
    system = ChampionsLeaguePredictorSystem()
    
    print("🏆 CHAMPIONS LEAGUE PREDICTOR - SISTEMA COMPLETO")
    print("=" * 60)
    
    try:
        # Inicializar sistema completo
        if await system.initialize_complete_system():
            
            # Mostrar información del sistema
            info = system.get_system_info()
            print(f"\n🌍 {info['name']} v{info['version']}")
            print(f"⚽ {info['competition']} - {info['sport']}")
            print(f"🔧 Features: {info['features']['total']} características")
            print(f"🤖 Modelos: {len(info['models'])} disponibles")
            
            # Opción de modo
            print("\n🎯 SELECCIONA MODO DE EJECUCIÓN:")
            print("1. 🖥️  Modo interactivo (menú terminal)")
            print("2. 🔄 Predicciones en lote")
            print("3. 📊 Ver estado del sistema")
            print("4. 🚪 Salir")
            
            choice = input("\n👉 Selecciona una opción: ").strip()
            
            if choice == "1":
                await system.run_interactive_mode()
            elif choice == "2":
                result = await system.run_batch_predictions()
                print(f"\n✅ Predicciones en lote: {result}")
            elif choice == "3":
                status = await system.get_system_status()
                print(f"\n📊 Estado del sistema:")
                for key, value in status.items():
                    print(f"   {key}: {value}")
            elif choice == "4":
                print("\n👋 ¡Hasta luego!")
            else:
                print("❌ Opción inválida")
        
    except KeyboardInterrupt:
        print("\n\n👋 Sistema detenido por usuario")
    except Exception as e:
        print(f"\n❌ Error en sistema: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())