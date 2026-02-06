import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import xgboost as xgb
import lightgbm as lgb

class ChampionsMatchPredictor:
    """Predictor especializado para UEFA Champions League"""
    
    def __init__(self, models_dir: str = "../models"):
        self.models_dir = models_dir
        self.feature_engineer = None
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
        self.model_performance = {}
        self.champions_weights = {}
        
        # Crear directorio de modelos
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"📁 Creado directorio: {models_dir}")
        
        # Configurar pesos específicos para Champions League
        self._setup_champions_weights()
    
    def _setup_champions_weights(self):
        """Configurar pesos específicos para Champions League"""
        self.champions_weights = {
            # Características UEFA tienen más peso
            'uefa_coefficient_diff': 2.0,
            'home_uefa_coefficient': 1.5,
            'away_uefa_coefficient': 1.5,
            
            # Experiencia europea es clave
            'champions_wins_diff': 1.8,
            'home_champions_wins': 1.5,
            'away_champions_wins': 1.5,
            
            # Factores de viaje internacionales
            'travel_distance_km': 1.3,
            'travel_fatigue_factor': 1.2,
            
            # Presión por etapa
            'pressure_factor': 1.4,
            'stage_importance': 1.3,
            
            # Formato específico
            'is_knockout_match': 1.2,
            'away_goals_advantage': 1.1,
            
            # Head-to-head histórico
            'h2h_home_win_rate': 1.3,
            'h2h_recent_form': 1.2,
            
            # Forma reciente
            'home_last5_win_rate': 1.1,
            'away_last5_win_rate': 1.1
        }
    
    async def train_champions_models_async(self, features_df: pd.DataFrame, targets_df: pd.Series):
        """Entrenar modelos de forma asíncrona para Champions League"""
        print("🏆 ENTRENANDO MODELOS CHAMPIONS LEAGUE...")
        print("=" * 60)
        
        # Guardar columnas de características
        self.feature_columns = features_df.columns.tolist()
        
        # Dividir datos con estratificación por etapa
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, targets_df, test_size=0.2, random_state=42, stratify=targets_df
        )
        
        print(f"📊 Datos de entrenamiento: {len(X_train)} partidos")
        print(f"📊 Datos de prueba: {len(X_test)} partidos")
        print(f"📊 Características: {len(X_train.columns)}")
        
        # Aplicar pesos Champions League
        X_train_weighted = self._apply_champions_weights(X_train)
        X_test_weighted = self._apply_champions_weights(X_test)
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_weighted)
        X_test_scaled = scaler.transform(X_test_weighted)
        
        self.scalers['champions'] = scaler
        
        # Definir modelos optimizados para Champions League
        models_config = {
            'random_forest_champions': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42
                ),
                'params': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [3, 5, 7]
                }
            },
            'xgboost_champions': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.07]
                }
            },
            'lightgbm_champions': {
                'model': lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    class_weight='balanced',
                    verbose=-1
                ),
                'params': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.07]
                }
            },
            'gradient_boosting_champions': {
                'model': GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42
                ),
                'params': {
                    'n_estimators': [150, 200, 250],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.07]
                }
            },
            'logistic_regression_champions': {
                'model': LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=2000,
                    C=0.5
                ),
                'params': {
                    'C': [0.1, 0.5, 1.0]
                }
            }
        }
        
        # Entrenar modelos en paralelo
        training_tasks = []
        for model_name, config in models_config.items():
            task = self._train_single_model_async(
                model_name, config['model'], X_train_scaled, y_train, 
                X_test_scaled, y_test, config['params']
            )
            training_tasks.append(task)
        
        # Esperar a que todos los modelos se entrenen
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Procesar resultados
        successful_models = 0
        for i, result in enumerate(results):
            model_name = list(models_config.keys())[i]
            if isinstance(result, Exception):
                print(f"❌ Error entrenando {model_name}: {result}")
            else:
                successful_models += 1
                print(f"✅ {model_name} entrenado exitosamente")
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO: {successful_models}/{len(models_config)} modelos")
        
        # Guardar modelos
        await self._save_models_async()
        
        # Mostrar comparación
        self._show_model_comparison()
        
        return self.model_performance
    
    async def _train_single_model_async(self, model_name: str, model, X_train, y_train, X_test, y_test, param_grid):
        """Entrenar un modelo específico de forma asíncrona"""
        try:
            print(f"🔄 Entrenando {model_name}...")
            
            # Simular procesamiento asíncrono
            await asyncio.sleep(0.1)
            
            # Optimización de hiperparámetros (opcional)
            if param_grid and len(param_grid) > 1:
                grid_search = GridSearchCV(
                    model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                print(f"   🎯 Mejores params: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                best_model = model
            
            # Evaluar modelo
            train_pred = best_model.predict(X_train)
            test_pred = best_model.predict(X_test)
            
            train_accuracy = accuracy_score(y_train, train_pred)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Validación cruzada
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
            
            # Guardar modelo y métricas
            self.models[model_name] = best_model
            self.model_performance[model_name] = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': self._get_feature_importance(best_model, model_name)
            }
            
            print(f"   ✅ Train Acc: {train_accuracy:.3f}")
            print(f"   ✅ Test Acc: {test_accuracy:.3f}")
            print(f"   ✅ CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en {model_name}: {e}")
            return False
    
    def _apply_champions_weights(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplicar pesos específicos de Champions League"""
        X_weighted = X.copy()
        
        for feature, weight in self.champions_weights.items():
            if feature in X_weighted.columns:
                X_weighted[feature] = X_weighted[feature] * weight
        
        return X_weighted
    
    def _get_feature_importance(self, model, model_name: str) -> Dict:
        """Obtener importancia de características"""
        try:
            if hasattr(model, 'feature_importances_'):
                # Para modelos basados en árboles
                importance_dict = dict(zip(self.feature_columns, model.feature_importances_))
                # Ordenar por importancia
                return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):
                # Para modelos lineales
                coef_dict = dict(zip(self.feature_columns, np.abs(model.coef_[0])))
                return dict(sorted(coef_dict.items(), key=lambda x: x[1], reverse=True))
            else:
                return {}
        except Exception as e:
            print(f"Error obteniendo feature importance: {e}")
            return {}
    
    async def _save_models_async(self):
        """Guardar modelos de forma asíncrona"""
        print("💾 GUARDANDO MODELOS...")
        
        save_tasks = []
        for model_name, model in self.models.items():
            task = self._save_single_model_async(model_name, model)
            save_tasks.append(task)
        
        results = await asyncio.gather(*save_tasks, return_exceptions=True)
        
        successful_saves = sum(1 for r in results if not isinstance(r, Exception))
        print(f"✅ {successful_saves}/{len(self.models)} modelos guardados")
    
    async def _save_single_model_async(self, model_name: str, model):
        """Guardar un modelo específico de forma asíncrona"""
        try:
            await asyncio.sleep(0.01)  # Simular I/O
            
            # Guardar modelo
            model_path = f"{self.models_dir}/{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return True
        except Exception as e:
            print(f"❌ Error guardando {model_name}: {e}")
            return False
    
    def _show_model_comparison(self):
        """Mostrar comparación de modelos"""
        print("\n📊 COMPARACIÓN DE MODELOS CHAMPIONS:")
        print("=" * 80)
        print(f"{'Modelo':<25} {'Train Acc':<12} {'Test Acc':<12} {'CV Mean':<12} {'CV Std':<12}")
        print("-" * 80)
        
        # Ordenar modelos por accuracy de prueba
        sorted_models = sorted(
            self.model_performance.items(),
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        for model_name, performance in sorted_models:
            print(f"{model_name:<25} {performance['train_accuracy']:<12.3f} "
                  f"{performance['test_accuracy']:<12.3f} {performance['cv_mean']:<12.3f} "
                  f"±{performance['cv_std']:<11.3f}")
        
        # Mejor modelo
        best_model = sorted_models[0]
        print(f"\n🏆 Mejor modelo Champions: {best_model[0]}")
        print(f"📊 Accuracy: {best_model[1]['test_accuracy']:.3f}")
        
        # Top features del mejor modelo
        if best_model[1]['feature_importance']:
            print(f"\n🎯 Top 5 Features más importantes ({best_model[0]}):")
            top_features = list(best_model[1]['feature_importance'].items())[:5]
            for feature, importance in top_features:
                print(f"   {feature}: {importance:.4f}")
    
    def load_models(self):
        """Cargar modelos entrenados"""
        print("📂 CARGANDO MODELOS CHAMPIONS...")
        
        models_files = {
            'random_forest_champions': 'random_forest_champions.pkl',
            'xgboost_champions': 'xgboost_champions.pkl',
            'lightgbm_champions': 'lightgbm_champions.pkl',
            'gradient_boosting_champions': 'gradient_boosting_champions.pkl',
            'logistic_regression_champions': 'logistic_regression_champions.pkl'
        }
        
        loaded_models = 0
        
        for model_name, filename in models_files.items():
            model_path = f"{self.models_dir}/{filename}"
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    loaded_models += 1
                    print(f"✅ {model_name} cargado")
                except Exception as e:
                    print(f"❌ Error cargando {model_name}: {e}")
            else:
                print(f"⚠️ No encontrado: {model_path}")
        
        # Cargar scaler
        scaler_path = f"{self.models_dir}/champions_scaler.pkl"
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers['champions'] = pickle.load(f)
                print("✅ Scaler cargado")
            except Exception as e:
                print(f"❌ Error cargando scaler: {e}")
        
        print(f"📊 Modelos cargados: {loaded_models}/{len(models_files)}")
        return loaded_models > 0
    
    async def predict_match_async(self, match_features: pd.Series, model_name: str = None) -> Dict:
        """Predecir resultado de un partido de forma asíncrona"""
        try:
            # Seleccionar mejor modelo si no se especifica
            if model_name is None:
                if not self.model_performance:
                    # Cargar performance si no está disponible
                    self._load_model_performance()
                
                best_model = max(
                    self.model_performance.items(),
                    key=lambda x: x[1]['test_accuracy']
                )[0]
                model_name = best_model
            
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not available")
            
            model = self.models[model_name]
            
            # Aplicar pesos Champions League
            weighted_features = self._apply_champions_weights(pd.DataFrame([match_features]))
            
            # Escalar características
            if 'champions' in self.scalers:
                scaled_features = self.scalers['champions'].transform(weighted_features)
            else:
                scaled_features = weighted_features.values
            
            # Simular procesamiento asíncrono
            await asyncio.sleep(0.01)
            
            # Predecir
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            
            # Mapear predicción
            prediction_map = {0: 'HOME_WIN', 1: 'DRAW', 2: 'AWAY_WIN'}
            result = prediction_map.get(prediction, 'UNKNOWN')
            
            # Obtener clases
            if hasattr(model, 'classes_'):
                classes = model.classes_
            else:
                classes = [0, 1, 2]
            
            # Crear diccionario de probabilidades
            prob_dict = {}
            for i, cls in enumerate(classes):
                if i < len(probabilities):
                    pred_map = {0: 'HOME_WIN', 1: 'DRAW', 2: 'AWAY_WIN'}
                    prob_dict[pred_map.get(cls, f'CLASS_{cls}')] = probabilities[i]
            
            # Calcular confianza
            confidence = max(probabilities)
            
            return {
                'prediction': result,
                'confidence': confidence,
                'probabilities': prob_dict,
                'model_used': model_name,
                'home_probability': prob_dict.get('HOME_WIN', 0),
                'draw_probability': prob_dict.get('DRAW', 0),
                'away_probability': prob_dict.get('AWAY_WIN', 0),
                'champions_factors': {
                    'uefa_impact': self._calculate_uefa_impact(match_features),
                    'travel_impact': self._calculate_travel_impact(match_features),
                    'pressure_impact': self._calculate_pressure_impact(match_features)
                }
            }
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'HOME_WIN': 0, 'DRAW': 0, 'AWAY_WIN': 0},
                'error': str(e)
            }
    
    def _calculate_uefa_impact(self, features: pd.Series) -> float:
        """Calcular impacto de coeficientes UEFA"""
        if 'uefa_coefficient_diff' in features:
            return abs(features['uefa_coefficient_diff']) * 0.01
        return 0.0
    
    def _calculate_travel_impact(self, features: pd.Series) -> float:
        """Calcular impacto de viaje"""
        if 'travel_fatigue_factor' in features:
            return (features['travel_fatigue_factor'] - 1.0) * 0.5
        return 0.0
    
    def _calculate_pressure_impact(self, features: pd.Series) -> float:
        """Calcular impacto de presión"""
        if 'pressure_factor' in features:
            return features['pressure_factor'] * 2.0
        return 0.0
    
    def _load_model_performance(self):
        """Cargar performance de modelos desde archivo"""
        try:
            performance_path = f"{self.models_dir}/champions_model_performance.json"
            if os.path.exists(performance_path):
                import json
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
        except Exception as e:
            print(f"Error cargando performance: {e}")
    
    def get_model_info(self) -> Dict:
        """Obtener información de modelos disponibles"""
        return {
            'available_models': list(self.models.keys()),
            'total_models': len(self.models),
            'feature_columns': self.feature_columns,
            'champions_weights': len(self.champions_weights),
            'model_performance': self.model_performance
        }

# Función principal para testing
async def main():
    """Función principal para testing de modelos"""
    predictor = ChampionsMatchPredictor()
    
    print("🏆 CHAMPIONS LEAGUE MATCH PREDICTOR")
    print("=" * 50)
    
    # Cargar características
    try:
        features_df = pd.read_csv("../data/processed/champions_features_2024.csv")
        
        # Crear targets (resultado del partido)
        def get_result(row):
            if pd.isna(row['home_score']) or pd.isna(row['away_score']):
                return 1  # DRAW por defecto para partidos no jugados
            elif row['home_score'] > row['away_score']:
                return 0  # HOME_WIN
            elif row['home_score'] < row['away_score']:
                return 2  # AWAY_WIN
            else:
                return 1  # DRAW
        
        targets = features_df.apply(get_result, axis=1)
        
        # Seleccionar solo características numéricas para entrenamiento
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_features if col not in ['home_score', 'away_score', 'match_id']]
        
        X = features_df[feature_columns]
        y = targets
        
        print(f"📊 Dataset: {len(X)} partidos, {len(X.columns)} características")
        
        # Entrenar modelos
        performance = await predictor.train_champions_models_async(X, y)
        
        # Probar predicción
        if len(predictor.models) > 0:
            test_match = X.iloc[0]
            prediction = await predictor.predict_match_async(test_match)
            
            print(f"\n🎯 EJEMPLO DE PREDICCIÓN:")
            print(f"   Partido: {features_df.iloc[0]['home_team']} vs {features_df.iloc[0]['away_team']}")
            print(f"   Predicción: {prediction['prediction']}")
            print(f"   Confianza: {prediction['confidence']:.3f}")
            print(f"   Probabilidades: Local={prediction['home_probability']:.3f}, "
                  f"Empate={prediction['draw_probability']:.3f}, Visitante={prediction['away_probability']:.3f}")
        
        print(f"\n🎉 MODELOS CHAMPIONS LEAGUE COMPLETADOS")
        
    except FileNotFoundError:
        print("❌ No se encontraron las características. Ejecuta primero el feature engineering.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())