#!/usr/bin/env python3
"""
Training script for Champions League predictions.
Step 5: Train baseline multiclass model (HomeWin, Draw, AwayWin)
"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_COLS = [
    'home_last5_w', 'home_last5_d', 'home_last5_l', 'home_last5_pts',
    'away_last5_w', 'away_last5_d', 'away_last5_l', 'away_last5_pts',
    'home_last5_all_pts', 'away_last5_all_pts',
    'home_goals_for', 'home_goals_against',
    'away_goals_for', 'away_goals_against',
    'h2h_home_w', 'h2h_away_w', 'h2h_draw',
    'home_days_rest', 'away_days_rest',
    'home_ranking_pts', 'away_ranking_pts', 'ranking_diff',
    'home_uefa_coef', 'away_uefa_coef', 'uefa_coef_diff',
    'is_knockout', 'matchday',
    # New knockout features
    'phase_type', 'home_knockout_exp', 'away_knockout_exp',
    'home_best_round', 'away_best_round',
    'h2h_champions_w', 'h2h_champions_away_w', 'h2h_champions_draw'
]

LABEL_COL = 'result'
LABEL_MAP = {0: 'Draw', 1: 'HomeWin', 2: 'AwayWin'}


def load_training_data(path: str = 'data/training/champions_training_full.csv') -> tuple:
    """Load training dataset and split into features/labels."""
    df = pd.read_csv(path)
    
    X = df[FEATURE_COLS].fillna(0)
    y = df[LABEL_COL].astype(int)
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train RandomForest classifier with scaling."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    print("Training RandomForest model...")
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate model using cross-validation."""
    print("\nEvaluating with 5-fold cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accuracy_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_weighted')
    
    results = {
        'accuracy_mean': accuracy_scores.mean(),
        'accuracy_std': accuracy_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std()
    }
    
    print(f"  Accuracy: {results['accuracy_mean']:.3f} (+/- {results['accuracy_std']:.3f})")
    print(f"  F1 Score: {results['f1_mean']:.3f} (+/- {results['f1_std']:.3f})")
    
    return results


def analyze_feature_importance(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Analyze feature importance from the trained model."""
    importances = pipeline.named_steps['classifier'].feature_importances_
    
    df_importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    for i, row in df_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return df_importance


def save_model(pipeline: Pipeline, path: str = 'models/champions_model.pkl') -> None:
    """Save trained model to file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\nModel saved to {path}")


def main():
    print("=" * 50)
    print("CHAMPIONS LEAGUE PREDICTION MODEL - TRAINING")
    print("=" * 50)
    
    X, y = load_training_data()
    print(f"\nDataset: {len(X)} samples, {len(FEATURE_COLS)} features")
    print(f"Class distribution:")
    for label, name in LABEL_MAP.items():
        count = (y == label).sum()
        pct = count / len(y) * 100
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    pipeline = train_model(X, y)
    
    results = evaluate_model(pipeline, X, y)
    
    importance_df = analyze_feature_importance(pipeline, X)
    
    save_model(pipeline)
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()
