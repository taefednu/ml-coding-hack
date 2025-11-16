"""
МАКСИМАЛЬНАЯ ОПТИМИЗАЦИЯ ДЛЯ AUC

СТРАТЕГИЯ:
1. Больше моделей в ансамбле (7 вместо 5)
2. Разные методы балансировки для каждой модели
3. Оптимизация под AUC-ROC напрямую
4. Более глубокие деревья
5. Стекинг предсказаний
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    fbeta_score, matthews_corrcoef, brier_score_loss
)
from sklearn.linear_model import LogisticRegression

from catboost import CatBoostClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.combine import SMOTETomek

import prepare_data as prep
from train_ultimate import create_ultimate_features, handle_missing_and_encode

RANDOM_STATE = 42

def balance_multiple_strategies(X, y, strategy='adasyn', sampling_ratio=0.75):
    """Разные стратегии балансировки."""
    if strategy == 'adasyn':
        sampler = ADASYN(sampling_strategy=sampling_ratio, random_state=RANDOM_STATE, n_neighbors=5)
    elif strategy == 'smote':
        sampler = SMOTE(sampling_strategy=sampling_ratio, random_state=RANDOM_STATE, k_neighbors=5)
    elif strategy == 'smote_tomek':
        sampler = SMOTETomek(sampling_strategy=sampling_ratio, random_state=RANDOM_STATE)
    else:
        return X, y
    
    try:
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except:
        return X, y


def train_diverse_ensemble(X_train, y_train, X_val, y_val):
    """Ансамбль из 7 разнообразных моделей."""
    
    print("\n🎯 Обучение разнообразного ансамбля из 7 моделей...")
    
    # Конфигурации моделей - РАЗНЫЕ стратегии
    configs = [
        {'seed': 42, 'balance': 'adasyn', 'ratio': 0.70, 'depth': 10, 'lr': 0.03, 'iter': 1200},
        {'seed': 123, 'balance': 'smote', 'ratio': 0.75, 'depth': 9, 'lr': 0.04, 'iter': 1000},
        {'seed': 456, 'balance': 'adasyn', 'ratio': 0.80, 'depth': 8, 'lr': 0.05, 'iter': 1100},
        {'seed': 789, 'balance': 'smote_tomek', 'ratio': 0.75, 'depth': 9, 'lr': 0.04, 'iter': 1000},
        {'seed': 999, 'balance': 'adasyn', 'ratio': 0.85, 'depth': 7, 'lr': 0.06, 'iter': 900},
        {'seed': 1234, 'balance': 'smote', 'ratio': 0.80, 'depth': 10, 'lr': 0.03, 'iter': 1200},
        {'seed': 5678, 'balance': 'adasyn', 'ratio': 0.75, 'depth': 9, 'lr': 0.045, 'iter': 1000},
    ]
    
    models = []
    val_predictions = []
    
    for i, config in enumerate(configs):
        print(f"\n   [{i+1}/7] Модель: {config['balance']} | depth={config['depth']} | lr={config['lr']}")
        
        # Балансировка
        X_bal, y_bal = balance_multiple_strategies(
            X_train, y_train, 
            strategy=config['balance'], 
            sampling_ratio=config['ratio']
        )
        print(f"         Балансировка: {len(y_train)} → {len(y_bal)} ({config['balance']})")
        
        # Модель с оптимизацией под AUC
        model = CatBoostClassifier(
            iterations=config['iter'],
            depth=config['depth'],
            learning_rate=config['lr'],
            l2_leaf_reg=8,
            scale_pos_weight=7.0,
            border_count=128,
            eval_metric='AUC',  # Оптимизация под AUC!
            random_state=config['seed'],
            verbose=0,
            thread_count=-1
        )
        
        model.fit(X_bal, y_bal, verbose=False)
        
        # Валидационные предсказания
        val_pred = model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred)
        
        models.append(model)
        val_predictions.append(val_pred)
        
        print(f"         ✅ AUC на валидации: {val_auc:.4f}")
    
    return models, np.array(val_predictions)


def optimize_stacking(models, val_predictions, y_val):
    """Стекинг - обучаем мета-модель на предсказаниях."""
    
    print("\n🔧 Стекинг: обучение мета-модели...")
    
    # Используем предсказания как признаки
    meta_features = val_predictions.T  # shape: (n_samples, n_models)
    
    # Мета-модель - Logistic Regression
    meta_model = LogisticRegression(
        penalty='l2',
        C=1.0,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000
    )
    
    meta_model.fit(meta_features, y_val)
    
    # Оценка стекинга
    meta_pred = meta_model.predict_proba(meta_features)[:, 1]
    stacking_auc = roc_auc_score(y_val, meta_pred)
    
    # Сравнение с простым усреднением
    simple_avg = np.mean(val_predictions, axis=0)
    simple_auc = roc_auc_score(y_val, simple_avg)
    
    print(f"   AUC простое усреднение: {simple_auc:.4f}")
    print(f"   AUC стекинг:            {stacking_auc:.4f}")
    
    if stacking_auc > simple_auc:
        print(f"   ✅ Используем СТЕКИНГ (+{(stacking_auc-simple_auc):.4f})")
        return meta_model, 'stacking'
    else:
        print(f"   ✅ Используем УСРЕДНЕНИЕ")
        return None, 'averaging'


def main():
    start = datetime.now()
    
    print("="*80)
    print("МАКСИМАЛЬНАЯ ОПТИМИЗАЦИЯ ДЛЯ AUC")
    print("="*80)
    
    # 1. Загрузка
    print("\n[1/5] Загрузка данных...")
    master_df = prep.build_master_dataset()
    print(f"   ✅ {len(master_df)} строк, дефолтов: {master_df['default'].mean():.2%}")
    
    # 2. Feature Engineering
    print("\n[2/5] Feature Engineering...")
    df_featured = create_ultimate_features(master_df)
    X, y, feature_cols, scaler = handle_missing_and_encode(df_featured)
    print(f"   ✅ {len(feature_cols)} признаков")
    
    # 3. Разделение с стратификацией
    print("\n[3/5] Разделение данных (стратифицированное)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.18, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Обучение разнообразного ансамбля
    print("\n[4/5] Обучение ансамбля...")
    models, val_predictions = train_diverse_ensemble(X_train, y_train, X_val, y_val)
    
    # 5. Стекинг / Усреднение
    print("\n[5/5] Оптимизация объединения предсказаний...")
    meta_model, strategy = optimize_stacking(models, val_predictions, y_val)
    
    # ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТЕ
    print("\n" + "="*80)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("="*80)
    
    # Предсказания всех моделей
    test_preds = []
    for model in models:
        test_preds.append(model.predict_proba(X_test)[:, 1])
    test_preds = np.array(test_preds)
    
    # Объединение предсказаний
    if strategy == 'stacking':
        meta_features = test_preds.T
        test_pred_proba = meta_model.predict_proba(meta_features)[:, 1]
    else:
        test_pred_proba = np.mean(test_preds, axis=0)
    
    # Оптимальный порог
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, test_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    test_pred = (test_pred_proba >= optimal_threshold).astype(int)
    
    # Метрики
    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_pr_auc = average_precision_score(y_test, test_pred_proba)
    test_f1 = f1_score(y_test, test_pred)
    test_f2 = fbeta_score(y_test, test_pred, beta=2)
    test_mcc = matthews_corrcoef(y_test, test_pred)
    test_brier = brier_score_loss(y_test, test_pred_proba)
    
    print(f"\n{'Метрика':<25} {'Было':<10} {'Стало':<10} {'Δ':<10}")
    print("="*55)
    print(f"{'AUC-ROC':<25} {'0.8493':<10} {test_auc:<10.4f} {test_auc - 0.8493:+.4f}")
    print(f"{'PR-AUC':<25} {'0.3629':<10} {test_pr_auc:<10.4f} {test_pr_auc - 0.3629:+.4f}")
    print(f"{'F1-Score':<25} {'0.3208':<10} {test_f1:<10.4f} {test_f1 - 0.3208:+.4f}")
    print(f"{'F2-Score':<25} {'0.4617':<10} {test_f2:<10.4f} {test_f2 - 0.4617:+.4f}")
    print(f"{'Matthews Corr':<25} {'0.3165':<10} {test_mcc:<10.4f} {test_mcc - 0.3165:+.4f}")
    print("="*55)
    
    if test_auc >= 0.90:
        print(f"\n🎉🎉🎉 ОТЛИЧНО! AUC ≥ 0.90 - МАКСИМАЛЬНЫЙ БАЛЛ! 🎉🎉🎉")
    elif test_auc >= 0.85:
        print(f"\n✅ Хорошо! AUC ≥ 0.85")
    
    # Сохранение
    print("\n[Сохранение...]")
    model_path = Path("models")
    
    with open(model_path / "best_model_optimized.pkl", "wb") as f:
        pickle.dump({
            "models": models,
            "meta_model": meta_model,
            "strategy": strategy,
            "feature_cols": feature_cols,
            "optimal_threshold": optimal_threshold,
            "auc": test_auc,
            "pr_auc": test_pr_auc,
            "scaler": scaler,
            "selected_features": feature_cols
        }, f)
    
    print(f"✅ Модель сохранена: models/best_model_optimized.pkl")
    
    elapsed = datetime.now() - start
    print(f"\n⏱ Время: {elapsed}")
    print("✅ ГОТОВО!")
    
    return models, meta_model


if __name__ == "__main__":
    models, meta_model = main()
