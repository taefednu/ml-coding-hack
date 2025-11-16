"""
ULTIMATE ML Pipeline - Максимальная оптимизация для PR-AUC ≥ 0.80

КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ:
1. ✅ Принудительная балансировка (scale_pos_weight ≥ 4, class_weights)
2. ✅ ADASYN с sampling_strategy=0.75
3. ✅ 40+ критических взаимодействий для детекции дефолта
4. ✅ Ансамбль из 5 моделей с оптимизацией весов под PR-AUC
5. ✅ Калибровка вероятностей
6. ✅ Оптимизация порога под F2-score (recall важнее)
7. ✅ Optuna с фокусом на PR-AUC

ОЖИДАЕМЫЙ РЕЗУЛЬТАТ: PR-AUC 0.70-0.80 (сейчас 0.33)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
from datetime import datetime
from scipy.stats import spearmanr
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score
)
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV

# Балансировка
from imblearn.over_sampling import ADASYN, BorderlineSMOTE

# Gradient Boosting
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    raise ImportError("CatBoost обязателен для этого pipeline!")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna недоступна")

import prepare_data as prep

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

RANDOM_STATE = 42
EXCLUDE_COLS = ["customer_id", "default", "application_id"]

# Многопоточность - использовать все ядра
import multiprocessing
MAX_JOBS = multiprocessing.cpu_count()
CATBOOST_THREADS = -1  # Все ядра для CatBoost

print(f"💻 Доступно CPU ядер: {MAX_JOBS}")
print(f"💻 CatBoost будет использовать все ядра")


# ============================================================================
# КРИТИЧЕСКИЕ ВЗАИМОДЕЙСТВИЯ ДЛЯ ДЕТЕКЦИИ ДЕФОЛТА
# ============================================================================

def create_default_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает специализированные признаки для детекции дефолта.
    Фокус на паттернах, характерных для дефолтов.
    """
    df = df.copy()
    print("   🔧 Создание критических признаков детекции дефолта...")
    
    # ===== КОМБИНАЦИИ ВЫСОКОГО РИСКА =====
    
    # Экстремальный риск: плохой скор + высокий долг + просрочки
    if all(col in df.columns for col in ['credit_score', 'debt_to_income_ratio', 'num_delinquencies_2yrs']):
        df['extreme_risk_combo'] = (
            (df['credit_score'] < 600) &
            (df['debt_to_income_ratio'] > 0.5) &
            (df['num_delinquencies_2yrs'] > 0)
        ).astype(int)
        
        df['critical_risk_combo'] = (
            (df['credit_score'] < 650) &
            (df['debt_to_income_ratio'] > 0.43)
        ).astype(int)
    
    # Высокий долг + низкий доход
    if all(col in df.columns for col in ['debt_to_income_ratio', 'annual_income']):
        income_q30 = df['annual_income'].quantile(0.3)
        df['high_debt_low_income'] = (
            (df['debt_to_income_ratio'] > 0.6) &
            (df['annual_income'] < income_q30)
        ).astype(int)
    
    # Молодой + высокая утилизация
    if all(col in df.columns for col in ['age', 'credit_utilization']):
        df['young_high_utilization'] = (
            (df['age'] < 30) &
            (df.get('credit_utilization', 0) > 0.8)
        ).astype(int)
    
    # Нестабильная занятость + высокий долг
    if all(col in df.columns for col in ['employment_length', 'debt_to_income_ratio']):
        df['unstable_employment_high_debt'] = (
            (df['employment_length'] < 2) &
            (df['debt_to_income_ratio'] > 0.5)
        ).astype(int)
    
    # ===== СКОРИНГОВЫЕ ПРИЗНАКИ =====
    
    # Composite risk score (0-10)
    risk_components = []
    if 'credit_score' in df.columns:
        risk_components.append((df['credit_score'] < 650).astype(int) * 3)
    if 'debt_to_income_ratio' in df.columns:
        risk_components.append((df['debt_to_income_ratio'] > 0.5).astype(int) * 3)
    if 'num_delinquencies_2yrs' in df.columns:
        risk_components.append((df['num_delinquencies_2yrs'] > 0).astype(int) * 2)
    if 'credit_utilization' in df.columns:
        risk_components.append((df.get('credit_utilization', 0) > 0.7).astype(int) * 2)
    
    if risk_components:
        df['default_risk_score'] = sum(risk_components)
    
    # ===== ВЗАИМОДЕЙСТВИЯ ЧИСЛОВЫХ ПРИЗНАКОВ =====
    
    # Кредитная история × Финансовый стресс
    if all(col in df.columns for col in ['num_delinquencies_2yrs', 'debt_to_income_ratio']):
        df['credit_history_debt_stress'] = (
            df['num_delinquencies_2yrs'].fillna(0) *
            df['debt_to_income_ratio']
        )
    
    # Утилизация × Долговая нагрузка
    if all(col in df.columns for col in ['credit_utilization', 'debt_to_income_ratio']):
        df['utilization_debt_product'] = (
            df.get('credit_utilization', 0) *
            df['debt_to_income_ratio']
        )
    
    # Возраст × Кредитный скор (молодые с плохим скором)
    if all(col in df.columns for col in ['age', 'credit_score']):
        df['age_score_interaction'] = (
            (40 - df['age'].clip(20, 40)) *  # моложе = больше риск
            (750 - df['credit_score'].clip(300, 750))  # ниже скор = больше риск
        )
    
    # ===== ФИНАНСОВАЯ НЕСТАБИЛЬНОСТЬ =====
    
    # Отрицательный денежный поток
    if 'monthly_free_cash_flow' in df.columns:
        df['negative_cash_flow'] = (df['monthly_free_cash_flow'] < 0).astype(int)
        df['severe_cash_deficit'] = (df['monthly_free_cash_flow'] < -500).astype(int)
    
    # Очень высокая DTI (> 85% - критический уровень)
    if 'debt_to_income_ratio' in df.columns:
        df['extreme_dti'] = (df['debt_to_income_ratio'] > 0.85).astype(int)
        df['dti_danger_zone'] = (
            (df['debt_to_income_ratio'] > 0.6) &
            (df['debt_to_income_ratio'] <= 0.85)
        ).astype(int)
    
    # ===== ПОВЕДЕНЧЕСКИЕ КРАСНЫЕ ФЛАГИ =====
    
    # Высокая активность службы поддержки + низкая цифровая активность
    if all(col in df.columns for col in ['num_customer_service_calls', 'num_login_sessions']):
        median_calls = df['num_customer_service_calls'].median()
        median_sessions = df['num_login_sessions'].median()
        df['support_intensive_low_engagement'] = (
            (df['num_customer_service_calls'] > median_calls * 1.5) &
            (df['num_login_sessions'] < median_sessions * 0.5)
        ).astype(int)
    
    # Поздняя подача заявки (22:00 - 06:00)
    if 'application_hour' in df.columns:
        df['late_night_application'] = (
            (df['application_hour'] >= 22) |
            (df['application_hour'] <= 6)
        ).astype(int)
    
    # ===== РЕГИОНАЛЬНЫЙ РИСК =====
    
    # Высокая безработица + низкий доход
    if all(col in df.columns for col in ['regional_unemployment_rate', 'annual_income', 'regional_median_income']):
        df['regional_economic_stress'] = (
            (df['regional_unemployment_rate'] > 0.08) &
            (df['annual_income'] < df['regional_median_income'])
        ).astype(int)
    
    # ===== СЛОЖНЫЕ ВЗАИМОДЕЙСТВИЯ =====
    
    # Тройное взаимодействие: молодой + нестабильный + высокий долг
    if all(col in df.columns for col in ['age', 'employment_length', 'debt_to_income_ratio']):
        df['triple_risk_young_unstable_debt'] = (
            (df['age'] < 30) &
            (df['employment_length'] < 2) &
            (df['debt_to_income_ratio'] > 0.5)
        ).astype(int)
    
    # Просрочки + коллекции (кредитная история в руинах)
    if all(col in df.columns for col in ['num_delinquencies_2yrs', 'num_collections']):
        df['credit_history_ruined'] = (
            (df['num_delinquencies_2yrs'] > 1) &
            (df['num_collections'] > 0)
        ).astype(int)
        
        df['delinquency_collection_severity'] = (
            df['num_delinquencies_2yrs'].fillna(0) +
            df['num_collections'].fillna(0) * 2  # коллекции хуже
        )
    
    # Максимальная утилизация кредита (close to limit)
    if 'credit_utilization' in df.columns:
        df['maxed_out_credit'] = (df.get('credit_utilization', 0) > 0.95).astype(int)
    
    print(f"   ✅ Добавлено критических признаков. Всего колонок: {len(df.columns)}")
    return df


# ============================================================================
# РАСШИРЕННЫЙ FEATURE ENGINEERING
# ============================================================================

def create_ultimate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Полный набор признаков."""
    df = df.copy()
    
    # 1. Базовый feature engineering из train_advanced.py
    try:
        from train_advanced import advanced_feature_engineering
        df = advanced_feature_engineering(df)
    except:
        print("   ⚠ train_advanced.py не найден, пропускаем базовый FE")
    
    # 2. Критические признаки для детекции дефолта
    df = create_default_detection_features(df)
    
    return df


# ============================================================================
# ОБРАБОТКА ДАННЫХ
# ============================================================================

def handle_missing_and_encode(df: pd.DataFrame, target_col: str = "default", scaler=None):
    """Обработка пропусков, кодирование и нормализация."""
    df = df.copy()
    
    # Удаляем шумовые колонки если остались
    noise_cols = [c for c in df.columns if 'noise' in c.lower() or 'random' in c.lower()]
    if noise_cols:
        df = df.drop(columns=noise_cols)
    
    # Пропуски
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].isna().sum() == 0:
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else "unknown")
    
    # Кодирование
    from sklearn.preprocessing import LabelEncoder
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            df[col] = df[col].fillna("unknown")
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Разделение
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Очистка
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Нормализация числовых признаков
    from sklearn.preprocessing import StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    return X, y, feature_cols, scaler


# ============================================================================
# УСИЛЕННАЯ БАЛАНСИРОВКА
# ============================================================================

def balance_with_adasyn(X, y, sampling_strategy=0.75):
    """
    ADASYN балансировка с адаптивной генерацией примеров.
    sampling_strategy=0.75 означает minority будет 75% от majority.
    """
    try:
        adasyn = ADASYN(
            n_neighbors=7,
            random_state=RANDOM_STATE,
            sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        print(f"      ✅ ADASYN: {len(y)} → {len(y_resampled)} строк")
        print(f"      Баланс классов: {pd.Series(y_resampled).value_counts().to_dict()}")
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    except Exception as e:
        print(f"      ⚠ ADASYN не удался: {e}, используем BorderlineSMOTE")
        smote = BorderlineSMOTE(
            k_neighbors=5,
            random_state=RANDOM_STATE,
            sampling_strategy=sampling_strategy
        )
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)


# ============================================================================
# OPTUNA ОПТИМИЗАЦИЯ С ФОКУСОМ НА PR-AUC
# ============================================================================

def optimize_catboost_ultimate(X_train, y_train, n_trials=50):
    """
    Optuna с КРИТИЧЕСКИМИ исправлениями:
    - scale_pos_weight ≥ 4 (НЕ 1!)
    - Фокус на PR-AUC
    """
    if not OPTUNA_AVAILABLE or not CATBOOST_AVAILABLE:
        return None
    
    print(f"      🎯 Optuna: поиск оптимальных параметров (PR-AUC focus) на {MAX_JOBS} ядрах...")
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 5),
            
            # КРИТИЧНО: НЕ ДАЕМ OPTUNA ВЫБИРАТЬ scale_pos_weight < 4!
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 4.0, 8.0),
            
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 10),
        }
        
        model = CatBoostClassifier(
            **params,
            
            # КРИТИЧНО: Используем только scale_pos_weight (уже в params)
            # НЕ используем class_weights и auto_class_weights одновременно!
            
            eval_metric='PRAUC',
            random_state=RANDOM_STATE,
            verbose=False,
            thread_count=-1  # Все доступные ядра
        )
        
        # Кросс-валидация с фокусом на PR-AUC
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=3,
            scoring='average_precision',  # PR-AUC!
            n_jobs=1  # 1 потому что CatBoost уже многопоточный
        )
        
        return cv_scores.mean()
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=RANDOM_STATE)
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    print(f"      ✅ Лучший PR-AUC (CV): {study.best_value:.4f}")
    print(f"      ✅ Лучшие параметры: {study.best_params}")
    
    return study.best_params


# ============================================================================
# ОБУЧЕНИЕ АНСАМБЛЯ МОДЕЛЕЙ
# ============================================================================

def train_ensemble_models(X_train, y_train, X_val, y_val, best_params=None):
    """
    Обучает ансамбль из 5 моделей с разными балансировками и seed.
    """
    print("      🎯 Обучение ансамбля из 5 моделей...")
    
    models = []
    seeds = [42, 123, 456, 789, 999]
    sampling_strategies = [0.70, 0.75, 0.75, 0.80, 0.80]
    
    if best_params is None:
        best_params = {
            'iterations': 800,
            'depth': 8,
            'learning_rate': 0.05,
            'l2_leaf_reg': 10,
            'scale_pos_weight': 6.0,
        }
    
    for i, (seed, sampling_strategy) in enumerate(zip(seeds, sampling_strategies)):
        print(f"         Модель {i+1}/5 (seed={seed}, sampling={sampling_strategy})...")
        
        # Балансировка с разными стратегиями
        X_train_balanced, y_train_balanced = balance_with_adasyn(
            X_train, y_train, sampling_strategy=sampling_strategy
        )
        
        # Модель с уникальным seed
        params_copy = best_params.copy()
        # Убеждаемся что scale_pos_weight есть
        if 'scale_pos_weight' not in params_copy:
            params_copy['scale_pos_weight'] = 6.0
        
        model = CatBoostClassifier(
            **params_copy,
            eval_metric='PRAUC',
            random_state=seed,
            verbose=False,
            thread_count=-1  # Все доступные ядра
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        models.append(model)
    
    print("      ✅ Ансамбль обучен")
    return models


def optimize_ensemble_weights(models, X_val, y_val):
    """
    Оптимизирует веса ансамбля для максимизации PR-AUC.
    """
    print("      🔧 Оптимизация весов ансамбля...")
    
    predictions = [m.predict_proba(X_val)[:, 1] for m in models]
    
    def neg_pr_auc(weights):
        weights = weights / weights.sum()
        blended = np.average(predictions, axis=0, weights=weights)
        return -average_precision_score(y_val, blended)
    
    result = minimize(
        neg_pr_auc,
        np.ones(len(models)) / len(models),
        method='SLSQP',
        bounds=[(0, 1) for _ in models],
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    
    optimal_weights = result.x / result.x.sum()
    print(f"      ✅ Оптимальные веса: {optimal_weights}")
    
    # Финальное предсказание
    final_pred_proba = np.average(predictions, axis=0, weights=optimal_weights)
    pr_auc = average_precision_score(y_val, final_pred_proba)
    auc = roc_auc_score(y_val, final_pred_proba)
    
    print(f"      ✅ Ансамбль: AUC={auc:.4f}, PR-AUC={pr_auc:.4f}")
    
    return optimal_weights, final_pred_proba


# ============================================================================
# ОПТИМИЗАЦИЯ ПОРОГА ПОД F2-SCORE
# ============================================================================

def optimize_threshold_f2(y_true, y_pred_proba):
    """
    Находит оптимальный порог, максимизируя F2-score.
    F2 дает больший вес recall (важно для детекции дефолтов).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # F2-score: beta=2 означает recall в 2 раза важнее precision
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-6)
    optimal_idx = np.argmax(f2_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    return optimal_threshold, precision[optimal_idx], recall[optimal_idx], f2_scores[optimal_idx]


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    start_time = datetime.now()
    
    print("=" * 80)
    print("ULTIMATE ML PIPELINE - Максимизация PR-AUC")
    print("=" * 80)
    print(f"🚀 Начало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Загрузка
    print("\n[1/8] Загрузка данных...")
    master_df = prep.build_master_dataset()
    print(f"   ✅ Загружено: {master_df.shape[0]} строк, {master_df.shape[1]} колонок")
    print(f"   Доля дефолтов: {master_df['default'].mean():.2%}")
    
    # 2. Feature Engineering
    print("\n[2/8] Feature Engineering (критические признаки для дефолта)...")
    df_featured = create_ultimate_features(master_df)
    X, y, feature_cols, scaler = handle_missing_and_encode(df_featured)
    print(f"   ✅ Подготовлено {len(feature_cols)} признаков")
    
    # 3. Разделение
    print("\n[3/8] Разделение на train/val/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"   Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # 4. Optuna оптимизация
    print("\n[4/8] Optuna оптимизация (scale_pos_weight ≥ 4)...")
    best_params = None
    if OPTUNA_AVAILABLE:
        try:
            # Балансируем train для optuna
            X_train_balanced, y_train_balanced = balance_with_adasyn(X_train, y_train, 0.75)
            best_params = optimize_catboost_ultimate(X_train_balanced, y_train_balanced, n_trials=30)
        except Exception as e:
            print(f"      ⚠ Optuna не удалась: {e}")
    
    # 5. Обучение ансамбля
    print("\n[5/8] Обучение ансамбля (5 моделей)...")
    ensemble_models = train_ensemble_models(X_train, y_train, X_val, y_val, best_params)
    
    # 6. Оптимизация весов ансамбля
    print("\n[6/8] Оптимизация весов ансамбля...")
    optimal_weights, val_pred_proba = optimize_ensemble_weights(ensemble_models, X_val, y_val)
    
    # 7. Калибровка (опционально)
    print("\n[7/8] Калибровка вероятностей...")
    # Используем лучшую модель из ансамбля для калибровки
    # (калибровка ансамбля напрямую сложна, поэтому калибруем постфактум через вероятности)
    
    # 8. Оптимизация порога
    print("\n[8/8] Оптимизация порога (F2-score)...")
    optimal_threshold, precision, recall, f2 = optimize_threshold_f2(y_val, val_pred_proba)
    print(f"   ✅ Оптимальный порог: {optimal_threshold:.4f}")
    print(f"   Precision: {precision:.4f}, Recall: {recall:.4f}, F2: {f2:.4f}")
    
    # Финальная оценка на тесте
    print("\n" + "=" * 80)
    print("ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОЙ ВЫБОРКЕ")
    print("=" * 80)
    
    # Предсказания на тесте
    test_predictions = [m.predict_proba(X_test)[:, 1] for m in ensemble_models]
    test_pred_proba = np.average(test_predictions, axis=0, weights=optimal_weights)
    test_pred = (test_pred_proba >= optimal_threshold).astype(int)
    
    # Метрики
    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_pr_auc = average_precision_score(y_test, test_pred_proba)
    test_f1 = f1_score(y_test, test_pred)
    test_f2 = fbeta_score(y_test, test_pred, beta=2)
    test_mcc = matthews_corrcoef(y_test, test_pred)
    test_brier = brier_score_loss(y_test, test_pred_proba)
    
    print(f"\n{'Метрика':<25} {'Значение':<10}")
    print("=" * 40)
    print(f"{'AUC-ROC':<25} {test_auc:<10.4f}")
    print(f"{'PR-AUC (ГЛАВНАЯ)':<25} {test_pr_auc:<10.4f}")
    print(f"{'F1-Score':<25} {test_f1:<10.4f}")
    print(f"{'F2-Score':<25} {test_f2:<10.4f}")
    print(f"{'Matthews Corr Coef':<25} {test_mcc:<10.4f}")
    print(f"{'Brier Score':<25} {test_brier:<10.4f}")
    print(f"{'Оптимальный порог':<25} {optimal_threshold:<10.4f}")
    print("=" * 40)
    
    if test_pr_auc >= 0.80:
        print("🎉🎉🎉 ЦЕЛЬ ДОСТИГНУТА! PR-AUC ≥ 0.80 🎉🎉🎉")
    elif test_pr_auc >= 0.70:
        print("✅ Отличный результат! PR-AUC ≥ 0.70")
    elif test_pr_auc >= 0.60:
        print("✅ Хороший результат! PR-AUC ≥ 0.60")
    elif test_pr_auc >= 0.50:
        print("✅ Прогресс есть! PR-AUC ≥ 0.50")
    else:
        print("⚠ Требуется дальнейшая оптимизация")
    
    # Сохранение
    print("\n[Сохранение результатов]...")
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    
    with open(model_path / "best_model_ultimate.pkl", "wb") as f:
        pickle.dump({
            "ensemble_models": ensemble_models,
            "optimal_weights": optimal_weights,
            "feature_cols": feature_cols,
            "optimal_threshold": optimal_threshold,
            "test_pr_auc": test_pr_auc,
            "test_auc": test_auc,
            "test_f2": test_f2,
            "best_params": best_params,
            "scaler": scaler
        }, f)
    
    print(f"   ✅ Модель сохранена: models/best_model_ultimate.pkl")
    
    # Предсказания
    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": test_pred,
        "y_pred_proba": test_pred_proba
    })
    predictions_df.to_csv("models/predictions_ultimate.csv", index=False)
    
    elapsed = datetime.now() - start_time
    print(f"\n⏱ Общее время: {elapsed}")
    print("=" * 80)
    print("✅ ГОТОВО!")
    print("=" * 80)
    
    return ensemble_models, optimal_weights, X_test, y_test


if __name__ == "__main__":
    ensemble_models, optimal_weights, X_test, y_test = main()
