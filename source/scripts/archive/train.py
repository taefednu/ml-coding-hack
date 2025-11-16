"""
ML Pipeline для предсказания дефолта заемщика.

Этот скрипт:
1. Загружает и подготавливает данные
2. Обрабатывает пропуски и выбросы
3. Создает фичи
4. Обучает несколько моделей с кросс-валидацией
5. Оценивает метрики (AUC, PR-AUC)
6. Сохраняет лучшую модель
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# ML библиотеки
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE

# Gradient Boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠ XGBoost недоступен, пропускаем: {type(e).__name__}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"⚠ LightGBM недоступен, пропускаем: {type(e).__name__}")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    CATBOOST_AVAILABLE = False
    print(f"⚠ CatBoost недоступен, пропускаем: {type(e).__name__}")

import prepare_data as prep


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

BASE_PATH = Path("data_sets")
RANDOM_STATE = 42
N_SPLITS = 5  # для StratifiedKFold

# Колонки, которые исключаем из обучения
EXCLUDE_COLS = ["customer_id", "default"]


# ============================================================================
# ФУНКЦИИ ПОДГОТОВКИ ДАННЫХ
# ============================================================================

def handle_missing_values(df: pd.DataFrame, target_col: str = "default") -> pd.DataFrame:
    """
    Обработка пропусков:
    - Числовые: медиана
    - Категориальные: мода или "unknown"
    """
    df = df.copy()
    
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
            
        if df[col].isna().sum() == 0:
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Числовые: медиана
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
        else:
            # Категориальные: мода или "unknown"
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("unknown")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Кодирует категориальные признаки:
    - Label Encoding для категориальных
    - Возвращает датафрейм и словарь энкодеров
    """
    df = df.copy()
    encoders = {}
    
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
            
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            # Заполняем пропуски перед кодированием
            df[col] = df[col].fillna("unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает дополнительные фичи (feature engineering).
    """
    df = df.copy()
    
    # Примеры фичей (добавь свои по необходимости):
    
    # 1. Долговая нагрузка (если есть доход и долг)
    if "monthly_income" in df.columns and "total_monthly_debt_payment" in df.columns:
        df["debt_to_income_ratio"] = (
            df["total_monthly_debt_payment"] / (df["monthly_income"] + 1e-6)
        )
    
    # 2. Использование кредита (если есть)
    if "credit_usage_amount" in df.columns and "available_credit" in df.columns:
        df["credit_utilization"] = (
            df["credit_usage_amount"] / (df["available_credit"] + df["credit_usage_amount"] + 1e-6)
        )
    
    # 3. Свободный денежный поток относительно дохода
    if "monthly_free_cash_flow" in df.columns and "monthly_income" in df.columns:
        df["cash_flow_ratio"] = (
            df["monthly_free_cash_flow"] / (df["monthly_income"] + 1e-6)
        )
    
    # 4. Возраст и стаж работы
    if "age" in df.columns and "employment_length" in df.columns:
        df["employment_to_age_ratio"] = (
            df["employment_length"] / (df["age"] + 1e-6)
        )
    
    return df


def prepare_features(df: pd.DataFrame, target_col: str = "default") -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Полная подготовка фичей:
    1. Обработка пропусков
    2. Feature engineering
    3. Кодирование категориальных
    4. Разделение на X и y
    """
    # Обработка пропусков
    df = handle_missing_values(df, target_col)
    
    # Feature engineering
    df = create_features(df)
    
    # Кодирование
    df, encoders = encode_categorical_features(df)
    
    # Разделение
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Убеждаемся, что все числовые
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    
    return X, y, feature_cols


# ============================================================================
# МОДЕЛИ
# ============================================================================

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                               X_test: pd.DataFrame, y_test: pd.Series):
    """Логистическая регрессия с балансировкой классов."""
    # Стандартизация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Модель с балансировкой
    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs"
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Предсказания
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Метрики
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        "model": model,
        "scaler": scaler,
        "auc": auc,
        "pr_auc": pr_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "Logistic Regression"
    }


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series):
    """Random Forest с балансировкой классов."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        "model": model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "Random Forest"
    }


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, y_test: pd.Series):
    """XGBoost с балансировкой классов."""
    if not XGBOOST_AVAILABLE:
        return None
    
    # Вычисляем scale_pos_weight для балансировки
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        "model": model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "XGBoost"
    }


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series):
    """LightGBM с балансировкой классов."""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        "model": model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "LightGBM"
    }


def train_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series):
    """CatBoost с балансировкой классов."""
    if not CATBOOST_AVAILABLE:
        return None
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = CatBoostClassifier(
        iterations=200,
        depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    return {
        "model": model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "CatBoost"
    }


# ============================================================================
# КРОСС-ВАЛИДАЦИЯ
# ============================================================================

def cross_validate_model(model_func, X: pd.DataFrame, y: pd.Series, 
                        cv_splits: int = N_SPLITS):
    """
    Кросс-валидация модели.
    """
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    
    auc_scores = []
    pr_auc_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        result = model_func(X_train, y_train, X_val, y_val)
        if result:
            auc_scores.append(result["auc"])
            pr_auc_scores.append(result["pr_auc"])
    
    return {
        "auc_mean": np.mean(auc_scores),
        "auc_std": np.std(auc_scores),
        "pr_auc_mean": np.mean(pr_auc_scores),
        "pr_auc_std": np.std(pr_auc_scores),
        "auc_scores": auc_scores,
        "pr_auc_scores": pr_auc_scores
    }


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 80)
    print("ML PIPELINE: Предсказание дефолта заемщика")
    print("=" * 80)
    
    # 1. Загрузка данных
    print("\n[1/6] Загрузка данных...")
    master_df = prep.build_master_dataset()
    print(f"   ✅ Загружено: {master_df.shape[0]} строк, {master_df.shape[1]} колонок")
    
    # Проверка таргета
    if "default" not in master_df.columns:
        raise ValueError("❌ Колонка 'default' не найдена!")
    
    print(f"\n   Распределение таргета:")
    print(f"   {master_df['default'].value_counts().to_dict()}")
    print(f"   Доля дефолтов: {master_df['default'].mean():.2%}")
    
    # 2. Подготовка фичей
    print("\n[2/6] Подготовка фичей...")
    X, y, feature_cols = prepare_features(master_df)
    print(f"   ✅ Подготовлено {len(feature_cols)} признаков")
    print(f"   Пропуски в X: {X.isna().sum().sum()}")
    
    # 3. Разделение на train/test
    print("\n[3/6] Разделение на train/test...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} строк")
    print(f"   Test: {X_test.shape[0]} строк")
    
    # 4. Обучение моделей
    print("\n[4/6] Обучение моделей...")
    models = []
    
    # Logistic Regression
    print("   🚀 Обучение Logistic Regression...")
    try:
        lr_result = train_logistic_regression(X_train, y_train, X_test, y_test)
        if lr_result:
            models.append(lr_result)
            print(f"      ✅ AUC: {lr_result['auc']:.4f}, PR-AUC: {lr_result['pr_auc']:.4f}")
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
    
    # Random Forest
    print("   🚀 Обучение Random Forest...")
    try:
        rf_result = train_random_forest(X_train, y_train, X_test, y_test)
        if rf_result:
            models.append(rf_result)
            print(f"      ✅ AUC: {rf_result['auc']:.4f}, PR-AUC: {rf_result['pr_auc']:.4f}")
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        print("   🚀 Обучение XGBoost...")
        try:
            xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
            if xgb_result:
                models.append(xgb_result)
                print(f"      ✅ AUC: {xgb_result['auc']:.4f}, PR-AUC: {xgb_result['pr_auc']:.4f}")
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print("   🚀 Обучение LightGBM...")
        try:
            lgb_result = train_lightgbm(X_train, y_train, X_test, y_test)
            if lgb_result:
                models.append(lgb_result)
                print(f"      ✅ AUC: {lgb_result['auc']:.4f}, PR-AUC: {lgb_result['pr_auc']:.4f}")
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
    
    # CatBoost
    if CATBOOST_AVAILABLE:
        print("   🚀 Обучение CatBoost...")
        try:
            cb_result = train_catboost(X_train, y_train, X_test, y_test)
            if cb_result:
                models.append(cb_result)
                print(f"      ✅ AUC: {cb_result['auc']:.4f}, PR-AUC: {cb_result['pr_auc']:.4f}")
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
    
    if not models:
        raise ValueError("❌ Ни одна модель не была обучена!")
    
    # 5. Выбор лучшей модели
    print("\n[5/6] Сравнение моделей...")
    print("\n" + "=" * 80)
    print(f"{'Модель':<20} {'AUC':<10} {'PR-AUC':<10}")
    print("=" * 80)
    
    for model_result in models:
        print(f"{model_result['name']:<20} {model_result['auc']:<10.4f} {model_result['pr_auc']:<10.4f}")
    
    # Лучшая по PR-AUC (важнее для несбалансированных данных)
    best_model = max(models, key=lambda x: x['pr_auc'])
    print("\n" + "=" * 80)
    print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model['name']}")
    print(f"   AUC: {best_model['auc']:.4f}")
    print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    print("=" * 80)
    
    # 6. Сохранение
    print("\n[6/6] Сохранение результатов...")
    
    # Сохраняем лучшую модель
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    
    with open(model_path / "best_model.pkl", "wb") as f:
        pickle.dump({
            "model": best_model["model"],
            "scaler": best_model["scaler"],
            "feature_cols": feature_cols,
            "model_name": best_model["name"],
            "auc": best_model["auc"],
            "pr_auc": best_model["pr_auc"]
        }, f)
    
    print(f"   ✅ Модель сохранена: models/best_model.pkl")
    
    # Сохраняем результаты всех моделей
    results_df = pd.DataFrame([
        {
            "model": m["name"],
            "auc": m["auc"],
            "pr_auc": m["pr_auc"]
        }
        for m in models
    ])
    results_df.to_csv("models/model_comparison.csv", index=False)
    print(f"   ✅ Сравнение моделей: models/model_comparison.csv")
    
    # Сохраняем финальные предсказания лучшей модели
    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": best_model["y_pred"],
        "y_pred_proba": best_model["y_pred_proba"]
    })
    predictions_df.to_csv("models/predictions.csv", index=False)
    print(f"   ✅ Предсказания: models/predictions.csv")
    
    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    
    return best_model, models, X_test, y_test, feature_cols


if __name__ == "__main__":
    best_model, all_models, X_test, y_test, feature_cols = main()

