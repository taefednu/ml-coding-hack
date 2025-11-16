"""
Улучшенный ML Pipeline для предсказания дефолта заемщика.
Фокус на максимизацию PR-AUC до 80%.

Улучшения:
1. Глубокий feature engineering
2. Настройка гиперпараметров (RandomizedSearchCV)
3. SMOTE для балансировки классов
4. Ансамбли моделей (Voting, Stacking)
5. Кросс-валидация с фокусом на PR-AUC
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
import time
from datetime import datetime, timedelta
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

# Прогресс-бары
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("⚠ tqdm не установлен. Установите: pip install tqdm")

from typing import Tuple
from sklearn.model_selection import (
    StratifiedKFold, 
    train_test_split,
    RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    make_scorer,
    classification_report,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Gradient Boosting
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"⚠ XGBoost недоступен: {type(e).__name__}")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"⚠ LightGBM недоступен: {type(e).__name__}")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    CATBOOST_AVAILABLE = False
    print(f"⚠ CatBoost недоступен: {type(e).__name__}")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except (ImportError, Exception):
    OPTUNA_AVAILABLE = False
    print("⚠ Optuna недоступна, используем RandomizedSearchCV")

import prepare_data as prep


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

BASE_PATH = Path("data_sets")
RANDOM_STATE = 42
N_SPLITS = 5
EXCLUDE_COLS = ["customer_id", "default", "application_id"]  # application_id может быть шумом
MAX_JOBS = int(os.environ.get("MAX_JOBS", "1"))
CATBOOST_THREADS = int(os.environ.get("CATBOOST_THREADS", str(MAX_JOBS)))

# Строковый скоуер безопасен даже в старых версиях sklearn
PR_AUC_SCORER = "average_precision"

# Веса ансамбля: 1.0 = только PR-AUC, 0.0 = только Spearman, по умолчанию сильный фокус на PR-AUC
ENSEMBLE_ALPHA = 0.7


# ============================================================================
# УЛУЧШЕННЫЙ FEATURE ENGINEERING
# ============================================================================

def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Глубокий feature engineering:
    - Взаимодействия признаков
    - Полиномиальные признаки
    - Группировки и агрегации
    - Временные признаки
    """
    df = df.copy()
    
    print("   🔧 Создание расширенных признаков...")

    def add_woe_encoding(column_name: str, smoothing: float = 5.0):
        """
        Добавляет WOE-энкодинг для категориального признака с легкой
        регуляризацией (сглаживание), чтобы уменьшить шум на редких категориях.
        """
        if column_name not in df.columns or "default" not in df.columns:
            return
        temp = df[[column_name, "default"]].dropna()
        if temp.empty:
            return
        stats = temp.groupby(column_name)["default"].agg(["sum", "count"])
        stats["non_default"] = stats["count"] - stats["sum"]
        total_pos = stats["sum"].sum()
        total_neg = stats["non_default"].sum()
        if total_pos == 0 or total_neg == 0:
            return
        n_unique = stats.shape[0]
        stats["pos_dist"] = (stats["sum"] + smoothing) / (total_pos + smoothing * n_unique)
        stats["neg_dist"] = (stats["non_default"] + smoothing) / (total_neg + smoothing * n_unique)
        stats["woe"] = np.log(stats["pos_dist"] / stats["neg_dist"])
        df[f"{column_name}_woe"] = df[column_name].map(stats["woe"]).fillna(0)

    def add_group_target_rate(column_name: str, new_col: str, smoothing: float = 5.0):
        """Сглаженная частота дефолтов по категориям (target encoding)."""
        if column_name not in df.columns or "default" not in df.columns:
            return
        temp = df[[column_name, "default"]].dropna()
        if temp.empty:
            return
        stats = temp.groupby(column_name)["default"].agg(["sum", "count"])
        global_rate = temp["default"].mean()
        stats[new_col] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
        df[new_col] = df[column_name].map(stats[new_col]).fillna(global_rate)
    
    # 1. Базовые финансовые соотношения
    if "monthly_income" in df.columns:
        if "total_monthly_debt_payment" in df.columns:
            df["debt_to_income_ratio"] = (
                df["total_monthly_debt_payment"] / (df["monthly_income"] + 1e-6)
            )
            df["debt_to_income_ratio_squared"] = df["debt_to_income_ratio"] ** 2
        
        if "monthly_free_cash_flow" in df.columns:
            df["cash_flow_ratio"] = (
                df["monthly_free_cash_flow"] / (df["monthly_income"] + 1e-6)
            )
            df["cash_surplus_ratio"] = (
                (df["monthly_income"] - df["total_monthly_debt_payment"].fillna(0))
                / (df["monthly_income"] + 1e-6)
                if "total_monthly_debt_payment" in df.columns else np.nan
            )
            if "total_monthly_debt_payment" in df.columns:
                df["cash_to_debt_ratio"] = (
                    (df["monthly_free_cash_flow"] + 1) / (df["total_monthly_debt_payment"] + 1)
                )
                df["free_cashflow_stress_flag"] = (
                    (df["cash_surplus_ratio"] < -0.1).astype(int)
                )
        
        if "loan_amount" in df.columns:
            df["loan_to_income_ratio"] = (
                df["loan_amount"] / (df["monthly_income"] * 12 + 1e-6)
            )
            if "total_credit_limit" in df.columns:
                df["loan_to_credit_limit_ratio"] = (
                    df["loan_amount"] / (df["total_credit_limit"] + 1e-6)
                )
            if "available_credit" in df.columns:
                df["loan_to_available_credit"] = (
                    df["loan_amount"] / (df["available_credit"] + 1e-6)
                )
    
    # 2. Кредитные метрики
    if "credit_score" in df.columns:
        # Нормализация кредитного скора
        df["credit_score_normalized"] = (df["credit_score"] - 300) / (850 - 300)
        
        # Биннинг кредитного скора
        df["credit_score_bin"] = pd.cut(
            df["credit_score"],
            bins=[0, 580, 670, 740, 850],
            labels=[0, 1, 2, 3]  # poor, fair, good, excellent
        ).astype(float).fillna(0)  # Заполняем NaN нулем
        
        # Взаимодействие кредитного скора с доходом (ВЫСОКИЙ ПРИОРИТЕТ)
        if "annual_income" in df.columns:
            df["credit_score_income_interaction"] = (
                df["credit_score_normalized"] * np.log1p(df["annual_income"])
            )
        
        # Взаимодействие кредитного скора с долговой нагрузкой (КЛЮЧЕВОЕ!)
        if "debt_to_income_ratio" in df.columns:
            df["credit_score_debt_interaction"] = (
                (1 - df["credit_score_normalized"]) * df["debt_to_income_ratio"]
            )
            df["credit_stress_ratio"] = (
                df["debt_to_income_ratio"] * (1 + df["credit_score_normalized"])
            )
    
    # 3. Использование кредита
    if "credit_usage_amount" in df.columns and "available_credit" in df.columns:
        df["credit_utilization"] = (
            df["credit_usage_amount"] / (df["available_credit"] + df["credit_usage_amount"] + 1e-6)
        )
        
        # Квадрат использования кредита (нелинейность)
        df["credit_utilization_squared"] = df["credit_utilization"] ** 2
        
        # Биннинг использования кредита
        df["credit_utilization_bin"] = pd.cut(
            df["credit_utilization"],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=[0, 1, 2, 3]  # low, medium, high, very_high
        ).astype(float).fillna(0)  # Заполняем NaN нулем
        
        # Взаимодействие: использование кредита + просрочки (КЛЮЧЕВОЕ!)
        if "num_delinquencies_2yrs" in df.columns:
            df["utilization_delinquency_interaction"] = (
                df["credit_utilization"] * df["num_delinquencies_2yrs"].fillna(0)
            )
        if "credit_score_normalized" in df.columns:
            df["score_utilization_interaction"] = (
                df["credit_utilization"] * (1 - df["credit_score_normalized"])
            )
    
    # 4. Долговая нагрузка
    if "total_debt_amount" in df.columns and "annual_income" in df.columns:
        df["debt_to_annual_income"] = (
            df["total_debt_amount"] / (df["annual_income"] + 1e-6)
        )
    
    if "debt_to_income_ratio" in df.columns and "credit_utilization" in df.columns:
        df["combined_risk_ratio"] = df["debt_to_income_ratio"] * (df["credit_utilization"] + 1)
    
    if "debt_to_income_ratio" in df.columns:
        df["debt_service_ratio"] = df["debt_to_income_ratio"] * df.get("loan_to_income_ratio", 1)
        df["dti_high_flag"] = (df["debt_to_income_ratio"] > 0.6).astype(int)
        df["dti_extreme_flag"] = (df["debt_to_income_ratio"] > 0.85).astype(int)
        try:
            df["dti_quantile_bin"] = pd.qcut(
                df["debt_to_income_ratio"].clip(0, 5), q=4, labels=[0, 1, 2, 3]
            ).astype(float)
        except ValueError:
            df["dti_quantile_bin"] = 0.0
    
    # 5. Возраст и стаж
    if "age" in df.columns:
        # Квадрат возраста (нелинейность)
        df["age_squared"] = df["age"] ** 2
        
        # Биннинг возраста
        df["age_bin"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 100],
            labels=[0, 1, 2, 3, 4]  # young, adult, middle, senior, elderly
        ).astype(float).fillna(1)  # Заполняем NaN средним значением (adult)
        
        if "employment_length" in df.columns:
            df["employment_to_age_ratio"] = (
                df["employment_length"] / (df["age"] + 1e-6)
            )
            
            # Взаимодействие возраста и стажа (молодой + короткий стаж = риск)
            df["age_employment_interaction"] = (
                (df["age"] < 30).astype(int) * (df["employment_length"] < 2).astype(int)
            )
    
    # 6. Кредитная история
    if "num_credit_accounts" in df.columns:
        if "oldest_credit_line_age" in df.columns:
            df["credit_accounts_per_year"] = (
                df["num_credit_accounts"] / (df["oldest_credit_line_age"] + 1e-6)
            )
        
        # Проблемы с кредитами
        if "num_delinquencies_2yrs" in df.columns:
            df["delinquency_rate"] = (
                df["num_delinquencies_2yrs"] / (df["num_credit_accounts"] + 1e-6)
            )
        
        if "num_collections" in df.columns:
            df["collection_rate"] = (
                df["num_collections"] / (df["num_credit_accounts"] + 1e-6)
            )
    
    # 7. Географические признаки
    if "regional_median_income" in df.columns and "annual_income" in df.columns:
        df["income_vs_regional"] = (
            df["annual_income"] / (df["regional_median_income"] + 1e-6)
        )
        df["income_gap_vs_region"] = (
            df["annual_income"] - df["regional_median_income"]
        )
    if "regional_unemployment_rate" in df.columns and "unemployment_squared" not in df.columns:
        df["unemployment_squared"] = df["regional_unemployment_rate"] ** 2
    
    if "regional_unemployment_rate" in df.columns:
        # Квадрат безработицы
        df["unemployment_squared"] = df["regional_unemployment_rate"] ** 2
    
    # 8. Временные признаки
    if "application_hour" in df.columns:
        # Циклические признаки для времени
        df["hour_sin"] = np.sin(2 * np.pi * df["application_hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["application_hour"] / 24)
    
    if "application_day_of_week" in df.columns:
        df["day_sin"] = np.sin(2 * np.pi * df["application_day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["application_day_of_week"] / 7)
    
    if "account_open_year" in df.columns:
        # Сколько лет назад открыт счет
        current_year = df["account_open_year"].max()
        df["account_age_years"] = current_year - df["account_open_year"]
        if "num_login_sessions" in df.columns:
            df["sessions_per_year"] = (
                df["num_login_sessions"] / (df["account_age_years"] + 1)
            )
        if "loan_amount" in df.columns:
            df["loan_per_account_age"] = df["loan_amount"] / (df["account_age_years"] + 1)
    
    # 9. Комплексные индексы риска
    if all(col in df.columns for col in ["credit_score_normalized", "debt_to_income_ratio", 
                                         "credit_utilization", "delinquency_rate"]):
        df["risk_index"] = (
            (1 - df["credit_score_normalized"]) * 0.3 +
            df["debt_to_income_ratio"].clip(0, 2) * 0.3 +
            df["credit_utilization"] * 0.2 +
            df["delinquency_rate"].fillna(0) * 0.2
        )
    
    # 10. Логарифмические преобразования для skewed признаков
    skewed_cols = ["annual_income", "monthly_income", "loan_amount", "total_debt_amount"]
    for col in skewed_cols:
        if col in df.columns:
            df[f"{col}_log"] = np.log1p(df[col])
    
    # 11. Поведенческие паттерны
    if all(col in df.columns for col in ["num_login_sessions", "num_customer_service_calls"]):
        df["service_call_ratio"] = (
            df["num_customer_service_calls"] / (df["num_login_sessions"] + 1)
        )
        median_sessions = df["num_login_sessions"].median()
        median_calls = df["num_customer_service_calls"].median()
        df["low_activity_high_calls"] = (
            (df["num_login_sessions"] < median_sessions) &
            (df["num_customer_service_calls"] > median_calls)
        ).astype(int)
    
    # 12. WOE-энкодинг ключевых категорий
    categorical_for_woe = [
        "employment_type",
        "preferred_contact",
        "state",
        "region",
        "account_status_code",
        "referral_code"
    ]
    for cat_col in categorical_for_woe:
        add_woe_encoding(cat_col, smoothing=10.0)

    target_rate_cols = {
        "state": "state_default_rate",
        "region": "region_default_rate",
        "employment_type": "employment_default_rate",
        "referral_code": "referral_default_rate"
    }
    for col, new_col in target_rate_cols.items():
        add_group_target_rate(col, new_col, smoothing=20.0)
    
    print(f"   ✅ Создано расширенных признаков. Всего признаков: {len(df.columns)}")
    
    return df


# ============================================================================
# ОБРАБОТКА ДАННЫХ
# ============================================================================

def handle_missing_values_advanced(df: pd.DataFrame, target_col: str = "default") -> pd.DataFrame:
    """Улучшенная обработка пропусков."""
    df = df.copy()
    
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
            
        if df[col].isna().sum() == 0:
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            # Для числовых: медиана, но с учетом распределения
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)
        else:
            # Для категориальных: мода или "unknown"
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("unknown")
    
    return df


def encode_categorical_advanced(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Кодирование категориальных признаков."""
    df = df.copy()
    encoders = {}
    
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
            
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            le = LabelEncoder()
            df[col] = df[col].fillna("unknown")
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders


def prepare_features_advanced(df: pd.DataFrame, target_col: str = "default") -> Tuple[pd.DataFrame, pd.Series, list]:
    """Полная подготовка фичей с улучшенным feature engineering."""
    # Обработка пропусков
    df = handle_missing_values_advanced(df, target_col)
    
    # Расширенный feature engineering
    df = advanced_feature_engineering(df)
    
    # Кодирование
    df, encoders = encode_categorical_advanced(df)
    
    # Разделение
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Убеждаемся, что все числовые
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
    
    # Замена inf и очень больших значений
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X, y, feature_cols


def balance_with_smoteenn(X: pd.DataFrame, y: pd.Series):
    """Применяет SMOTEENN для усиления сигнала положительного класса."""
    try:
        sampler = SMOTEENN(random_state=RANDOM_STATE)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except Exception as exc:
        print(f"      ⚠ Не удалось применить SMOTEENN: {exc}")
        return X, y


def balance_with_smote_tomek(X: pd.DataFrame, y: pd.Series):
    """Использует SMOTE-Tomek для балансировки данных."""
    try:
        sampler = SMOTETomek(random_state=RANDOM_STATE)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
    except Exception as exc:
        print(f"      ⚠ Не удалось применить SMOTETomek: {exc}. Возвращаем SMOTEENN.")
        return balance_with_smoteenn(X, y)


# ============================================================================
# МОДЕЛИ С НАСТРОЙКОЙ ГИПЕРПАРАМЕТРОВ
# ============================================================================

def train_tuned_catboost(X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series):
    """CatBoost с настройкой гиперпараметров и агрессивной балансировкой."""
    if not CATBOOST_AVAILABLE:
        return None
    
    print("      🔍 Настройка гиперпараметров CatBoost...")

    print("      🔁 Балансировка классов через SMOTE-Tomek...")
    X_train_balanced, y_train_balanced = balance_with_smote_tomek(X_train, y_train)
    
    # Агрессивная балансировка для PR-AUC
    base_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

    if OPTUNA_AVAILABLE:
        try:
            best_model, study = tune_catboost_with_optuna(X_train_balanced, y_train_balanced)
            if best_model:
                y_pred_proba = best_model.predict_proba(X_val)[:, 1]
                y_pred = best_model.predict(X_val)
                auc = roc_auc_score(y_val, y_pred_proba)
                pr_auc = average_precision_score(y_val, y_pred_proba)
                spearman_corr, _ = spearmanr(y_val, y_pred_proba)
                return {
                    "model": best_model,
                    "scaler": None,
                    "auc": auc,
                    "pr_auc": pr_auc,
                    "spearman": spearman_corr,
                    "y_pred_proba": y_pred_proba,
                    "y_pred": y_pred,
                    "name": "CatBoost (Optuna Tuned)",
                    "best_params": study.best_params
                }
        except Exception as exc:
            print(f"      ⚠ Optuna не удалась: {exc}. Переходим к RandomizedSearchCV.")
    
    # Параметры для поиска с фокусом на PR-AUC (умеренный объем)
    param_distributions = {
        'iterations': [300, 500, 700],
        'depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'l2_leaf_reg': [3, 5, 7],
        'scale_pos_weight': [base_weight, base_weight * 1.5, base_weight * 2],
        'min_data_in_leaf': [1, 3]
    }
    
    # Базовая модель с фокусом на PR-AUC и прогресс-баром
    base_model = CatBoostClassifier(
        random_state=RANDOM_STATE,
        verbose=100,  # Показываем прогресс каждые 100 итераций
        eval_metric='PRAUC',
        loss_function='Logloss',
        thread_count=CATBOOST_THREADS  # Учитываем ограничения окружения
    )
    
    # RandomizedSearchCV с прогресс-баром
    n_iter = 15
    cv_folds = 2
    total_fits = n_iter * cv_folds
    
    print(f"      📊 Всего будет обучено: {total_fits} моделей")
    print(f"      ⏱ Начало: {datetime.now().strftime('%H:%M:%S')}")
    print(f"      💻 Используются ресурсы CPU (n_jobs={MAX_JOBS}, thread_count={CATBOOST_THREADS})")
    
    # RandomizedSearchCV (оптимизировано: меньше итераций, меньше CV)
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        scoring=PR_AUC_SCORER,
        cv=cv_folds,
        n_jobs=MAX_JOBS,
        random_state=RANDOM_STATE,
        verbose=1  # Показываем прогресс sklearn (будет видно в консоли)
    )
    
    start_time = time.time()
    print(f"      🔄 Обучение {total_fits} моделей (sklearn покажет прогресс)...")
    random_search.fit(X_train_balanced, y_train_balanced)
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    mins = int((elapsed_time % 3600) // 60)
    secs = int(elapsed_time % 60)
    print(f"      ✅ Завершено за: {hours:02d}:{mins:02d}:{secs:02d} ({elapsed_time/60:.1f} минут)")
    
    best_model = random_search.best_estimator_
    
    # Предсказания
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    spearman_corr, _ = spearmanr(y_val, y_pred_proba)
    
    print(f"      ✅ Лучшие параметры: {random_search.best_params_}")
    
    return {
        "model": best_model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "spearman": spearman_corr,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "CatBoost (Tuned)",
        "best_params": random_search.best_params_
    }


def train_tuned_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series):
    """LightGBM с настройкой гиперпараметров."""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    print("      🔍 Настройка гиперпараметров LightGBM...")
    print("      🔁 Балансировка классов через SMOTE-Tomek...")
    X_train_balanced, y_train_balanced = balance_with_smote_tomek(X_train, y_train)
    
    base_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()
    
    param_distributions = {
        'n_estimators': [300, 500, 700],
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 10, 15],
        'learning_rate': [0.03, 0.05, 0.1],
        'subsample': [0.7, 0.85, 1.0],
        'colsample_bytree': [0.7, 0.85, 1.0],
        'reg_alpha': [0, 1, 5],
        'reg_lambda': [0, 1, 5],
        'scale_pos_weight': [base_weight, base_weight * 1.5, base_weight * 2]
    }
    
    base_model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        random_state=RANDOM_STATE,
        n_jobs=MAX_JOBS
    )
    
    n_iter = 12
    cv_folds = 2
    total_fits = n_iter * cv_folds
    
    print(f"      📊 Всего будет обучено: {total_fits} моделей")
    print(f"      ⏱ Начало: {datetime.now().strftime('%H:%M:%S')}")
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        scoring=PR_AUC_SCORER,
        cv=cv_folds,
        n_jobs=MAX_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    start_time = time.time()
    print(f"      🔄 Обучение...", end="", flush=True)
    random_search.fit(X_train_balanced, y_train_balanced)
    elapsed_time = time.time() - start_time
    print(f"\r      ✅ Завершено за: {elapsed_time/60:.1f} минут")
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    spearman_corr, _ = spearmanr(y_val, y_pred_proba)
    
    print(f"      ✅ Лучшие параметры: {random_search.best_params_}")
    
    return {
        "model": best_model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "spearman": spearman_corr,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "LightGBM (Tuned)",
        "best_params": random_search.best_params_
    }


def train_tuned_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series):
    """Random Forest с настройкой гиперпараметров."""
    print("      🔍 Настройка гиперпараметров Random Forest...")

    print("      🔁 Балансировка классов через SMOTE-Tomek...")
    X_train_balanced, y_train_balanced = balance_with_smote_tomek(X_train, y_train)
    
    param_distributions = {
        'n_estimators': [300, 500, 700],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    base_model = RandomForestClassifier(
        random_state=RANDOM_STATE, 
        n_jobs=MAX_JOBS,
        verbose=1 if TQDM_AVAILABLE else 0
    )
    
    n_iter = 8
    cv_folds = 2
    total_fits = n_iter * cv_folds
    
    print(f"      📊 Всего будет обучено: {total_fits} моделей")
    print(f"      ⏱ Начало: {datetime.now().strftime('%H:%M:%S')}")
    print(f"      💻 Используются ресурсы CPU (n_jobs={MAX_JOBS})")
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        scoring=PR_AUC_SCORER,
        cv=cv_folds,
        n_jobs=MAX_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    start_time = time.time()
    print(f"      🔄 Обучение...", end="", flush=True)
    random_search.fit(X_train_balanced, y_train_balanced)
    elapsed_time = time.time() - start_time
    print(f"\r      ✅ Завершено за: {elapsed_time/60:.1f} минут")
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred = best_model.predict(X_val)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    spearman_corr, _ = spearmanr(y_val, y_pred_proba)
    
    print(f"      ✅ Лучшие параметры: {random_search.best_params_}")
    
    return {
        "model": best_model,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "spearman": spearman_corr,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "Random Forest (Tuned)",
        "best_params": random_search.best_params_
    }


def train_tuned_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                                    X_val: pd.DataFrame, y_val: pd.Series, use_borderline=True):
    """Logistic Regression с настройкой и SMOTE/BorderlineSMOTE."""
    print("      🔍 Настройка Logistic Regression с SMOTE...")
    
    # BorderlineSMOTE для балансировки (фокус на граничных случаях)
    if use_borderline:
        try:
            smote = BorderlineSMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print("      ✅ Использован BorderlineSMOTE")
        except:
            smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
            print("      ✅ Использован SMOTE (fallback)")
    else:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Стандартизация
    scaler = RobustScaler()  # RobustScaler более устойчив к выбросам
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_val_scaled = scaler.transform(X_val)
    
    param_distributions = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'lbfgs'],
        'class_weight': ['balanced', None]
    }
    
    base_model = LogisticRegression(
        random_state=RANDOM_STATE, 
        max_iter=2000,
        n_jobs=MAX_JOBS
    )
    
    n_iter = 5
    cv_folds = 2
    total_fits = n_iter * cv_folds
    
    print(f"      📊 Всего будет обучено: {total_fits} моделей")
    print(f"      ⏱ Начало: {datetime.now().strftime('%H:%M:%S')}")
    print(f"      💻 Используются ресурсы CPU (n_jobs={MAX_JOBS})")
    
    random_search = RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=n_iter,
        scoring=PR_AUC_SCORER,
        cv=cv_folds,
        n_jobs=MAX_JOBS,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    start_time = time.time()
    print(f"      🔄 Обучение...", end="", flush=True)
    random_search.fit(X_train_scaled, y_train_balanced)
    elapsed_time = time.time() - start_time
    print(f"\r      ✅ Завершено за: {elapsed_time/60:.1f} минут")
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val_scaled)[:, 1]
    y_pred = best_model.predict(X_val_scaled)
    
    auc = roc_auc_score(y_val, y_pred_proba)
    pr_auc = average_precision_score(y_val, y_pred_proba)
    spearman_corr, _ = spearmanr(y_val, y_pred_proba)
    
    print(f"      ✅ Лучшие параметры: {random_search.best_params_}")
    
    return {
        "model": best_model,
        "scaler": scaler,
        "smote": smote,
        "auc": auc,
        "pr_auc": pr_auc,
        "spearman": spearman_corr,
        "y_pred_proba": y_pred_proba,
        "y_pred": y_pred,
        "name": "Logistic Regression (Tuned + SMOTE)",
        "best_params": random_search.best_params_
    }


# ============================================================================
# АНСАМБЛЬ МОДЕЛЕЙ
# ============================================================================

def create_ensemble(models_results: list, X_val: pd.DataFrame, y_val: pd.Series):
    """Создает ансамбль из лучших моделей."""
    if len(models_results) < 2:
        return None
    
    print("\n   🎯 Создание ансамбля моделей...")
    
    sorted_models = sorted(models_results, key=lambda x: x['pr_auc'], reverse=True)[:3]
    
    predictions = []
    model_names = []
    
    for model_result in sorted_models:
        model = model_result['model']
        scaler = model_result.get('scaler')
        
        if scaler:
            X_val_scaled = scaler.transform(X_val)
            pred = model.predict_proba(X_val_scaled)[:, 1]
        else:
            pred = model.predict_proba(X_val)[:, 1]
        
        predictions.append(pred)
        model_names.append(model_result['name'])
    
    pr_auc_values = np.array([m['pr_auc'] for m in sorted_models])
    spearman_values = np.array([m.get('spearman', 0.0) for m in sorted_models])
    # Нормируем Spearman в [0, 1] через сдвиг, чтобы сочетать с PR-AUC
    spearman_norm = (spearman_values + 1.0) / 2.0
    combined_scores = ENSEMBLE_ALPHA * pr_auc_values + (1.0 - ENSEMBLE_ALPHA) * spearman_norm
    # На случай, если все значения нулевые или NaN
    if np.all(combined_scores <= 0) or np.all(~np.isfinite(combined_scores)):
        combined_scores = pr_auc_values
    weights = combined_scores / combined_scores.sum()
    
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    ensemble_pred_binary = (ensemble_pred >= 0.5).astype(int)
    
    auc = roc_auc_score(y_val, ensemble_pred)
    pr_auc = average_precision_score(y_val, ensemble_pred)
    spearman_corr, _ = spearmanr(y_val, ensemble_pred)
    
    return {
        "model": sorted_models,
        "weights": weights,
        "scaler": None,
        "auc": auc,
        "pr_auc": pr_auc,
        "spearman": spearman_corr,
        "y_pred_proba": ensemble_pred,
        "y_pred": ensemble_pred_binary,
        "name": f"Ensemble ({len(sorted_models)} models)",
        "model_names": model_names
    }


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    pipeline_start_time = time.time()
    
    print("=" * 80)
    print("УЛУЧШЕННЫЙ ML PIPELINE: Предсказание дефолта заемщика")
    print("Цель: PR-AUC >= 0.80")
    print("=" * 80)
    print(f"🚀 Начало: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Режим: Используются доступные ресурсы (n_jobs={MAX_JOBS}, thread_count={CATBOOST_THREADS})")
    print("=" * 80)
    
    # 1. Загрузка данных
    print("\n[1/7] Загрузка данных...")
    master_df = prep.build_master_dataset()
    print(f"   ✅ Загружено: {master_df.shape[0]} строк, {master_df.shape[1]} колонок")
    
    if "default" not in master_df.columns:
        raise ValueError("❌ Колонка 'default' не найдена!")
    
    print(f"\n   Распределение таргета:")
    print(f"   {master_df['default'].value_counts().to_dict()}")
    print(f"   Доля дефолтов: {master_df['default'].mean():.2%}")
    
    # 2. Подготовка фичей
    print("\n[2/7] Подготовка фичей (расширенный feature engineering)...")
    X, y, feature_cols = prepare_features_advanced(master_df)
    print(f"   ✅ Подготовлено {len(feature_cols)} признаков")
    print(f"   Пропуски в X: {X.isna().sum().sum()}")
    
    # 3. Разделение на train/val/test
    print("\n[3/7] Разделение на train/val/test...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"   Train: {X_train.shape[0]} строк")
    print(f"   Val:   {X_val.shape[0]} строк")
    print(f"   Test:  {X_test.shape[0]} строк")
    
    # 4. Обучение моделей с настройкой
    print("\n[4/7] Обучение моделей с настройкой гиперпараметров...")
    print("   ⚠ Это займет время (настройка гиперпараметров)...")
    
    # Подсчет общего количества обучений
    total_model_fits = 8 * 2 + 5 * 2  # RF + LR
    model_types_count = 2  # RF, LR
    if CATBOOST_AVAILABLE:
        total_model_fits += 15 * 2  # CatBoost
        model_types_count += 1
    if LIGHTGBM_AVAILABLE:
        total_model_fits += 12 * 2  # LightGBM
        model_types_count += 1
    
    print(f"   📊 Всего моделей для обучения: {model_types_count} типа(ов)")
    print(f"   📈 Всего будет обучено: ~{total_model_fits} моделей (с учетом CV)")
    print(f"   ⏱ Начало обучения: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   💻 Используются ресурсы CPU (n_jobs={MAX_JOBS}, thread_count={CATBOOST_THREADS})")
    print()
    
    # Общее время начала
    total_start_time = time.time()
    models = []
    model_times = {}
    completed_models = 0
    
    def print_progress():
        elapsed_total = time.time() - total_start_time
        avg_time = elapsed_total / max(completed_models, 1)
        remaining_models = max(model_types_count - completed_models, 0)
        estimated_remaining = avg_time * remaining_models
        progress_pct = (completed_models / model_types_count) * 100
        print(f"      📊 Прогресс: {completed_models}/{model_types_count} моделей ({progress_pct:.0f}%) | ⏳ Осталось: ~{estimated_remaining/60:.1f} мин")

    # CatBoost
    if CATBOOST_AVAILABLE:
        print(f"   🚀 [{completed_models+1}/{model_types_count}] Обучение CatBoost (Tuned)...")
        model_start = time.time()
        try:
            cb_result = train_tuned_catboost(X_train, y_train, X_val, y_val)
            if cb_result:
                models.append(cb_result)
                model_times['CatBoost'] = time.time() - model_start
                completed_models += 1
                print(f"      ✅ AUC: {cb_result['auc']:.4f}, PR-AUC: {cb_result['pr_auc']:.4f}, Spearman: {cb_result['spearman']:.4f} ({model_times['CatBoost']/60:.1f} мин)")
                print_progress()
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")

    # LightGBM
    if LIGHTGBM_AVAILABLE:
        print(f"   🚀 [{completed_models+1}/{model_types_count}] Обучение LightGBM (Tuned)...")
        model_start = time.time()
        try:
            lgb_result = train_tuned_lightgbm(X_train, y_train, X_val, y_val)
            if lgb_result:
                models.append(lgb_result)
                model_times['LightGBM'] = time.time() - model_start
                completed_models += 1
                print(f"      ✅ AUC: {lgb_result['auc']:.4f}, PR-AUC: {lgb_result['pr_auc']:.4f}, Spearman: {lgb_result['spearman']:.4f} ({model_times['LightGBM']/60:.1f} мин)")
                print_progress()
        except Exception as e:
            print(f"      ❌ Ошибка: {e}")
    
    # Random Forest
    print(f"   🚀 [{completed_models+1}/{model_types_count}] Обучение Random Forest (Tuned)...")
    model_start = time.time()
    try:
        rf_result = train_tuned_random_forest(X_train, y_train, X_val, y_val)
        if rf_result:
            models.append(rf_result)
            model_times['Random Forest'] = time.time() - model_start
            completed_models += 1
            print(f"      ✅ AUC: {rf_result['auc']:.4f}, PR-AUC: {rf_result['pr_auc']:.4f}, Spearman: {rf_result['spearman']:.4f} ({model_times['Random Forest']/60:.1f} мин)")
            print_progress()
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
    
    # Logistic Regression с SMOTE
    print(f"   🚀 [{completed_models+1}/{model_types_count}] Обучение Logistic Regression (Tuned + SMOTE)...")
    model_start = time.time()
    try:
        lr_result = train_tuned_logistic_regression(X_train, y_train, X_val, y_val)
        if lr_result:
            models.append(lr_result)
            model_times['Logistic Regression'] = time.time() - model_start
            completed_models += 1
            print(f"      ✅ AUC: {lr_result['auc']:.4f}, PR-AUC: {lr_result['pr_auc']:.4f}, Spearman: {lr_result['spearman']:.4f} ({model_times['Logistic Regression']/60:.1f} мин)")
            print_progress()
    except Exception as e:
        print(f"      ❌ Ошибка: {e}")
    
    # Показываем общее время
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = int(total_elapsed % 60)
    print(f"\n   ⏱ Общее время обучения моделей: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_elapsed/60:.1f} минут)")
    print(f"   📊 Обучено моделей: {len(models)}")
    
    if not models:
        raise ValueError("❌ Ни одна модель не была обучена!")
    
    # 5. Создание ансамбля
    print("\n[5/7] Создание ансамбля...")
    ensemble_result = create_ensemble(models, X_val, y_val)
    if ensemble_result:
        models.append(ensemble_result)
        print(f"      ✅ Ансамбль: AUC: {ensemble_result['auc']:.4f}, PR-AUC: {ensemble_result['pr_auc']:.4f}, Spearman: {ensemble_result['spearman']:.4f}")
    
    # 6. Сравнение моделей
    print("\n[6/7] Сравнение моделей...")
    print("\n" + "=" * 80)
    print(f"{'Модель':<40} {'AUC':<10} {'PR-AUC':<10} {'Spearman':<10}")
    print("=" * 80)
    
    for model_result in models:
        spearman_val = model_result.get('spearman')
        spearman_str = f"{spearman_val:.4f}" if spearman_val is not None else "-"
        print(f"{model_result['name']:<40} {model_result['auc']:<10.4f} {model_result['pr_auc']:<10.4f} {spearman_str:<10}")
    
    # Лучшая по PR-AUC
    best_model = max(models, key=lambda x: x['pr_auc'])
    print("\n" + "=" * 80)
    print(f"🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model['name']}")
    print(f"   AUC: {best_model['auc']:.4f}")
    print(f"   PR-AUC: {best_model['pr_auc']:.4f}")
    print(f"   Spearman: {best_model['spearman']:.4f}")
    
    if best_model['pr_auc'] >= 0.80:
        print("   🎉 ЦЕЛЬ ДОСТИГНУТА! PR-AUC >= 0.80")
    elif best_model['pr_auc'] >= 0.60:
        print("   ✅ Очень хороший результат! PR-AUC >= 0.60")
    elif best_model['pr_auc'] >= 0.40:
        print("   ✅ Хороший результат! PR-AUC >= 0.40")
    else:
        print("   ⚠ Можно улучшить дальше")
    print("=" * 80)
    
    # 7. Оптимизация порога классификации (ВЫСОКИЙ ПРИОРИТЕТ)
    print("\n[7/8] Оптимизация порога классификации...")
    
    # Применяем лучшую модель к валидационной выборке для поиска порога
    if isinstance(best_model['model'], list):
        # Ансамбль
        val_predictions = []
        for model_result in best_model['model']:
            model = model_result['model']
            scaler = model_result.get('scaler')
            if scaler:
                X_val_scaled = scaler.transform(X_val)
                pred = model.predict_proba(X_val_scaled)[:, 1]
            else:
                pred = model.predict_proba(X_val)[:, 1]
            val_predictions.append(pred)
        val_pred_proba = np.average(val_predictions, axis=0, weights=best_model['weights'])
    else:
        model = best_model['model']
        scaler = best_model.get('scaler')
        if scaler:
            X_val_scaled = scaler.transform(X_val)
            val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        else:
            val_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Поиск оптимального порога для максимизации F1 (баланс precision и recall)
    from sklearn.metrics import f1_score, precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_val, val_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"      ✅ Оптимальный порог: {optimal_threshold:.4f} (вместо 0.5)")
    print(f"      Precision при этом пороге: {precision[optimal_idx]:.4f}")
    print(f"      Recall при этом пороге: {recall[optimal_idx]:.4f}")
    print(f"      F1 при этом пороге: {f1_scores[optimal_idx]:.4f}")
    
    # 8. Финальная оценка на тестовой выборке
    print("\n[8/8] Финальная оценка на тестовой выборке...")
    
    # Применяем лучшую модель к тестовой выборке
    if isinstance(best_model['model'], list):
        # Ансамбль
        predictions = []
        for i, model_result in enumerate(best_model['model']):
            model = model_result['model']
            scaler = model_result.get('scaler')
            if scaler:
                X_test_scaled = scaler.transform(X_test)
                pred = model.predict_proba(X_test_scaled)[:, 1]
            else:
                pred = model.predict_proba(X_test)[:, 1]
            predictions.append(pred)
        
        test_pred_proba = np.average(predictions, axis=0, weights=best_model['weights'])
    else:
        # Одна модель
        model = best_model['model']
        scaler = best_model.get('scaler')
        if scaler:
            X_test_scaled = scaler.transform(X_test)
            test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Используем оптимальный порог
    test_pred = (test_pred_proba >= optimal_threshold).astype(int)
    
    test_auc = roc_auc_score(y_test, test_pred_proba)
    test_pr_auc = average_precision_score(y_test, test_pred_proba)
    test_spearman, _ = spearmanr(y_test, test_pred_proba)
    test_f1 = f1_score(y_test, test_pred)
    
    print(f"   Тестовая выборка:")
    print(f"   AUC: {test_auc:.4f}")
    print(f"   PR-AUC: {test_pr_auc:.4f}")
    print(f"   Spearman: {test_spearman:.4f}")
    print(f"   F1 (порог {optimal_threshold:.4f}): {test_f1:.4f}")
    
    # 9. Сохранение
    print("\n[9/9] Сохранение результатов...")
    
    model_path = Path("models")
    model_path.mkdir(exist_ok=True)
    
    # Сохраняем лучшую модель
    with open(model_path / "best_model_advanced.pkl", "wb") as f:
        pickle.dump({
            "model": best_model["model"],
            "scaler": best_model.get("scaler"),
            "smote": best_model.get("smote"),
            "feature_cols": feature_cols,
            "model_name": best_model["name"],
            "auc": test_auc,
            "pr_auc": test_pr_auc,
            "spearman": test_spearman,
            "f1": test_f1,
            "best_params": best_model.get("best_params"),
            "weights": best_model.get("weights"),
            "optimal_threshold": optimal_threshold
        }, f)
    
    print(f"   ✅ Модель сохранена: models/best_model_advanced.pkl")
    
    # Сохраняем результаты
    results_df = pd.DataFrame([
        {
            "model": m["name"],
            "auc": m["auc"],
            "pr_auc": m["pr_auc"]
        }
        for m in models
    ])
    results_df.to_csv("models/model_comparison_advanced.csv", index=False)
    print(f"   ✅ Сравнение моделей: models/model_comparison_advanced.csv")
    
    # Сохраняем предсказания
    predictions_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": test_pred,
        "y_pred_proba": test_pred_proba
    })
    predictions_df.to_csv("models/predictions_advanced.csv", index=False)
    print(f"   ✅ Предсказания: models/predictions_advanced.csv")
    
    # Финальная статистика
    total_pipeline_time = time.time() - pipeline_start_time
    hours = int(total_pipeline_time // 3600)
    minutes = int((total_pipeline_time % 3600) // 60)
    seconds = int(total_pipeline_time % 60)
    
    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"⏱ Общее время pipeline: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"📅 Завершено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return best_model, models, X_test, y_test, feature_cols


if __name__ == "__main__":
    best_model, all_models, X_test, y_test, feature_cols = main()
def tune_catboost_with_optuna(X_train: pd.DataFrame, y_train: pd.Series):
    """Оптимизация CatBoost через Optuna для максимизации PR-AUC."""
    if not OPTUNA_AVAILABLE or not CATBOOST_AVAILABLE:
        return None, None
    print("      🎯 Optuna: начало поиска гиперпараметров CatBoost...")
    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 5, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "iterations": trial.suggest_int("iterations", 300, 800)
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            model = CatBoostClassifier(
                eval_metric="PRAUC",
                loss_function="Logloss",
                random_state=RANDOM_STATE,
                verbose=False,
                thread_count=CATBOOST_THREADS,
                **params
            )
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
            y_pred = model.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, y_pred))
        return float(np.mean(scores))
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_params = study.best_params
    print(f"      ✅ Optuna завершена. Лучшие параметры: {best_params}")
    best_model = CatBoostClassifier(
        eval_metric="PRAUC",
        loss_function="Logloss",
        random_state=RANDOM_STATE,
        thread_count=CATBOOST_THREADS,
        **best_params
    )
    best_model.fit(X_train, y_train, verbose=100)
    return best_model, study
