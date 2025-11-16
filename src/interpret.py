"""
Интерпретация модели: SHAP анализ и важность признаков.

Этот скрипт:
1. Загружает обученную модель
2. Вычисляет SHAP values
3. Определяет TOP-10 важных признаков
4. Генерирует интерпретации для бизнеса
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠ SHAP не установлен. Установите: pip install shap")

import matplotlib.pyplot as plt
import seaborn as sns

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================================

def load_model(model_path: str = "models/best_model.pkl"):
    """Загружает обученную модель."""
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    return model_data


# ============================================================================
# SHAP АНАЛИЗ
# ============================================================================

def compute_shap_values(model_data: dict, X_sample: pd.DataFrame, n_samples: int = 100):
    """
    Вычисляет SHAP values для интерпретации модели.
    
    Args:
        model_data: словарь с моделью и метаданными
        X_sample: датафрейм с признаками
        n_samples: количество образцов для SHAP (для ускорения)
    """
    if not SHAP_AVAILABLE:
        print("❌ SHAP не установлен, пропускаем анализ")
        return None
    
    model = model_data["model"]
    scaler = model_data.get("scaler")
    
    # Подготовка данных
    if scaler:
        X_processed = scaler.transform(X_sample)
    else:
        X_processed = X_sample.values
    
    # Для больших датасетов используем выборку
    if len(X_sample) > n_samples:
        sample_idx = np.random.choice(len(X_sample), n_samples, replace=False)
        X_shap = X_processed[sample_idx]
    else:
        X_shap = X_processed
    
    # Выбор explainer в зависимости от типа модели
    model_name = model_data.get("model_name", "").lower()
    
    try:
        if "tree" in str(type(model)).lower() or "forest" in str(type(model)).lower() or \
           "xgb" in model_name or "lgb" in model_name or "catboost" in model_name:
            # Tree-based модели
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            
            # Для бинарной классификации берем значения для класса 1
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            # Линейные модели и другие
            explainer = shap.LinearExplainer(model, X_shap)
            shap_values = explainer.shap_values(X_shap)
        
        return {
            "shap_values": shap_values,
            "explainer": explainer,
            "X_shap": X_shap,
            "feature_names": X_sample.columns.tolist()
        }
    except Exception as e:
        print(f"⚠ Ошибка при вычислении SHAP: {e}")
        print("   Пробуем KernelExplainer...")
        try:
            # Fallback на KernelExplainer (медленнее, но универсальнее)
            def model_predict(X):
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    X_scaled = X
                return model.predict_proba(X_scaled)[:, 1]
            
            explainer = shap.KernelExplainer(model_predict, X_shap[:50])  # Маленькая выборка для background
            shap_values = explainer.shap_values(X_shap[:100])  # Еще меньше для вычислений
            
            return {
                "shap_values": shap_values,
                "explainer": explainer,
                "X_shap": X_shap[:100],
                "feature_names": X_sample.columns.tolist()
            }
        except Exception as e2:
            print(f"❌ Не удалось вычислить SHAP: {e2}")
            return None


# ============================================================================
# ВАЖНОСТЬ ПРИЗНАКОВ
# ============================================================================

def get_feature_importance(model_data: dict, feature_cols: list, method: str = "shap"):
    """
    Получает важность признаков.
    
    Args:
        model_data: словарь с моделью
        feature_cols: список имен признаков
        method: "shap" или "model" (встроенная важность модели)
    """
    model = model_data["model"]
    model_name = model_data.get("model_name", "").lower()
    
    # Пробуем встроенную важность модели
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        return importance_df
    
    # Если нет встроенной важности, возвращаем None
    return None


def get_top_features_shap(shap_result: dict, top_n: int = 10):
    """
    Извлекает TOP-N признаков по среднему абсолютному SHAP значению.
    """
    if shap_result is None:
        return None
    
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    
    # Среднее абсолютное значение SHAP для каждого признака
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values("mean_abs_shap", ascending=False)
    
    return importance_df.head(top_n)


# ============================================================================
# БИЗНЕС-ИНТЕРПРЕТАЦИЯ
# ============================================================================

def interpret_feature_impact(shap_result: dict, feature_name: str, X_sample: pd.DataFrame):
    """
    Интерпретирует влияние признака на предсказание человеческим языком.
    """
    if shap_result is None:
        return None
    
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    
    if feature_name not in feature_names:
        return None
    
    idx = feature_names.index(feature_name)
    feature_shap = shap_values[:, idx]
    feature_values = X_sample[feature_name].values[:len(feature_shap)]
    
    # Анализ
    mean_shap = feature_shap.mean()
    positive_impact = (feature_shap > 0).sum()
    negative_impact = (feature_shap < 0).sum()
    
    # Корреляция между значением признака и SHAP
    correlation = np.corrcoef(feature_values, feature_shap)[0, 1]
    
    interpretation = {
        "feature": feature_name,
        "mean_impact": mean_shap,
        "positive_cases": positive_impact,
        "negative_cases": negative_impact,
        "correlation": correlation,
        "description": ""
    }
    
    # Генерация описания
    if mean_shap > 0:
        direction = "увеличивает"
    else:
        direction = "уменьшает"
    
    if abs(correlation) > 0.3:
        if correlation > 0:
            relationship = "Чем выше значение признака, тем выше риск дефолта"
        else:
            relationship = "Чем выше значение признака, тем ниже риск дефолта"
    else:
        relationship = "Влияние нелинейное"
    
    interpretation["description"] = (
        f"Признак '{feature_name}' в среднем {direction} вероятность дефолта на {abs(mean_shap):.4f}. "
        f"{relationship}."
    )
    
    return interpretation


def generate_business_insights(shap_result: dict, top_features: pd.DataFrame, 
                               X_sample: pd.DataFrame) -> list:
    """
    Генерирует бизнес-выводы на основе SHAP анализа.
    """
    insights = []
    
    if shap_result is None or top_features is None:
        return ["⚠ SHAP анализ недоступен"]
    
    insights.append("=" * 80)
    insights.append("БИЗНЕС-ВЫВОДЫ: Топ-10 факторов риска дефолта")
    insights.append("=" * 80)
    insights.append("")
    
    for idx, row in top_features.iterrows():
        feature = row["feature"]
        importance = row["mean_abs_shap"]
        
        interpretation = interpret_feature_impact(shap_result, feature, X_sample)
        
        if interpretation:
            insights.append(f"{idx + 1}. {feature}")
            insights.append(f"   Важность: {importance:.4f}")
            insights.append(f"   {interpretation['description']}")
            insights.append("")
    
    return insights


# ============================================================================
# ВИЗУАЛИЗАЦИЯ SHAP
# ============================================================================

def plot_shap_summary(shap_result: dict, top_n: int = 20, save_path: str = "plots/shap_summary.png"):
    """
    Строит summary plot для SHAP значений.
    """
    if shap_result is None:
        print("⚠ SHAP результаты недоступны для визуализации")
        return
    
    Path("plots").mkdir(exist_ok=True)
    
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    X_shap = shap_result["X_shap"]
    
    # Создаем DataFrame для SHAP
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    
    # Берем TOP-N признаков
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)
    top_features = mean_abs.head(top_n).index.tolist()
    
    shap_values_top = shap_values[:, [feature_names.index(f) for f in top_features]]
    
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_top,
            X_shap[:, [feature_names.index(f) for f in top_features]],
            feature_names=top_features,
            show=False,
            max_display=top_n
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✅ Сохранено: {save_path}")
    except Exception as e:
        print(f"⚠ Ошибка при построении summary plot: {e}")


def plot_shap_bar(shap_result: dict, top_n: int = 15, save_path: str = "plots/shap_bar.png"):
    """
    Строит bar plot средних абсолютных SHAP значений.
    """
    if shap_result is None:
        return
    
    Path("plots").mkdir(exist_ok=True)
    
    shap_values = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(importance_df)), importance_df["importance"].values)
    plt.yticks(range(len(importance_df)), importance_df["feature"].values)
    plt.xlabel("Среднее абсолютное SHAP значение")
    plt.title(f"TOP-{top_n} Важных признаков (SHAP)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 80)
    print("ИНТЕРПРЕТАЦИЯ МОДЕЛИ: SHAP анализ")
    print("=" * 80)
    
    # 1. Загрузка модели
    print("\n[1/4] Загрузка модели...")
    try:
        model_data = load_model()
        print(f"   ✅ Модель загружена: {model_data.get('model_name', 'Unknown')}")
        print(f"   AUC: {model_data.get('auc', 'N/A'):.4f}")
        print(f"   PR-AUC: {model_data.get('pr_auc', 'N/A'):.4f}")
    except FileNotFoundError:
        print("   ❌ Модель не найдена! Сначала запустите train.py")
        return
    
    # 2. Загрузка данных для интерпретации
    print("\n[2/4] Загрузка данных...")
    import prepare_data as prep
    master_df = prep.build_master_dataset()
    
    # Подготовка фичей (как в train.py)
    from train import prepare_features
    X, y, feature_cols = prepare_features(master_df)
    
    # Используем выборку для SHAP (для ускорения)
    sample_size = min(500, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    print(f"   ✅ Используем выборку: {len(X_sample)} строк")
    
    # 3. SHAP анализ
    print("\n[3/4] Вычисление SHAP values...")
    if not SHAP_AVAILABLE:
        print("   ⚠ SHAP не установлен. Установите: pip install shap")
        shap_result = None
    else:
        shap_result = compute_shap_values(model_data, X_sample, n_samples=200)
        if shap_result:
            print("   ✅ SHAP values вычислены")
        else:
            print("   ⚠ Не удалось вычислить SHAP values")
    
    # 4. Извлечение важных признаков
    print("\n[4/4] Анализ важности признаков...")
    
    # Через SHAP
    if shap_result:
        top_features_shap = get_top_features_shap(shap_result, top_n=10)
        if top_features_shap is not None:
            print("\n   TOP-10 признаков по SHAP:")
            print("   " + "-" * 70)
            for idx, row in top_features_shap.iterrows():
                print(f"   {idx + 1:2d}. {row['feature']:<40} {row['mean_abs_shap']:.4f}")
    
    # Через встроенную важность модели
    feature_importance = get_feature_importance(model_data, feature_cols)
    if feature_importance is not None:
        print("\n   TOP-10 признаков по встроенной важности модели:")
        print("   " + "-" * 70)
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   {idx + 1:2d}. {row['feature']:<40} {row['importance']:.4f}")
    
    # 5. Генерация бизнес-выводов
    print("\n" + "=" * 80)
    print("БИЗНЕС-ВЫВОДЫ")
    print("=" * 80)
    
    if shap_result and top_features_shap is not None:
        insights = generate_business_insights(shap_result, top_features_shap, X_sample)
        for insight in insights:
            print(insight)
        
        # Сохраняем выводы
        with open("models/business_insights.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(insights))
        print("\n   ✅ Бизнес-выводы сохранены: models/business_insights.txt")
    
    # 6. Визуализация
    print("\n[5/5] Создание визуализаций...")
    if shap_result:
        plot_shap_summary(shap_result, top_n=15, save_path="plots/shap_summary.png")
        plot_shap_bar(shap_result, top_n=15, save_path="plots/shap_bar.png")
    
    # Сохраняем топ признаки
    if shap_result and top_features_shap is not None:
        top_features_shap.to_csv("models/top_features_shap.csv", index=False)
        print("   ✅ Топ признаки сохранены: models/top_features_shap.csv")
    
    if feature_importance is not None:
        feature_importance.to_csv("models/feature_importance_model.csv", index=False)
        print("   ✅ Важность признаков сохранена: models/feature_importance_model.csv")
    
    print("\n" + "=" * 80)
    print("✅ ИНТЕРПРЕТАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)
    
    return shap_result, top_features_shap, feature_importance


if __name__ == "__main__":
    shap_result, top_features, feature_importance = main()

