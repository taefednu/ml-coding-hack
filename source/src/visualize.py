"""
Визуализация результатов моделирования.

Этот скрипт создает:
1. ROC кривые
2. PR кривые
3. Confusion Matrix
4. Feature Importance plots
5. Распределения признаков
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

# Настройка визуализации
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_predictions(predictions_path: str = "models/predictions.csv"):
    """Загружает предсказания модели."""
    return pd.read_csv(predictions_path)


def load_model(model_path: str = "models/best_model.pkl"):
    """Загружает модель."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ============================================================================
# ROC И PR КРИВЫЕ
# ============================================================================

def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   save_path: str = "plots/roc_curve.png"):
    """
    Строит ROC кривую.
    """
    Path("plots").mkdir(exist_ok=True)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve: Предсказание дефолта', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")
    
    return auc


def plot_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                  save_path: str = "plots/pr_curve.png"):
    """
    Строит Precision-Recall кривую.
    """
    Path("plots").mkdir(exist_ok=True)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    # Baseline (случайный классификатор)
    baseline = y_true.mean()
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                label=f'Baseline (AP = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve: Предсказание дефолта', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")
    
    return pr_auc


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         save_path: str = "plots/confusion_matrix.png"):
    """
    Строит confusion matrix.
    """
    Path("plots").mkdir(exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'],
                cbar_kws={'label': 'Count'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    # Добавляем проценты
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j + 0.5, i + 0.7, f'\n({cm_normalized[i, j]:.1%})',
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15,
                           save_path: str = "plots/feature_importance.png"):
    """
    Строит график важности признаков.
    """
    if importance_df is None or len(importance_df) == 0:
        print("   ⚠ Данные о важности признаков недоступны")
        return
    
    Path("plots").mkdir(exist_ok=True)
    
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_features.iloc[:, 1].values)
    plt.yticks(range(len(top_features)), top_features.iloc[:, 0].values)
    plt.xlabel('Важность', fontsize=12)
    plt.title(f'TOP-{top_n} Важных Признаков', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")


# ============================================================================
# РАСПРЕДЕЛЕНИЯ ПРИЗНАКОВ
# ============================================================================

def plot_feature_distributions(df: pd.DataFrame, feature_cols: list, 
                               target_col: str = "default", top_n: int = 6,
                               save_path: str = "plots/feature_distributions.png"):
    """
    Строит распределения признаков для классов дефолта и не-дефолта.
    """
    Path("plots").mkdir(exist_ok=True)
    
    # Берем только числовые признаки
    numeric_cols = [col for col in feature_cols 
                   if pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_cols) == 0:
        print("   ⚠ Нет числовых признаков для визуализации")
        return
    
    # Берем топ признаков по вариативности
    variances = df[numeric_cols].var().sort_values(ascending=False)
    top_features = variances.head(top_n).index.tolist()
    
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Распределения для каждого класса
        df[df[target_col] == 0][feature].hist(alpha=0.5, label='No Default', 
                                               bins=30, ax=ax, color='blue')
        df[df[target_col] == 1][feature].hist(alpha=0.5, label='Default', 
                                               bins=30, ax=ax, color='red')
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'Распределение: {feature}', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Скрываем лишние subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")


# ============================================================================
# СРАВНЕНИЕ МОДЕЛЕЙ
# ============================================================================

def plot_model_comparison(comparison_path: str = "models/model_comparison.csv",
                         save_path: str = "plots/model_comparison.png"):
    """
    Строит сравнение моделей по метрикам.
    """
    Path("plots").mkdir(exist_ok=True)
    
    try:
        df = pd.read_csv(comparison_path)
    except FileNotFoundError:
        print("   ⚠ Файл сравнения моделей не найден")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC
    ax1.barh(df['model'], df['auc'])
    ax1.set_xlabel('AUC', fontsize=12)
    ax1.set_title('Сравнение моделей: AUC', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # PR-AUC
    ax2.barh(df['model'], df['pr_auc'], color='orange')
    ax2.set_xlabel('PR-AUC', fontsize=12)
    ax2.set_title('Сравнение моделей: PR-AUC', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Сохранено: {save_path}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 80)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # 1. Загрузка предсказаний
    print("\n[1/6] Загрузка предсказаний...")
    try:
        predictions_df = load_predictions()
        y_true = predictions_df['y_true'].values
        y_pred = predictions_df['y_pred'].values
        y_pred_proba = predictions_df['y_pred_proba'].values
        print(f"   ✅ Загружено {len(predictions_df)} предсказаний")
    except FileNotFoundError:
        print("   ❌ Файл с предсказаниями не найден! Сначала запустите train.py")
        return
    
    # 2. ROC кривая
    print("\n[2/6] Построение ROC кривой...")
    auc = plot_roc_curve(y_true, y_pred_proba)
    print(f"   AUC: {auc:.4f}")
    
    # 3. PR кривая
    print("\n[3/6] Построение PR кривой...")
    pr_auc = plot_pr_curve(y_true, y_pred_proba)
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    # 4. Confusion Matrix
    print("\n[4/6] Построение Confusion Matrix...")
    plot_confusion_matrix(y_true, y_pred)
    
    # 5. Feature Importance
    print("\n[5/6] Построение графиков важности признаков...")
    
    # Из SHAP
    try:
        shap_importance = pd.read_csv("models/top_features_shap.csv")
        plot_feature_importance(
            shap_importance.rename(columns={"feature": "feature", "mean_abs_shap": "importance"}),
            top_n=15,
            save_path="plots/feature_importance_shap.png"
        )
    except FileNotFoundError:
        print("   ⚠ SHAP важность не найдена")
    
    # Из модели
    try:
        model_importance = pd.read_csv("models/feature_importance_model.csv")
        plot_feature_importance(
            model_importance,
            top_n=15,
            save_path="plots/feature_importance_model.png"
        )
    except FileNotFoundError:
        print("   ⚠ Важность модели не найдена")
    
    # 6. Распределения признаков
    print("\n[6/6] Построение распределений признаков...")
    try:
        import prepare_data as prep
        master_df = prep.build_master_dataset()
        from train import prepare_features
        X, y, feature_cols = prepare_features(master_df)
        X['default'] = y
        plot_feature_distributions(X, feature_cols, top_n=6)
    except Exception as e:
        print(f"   ⚠ Ошибка при построении распределений: {e}")
    
    # 7. Сравнение моделей
    print("\n[7/7] Сравнение моделей...")
    plot_model_comparison()
    
    print("\n" + "=" * 80)
    print("✅ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("=" * 80)
    print("\nВсе графики сохранены в папке plots/")


if __name__ == "__main__":
    main()

