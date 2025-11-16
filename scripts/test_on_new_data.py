"""
Тестирование обученной модели на новых данных

ИСПОЛЬЗОВАНИЕ:
1. Положите новые данные в папку test_data/
2. Запустите: python test_on_new_data.py
3. Результаты сохранятся в test_data/predictions.csv

ТРЕБУЕМЫЕ ФАЙЛЫ (аналогично тренировочным):
- test_data/application_metadata.csv
- test_data/credit_hystory.csv (опционально)
- test_data/demographics.csv (опционально)
- test_data/financial_ratios.jsonl (опционально)
- test_data/geographic_data.xml (опционально)

ИЛИ уже готовый:
- test_data/master_dataset.csv
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

import prepare_data as prep
from train_ultimate import create_ultimate_features, handle_missing_and_encode

def load_test_data():
    """Загрузка новых тестовых данных."""
    test_path = Path("test_data")
    
    # Вариант 1: Готовый master_dataset
    master_file = test_path / "master_dataset.csv"
    if master_file.exists():
        print("✅ Загружен готовый master_dataset.csv")
        return pd.read_csv(master_file)
    
    # Вариант 2: Собираем из отдельных файлов  
    print("🔧 Собираем данные из отдельных файлов...")
    
    # Временно меняем BASE_PATH в prepare_data
    original_base_path = prep.BASE_PATH
    try:
        prep.BASE_PATH = test_path
        master_df = prep.build_master_dataset()
        
        # Загружаем эталонные результаты если есть
        results_file = test_path / "results.csv"
        if results_file.exists():
            print("✅ Найден файл results.csv с эталонными результатами")
            results_df = pd.read_csv(results_file)
            
            # Джойним по customer_id
            if 'customer_id' in results_df.columns and 'customer_id' in master_df.columns:
                master_df = master_df.merge(
                    results_df[['customer_id', 'default']], 
                    on='customer_id',
                    how='left'
                )
                print(f"   ✅ Объединено с results.csv по customer_id")
        
        return master_df
    finally:
        prep.BASE_PATH = original_base_path


def load_best_model():
    """Загрузка лучшей обученной модели."""
    model_files = [
        "models/best_model_optimized.pkl",  # Приоритет: модель из train_max_auc.py
        "models/best_model_normalized.pkl",
        "models/best_model_final.pkl",
        "models/best_model_ultimate.pkl",
        "models/best_model_advanced.pkl"
    ]
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Загружена модель: {model_file}")
            with open(model_file, "rb") as f:
                return pickle.load(f), model_file
    
    raise FileNotFoundError(
        "Модель не найдена! Сначала обучите модель:\n"
        "  make train\n"
        "  или\n"
        "  python scripts/train_max_auc.py"
    )


def predict_on_new_data(test_df, model_data, selected_features=None):
    """Применение модели к новым данным."""
    
    # Feature engineering (те же преобразования что и при обучении)
    print("🔧 Feature Engineering...")
    test_featured = create_ultimate_features(test_df)
    
    # Обработка (кодирование, заполнение пропусков, нормализация)
    scaler = model_data.get('scaler')
    X_test, y_test, feature_cols, _ = handle_missing_and_encode(test_featured, scaler=None)  # Сначала без scaler
    
    # Используем те же признаки что и при обучении
    train_features = model_data.get('feature_cols', model_data.get('selected_features'))
    if train_features is not None:
        print(f"🔧 Выравнивание признаков: {len(X_test.columns)} → {len(train_features)}")
        
        # Добавляем недостающие признаки с нулями
        missing_features = set(train_features) - set(X_test.columns)
        if missing_features:
            print(f"   ⚠ Добавляем {len(missing_features)} недостающих признаков")
            for feat in missing_features:
                X_test[feat] = 0
        
        # Удаляем лишние признаки
        extra_features = set(X_test.columns) - set(train_features)
        if extra_features:
            print(f"   ⚠ Удаляем {len(extra_features)} лишних признаков")
            X_test = X_test.drop(columns=list(extra_features))
        
        # Переупорядочиваем в том же порядке
        X_test = X_test[train_features]
    
    # Применяем нормализацию
    if scaler is not None:
        print(f"   🔧 Применяем нормализацию (StandardScaler)")
        X_test_scaled = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Предсказания
    models = model_data.get('models')
    meta_model = model_data.get('meta_model')
    strategy = model_data.get('strategy', 'averaging')
    optimal_weights = model_data.get('optimal_weights')  # Для старых моделей
    optimal_threshold = model_data.get('optimal_threshold', 0.5)
    
    if models is None:
        raise ValueError("В модели нет ансамбля!")
    
    print(f"🎯 Применяем ансамбль из {len(models)} моделей...")
    
    # Предсказания каждой модели
    predictions = []
    for i, model in enumerate(models):
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(X_test)[:, 1]
        else:
            pred = model.predict(X_test)
        predictions.append(pred)
        print(f"   ✅ Модель {i+1}/{len(models)}")
    
    # Применяем стратегию (стекинг или усреднение)
    if strategy == 'stacking' and meta_model is not None:
        # Стекинг: используем мета-модель
        meta_features = np.array(predictions).T
        y_pred_proba = meta_model.predict_proba(meta_features)[:, 1]
        print(f"   ✅ Стекинг (meta-model)")
    elif optimal_weights is not None:
        # Взвешенное усреднение (для старых моделей)
        y_pred_proba = np.average(predictions, axis=0, weights=optimal_weights)
        print(f"   ✅ Взвешенное усреднение: {optimal_weights}")
    else:
        # Простое усреднение
        y_pred_proba = np.mean(predictions, axis=0)
        print(f"   ✅ Усреднение ({strategy})")
    
    # Бинарные предсказания
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    print(f"   ✅ Порог: {optimal_threshold:.4f}")
    
    return y_pred, y_pred_proba, y_test


def evaluate_predictions(y_true, y_pred, y_pred_proba, has_labels=True):
    """Оценка качества предсказаний."""
    
    if not has_labels:
        print("\n" + "="*80)
        print("⚠ Целевая переменная 'default' отсутствует - показываем только предсказания")
        print("="*80)
        
        print(f"\nРаспределение предсказаний:")
        print(pd.Series(y_pred).value_counts(normalize=True))
        print(f"\nСредняя вероятность дефолта: {y_pred_proba.mean():.4f}")
        print(f"Медианная вероятность: {np.median(y_pred_proba):.4f}")
        return
    
    # Фильтруем только те строки где есть метка
    valid_mask = ~y_true.isna()
    if valid_mask.sum() == 0:
        print("\n⚠ Нет валидных меток для оценки - только предсказания")
        has_labels = False
        evaluate_predictions(y_true, y_pred, y_pred_proba, has_labels=False)
        return
    
    if valid_mask.sum() < len(y_true):
        print(f"\n⚠ Только {valid_mask.sum()} из {len(y_true)} имеют метки - оцениваем только их")
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        y_pred_proba = y_pred_proba[valid_mask]
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ НА НОВЫХ ТЕСТОВЫХ ДАННЫХ")
    print("="*80)
    
    # Основные метрики
    auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"\n{'Метрика':<25} {'Значение':<10}")
    print("="*35)
    print(f"{'AUC-ROC':<25} {auc:<10.4f}")
    print(f"{'PR-AUC (ГЛАВНАЯ)':<25} {pr_auc:<10.4f}")
    print(f"{'F1-Score':<25} {f1:<10.4f}")
    print(f"{'F2-Score':<25} {f2:<10.4f}")
    print(f"{'Matthews Correlation':<25} {mcc:<10.4f}")
    print("="*35)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nМатрица ошибок:")
    print(f"  TN: {cm[0,0]:>6}  |  FP: {cm[0,1]:>6}")
    print(f"  FN: {cm[1,0]:>6}  |  TP: {cm[1,1]:>6}")
    
    # Classification Report
    print(f"\nДетальный отчет:")
    print(classification_report(y_true, y_pred, target_names=['No Default', 'Default']))
    
    # Оптимальный порог для данных
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-6)
    optimal_idx = np.argmax(f2_scores)
    optimal_new_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"\n💡 Рекомендуемый порог для этих данных: {optimal_new_threshold:.4f}")
    print(f"   (Precision: {precision[optimal_idx]:.4f}, Recall: {recall[optimal_idx]:.4f})")


def main():
    print("="*80)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ НА НОВЫХ ДАННЫХ")
    print("="*80)
    
    # 1. Загрузка тестовых данных
    print("\n[1/4] Загрузка тестовых данных...")
    try:
        test_df = load_test_data()
        print(f"   ✅ Загружено: {test_df.shape[0]} строк, {test_df.shape[1]} колонок")
        
        # Проверяем наличие целевой переменной
        has_labels = 'default' in test_df.columns
        if has_labels:
            print(f"   ✅ Целевая переменная найдена. Доля дефолтов: {test_df['default'].mean():.2%}")
        else:
            print(f"   ⚠ Целевая переменная 'default' отсутствует - только предсказания")
            
    except Exception as e:
        print(f"   ❌ Ошибка загрузки данных: {e}")
        return
    
    # 2. Загрузка модели
    print("\n[2/4] Загрузка обученной модели...")
    try:
        model_data, model_file = load_best_model()
        selected_features = model_data.get('selected_features')
        print(f"   ✅ PR-AUC модели: {model_data.get('pr_auc', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки модели: {e}")
        return
    
    # 3. Предсказания
    print("\n[3/4] Применение модели...")
    try:
        y_pred, y_pred_proba, y_test = predict_on_new_data(
            test_df, model_data, selected_features
        )
        print(f"   ✅ Предсказания получены для {len(y_pred)} объектов")
    except Exception as e:
        print(f"   ❌ Ошибка предсказания: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Оценка
    print("\n[4/4] Оценка результатов...")
    evaluate_predictions(y_test, y_pred, y_pred_proba, has_labels=has_labels)
    
    # 5. Сохранение
    print("\n[Сохранение результатов...]")
    output_df = pd.DataFrame({
        'prediction': y_pred,
        'probability': y_pred_proba
    })
    
    if has_labels:
        output_df['actual'] = y_test.values
    
    # Добавляем ID если есть
    if 'customer_id' in test_df.columns:
        output_df['customer_id'] = test_df['customer_id'].values
    elif 'application_id' in test_df.columns:
        output_df['application_id'] = test_df['application_id'].values
    
    output_path = Path("test_data") / "predictions.csv"
    output_df.to_csv(output_path, index=False)
    print(f"✅ Предсказания сохранены: {output_path}")
    
    print("\n" + "="*80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*80)


if __name__ == "__main__":
    main()
