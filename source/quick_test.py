#!/usr/bin/env python3
"""
Быстрый тест модели на тестовых данных с полными результатами
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("🚀 БЫСТРЫЙ ТЕСТ НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 80)
    
    # Загрузка модели
    print("\n📦 Загрузка модели...")
    model_path = Path('models/best_model_optimized.pkl')
    
    if not model_path.exists():
        print("❌ Модель не найдена! Запустите сначала обучение.")
        return
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Проверка структуры модели
    if isinstance(model_dict, dict):
        print(f"✅ Модель загружена: {model_path}")
        print(f"  Структура: {list(model_dict.keys())}")
        
        if 'meta_model' in model_dict:
            model = model_dict['meta_model']
            feature_cols = model_dict.get('feature_cols', None)
            selected_features = model_dict.get('selected_features', None)
            scaler = model_dict.get('scaler', None)
            optimal_threshold = model_dict.get('optimal_threshold', 0.5)
        else:
            print("❌ meta_model не найден в словаре модели!")
            return
    else:
        model = model_dict
        feature_cols = None
        selected_features = None
        scaler = None
        optimal_threshold = 0.5
        print(f"✅ Модель загружена: {model_path}")
    
    # Загрузка тестовых данных
    print("\n📊 Загрузка тестовых данных...")
    test_dir = Path('test_data')
    
    # Проверка наличия файлов
    required_files = [
        'application_metadata.csv',
        'demographics.csv',
        'credit_hystory.csv',
        'financial_ratios.jsonl',
        'geographic_data.xml'
    ]
    
    missing_files = [f for f in required_files if not (test_dir / f).exists()]
    if missing_files:
        print(f"⚠️  Отсутствуют файлы: {missing_files}")
        print("Попытка использовать доступные файлы...")
    
    # Загрузка данных
    try:
        # Application metadata
        app = pd.read_csv(test_dir / 'application_metadata.csv')
        print(f"  ✓ application_metadata: {len(app):,} записей")
        
        # Demographics
        demo = pd.read_csv(test_dir / 'demographics.csv')
        print(f"  ✓ demographics: {len(demo):,} записей")
        
        # Credit history
        if (test_dir / 'credit_hystory.csv').exists():
            credit = pd.read_csv(test_dir / 'credit_hystory.csv')
        else:
            credit = pd.read_csv(test_dir / 'credit_history.csv')
        print(f"  ✓ credit_history: {len(credit):,} записей")
        
        # Financial ratios
        financial = pd.read_json(test_dir / 'financial_ratios.jsonl', lines=True)
        print(f"  ✓ financial_ratios: {len(financial):,} записей")
        
        # Geographic data
        geographic = pd.read_xml(test_dir / 'geographic_data.xml')
        print(f"  ✓ geographic_data: {len(geographic):,} записей")
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return
    
    # Объединение данных
    print("\n🔗 Объединение данных...")
    df = app.copy()
    
    # Rename ID columns to customer_ref
    if 'customer_ref' not in demo.columns:
        if 'cust_id' in demo.columns:
            demo = demo.rename(columns={'cust_id': 'customer_ref'})
        elif 'id' in demo.columns:
            demo = demo.rename(columns={'id': 'customer_ref'})
    
    if 'customer_ref' not in credit.columns:
        if 'customer_number' in credit.columns:
            credit = credit.rename(columns={'customer_number': 'customer_ref'})
        elif 'cust_id' in credit.columns:
            credit = credit.rename(columns={'cust_id': 'customer_ref'})
    
    if 'customer_ref' not in financial.columns:
        if 'cust_num' in financial.columns:
            financial = financial.rename(columns={'cust_num': 'customer_ref'})
        elif 'cust_id' in financial.columns:
            financial = financial.rename(columns={'cust_id': 'customer_ref'})
    
    if 'customer_ref' not in geographic.columns:
        if 'id' in geographic.columns:
            geographic = geographic.rename(columns={'id': 'customer_ref'})
        elif 'cust_id' in geographic.columns:
            geographic = geographic.rename(columns={'cust_id': 'customer_ref'})
    
    # Merge all sources
    df = df.merge(demo, on='customer_ref', how='left', suffixes=('', '_demo'))
    df = df.merge(credit, on='customer_ref', how='left', suffixes=('', '_credit'))
    df = df.merge(financial, on='customer_ref', how='left', suffixes=('', '_fin'))
    df = df.merge(geographic, on='customer_ref', how='left', suffixes=('', '_geo'))
    
    print(f"✅ Объединено: {len(df):,} записей, {len(df.columns)} колонок")
    
    # Сохранение customer_ref для результатов
    customer_ids = df['customer_ref'].copy()
    
    # Удаление ненужных колонок
    drop_cols = ['customer_ref', 'default_flag', 'default', 'target', 'label']
    drop_cols = [col for col in drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  Удалены колонки: {drop_cols}")
    
    # Заполнение пропусков
    print("\n🔧 Обработка пропусков...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
    
    # Удаление всех нечисловых колонок для модели
    df = df.select_dtypes(include=[np.number])
    
    print(f"✅ Пропуски заполнены, оставлены только числовые признаки")
    
    # Получение признаков из модели
    print("\n🎯 Подготовка признаков для предсказания...")
    
    # Используем признаки из модели если доступны
    if selected_features is not None:
        expected_features = selected_features
        print(f"  Используются selected_features из модели")
    elif feature_cols is not None:
        expected_features = feature_cols
        print(f"  Используются feature_cols из модели")
    else:
        # Получаем имена признаков из модели
        try:
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
            elif hasattr(model, 'feature_name_'):
                expected_features = model.feature_name_
            else:
                print("⚠️  Не удалось получить список признаков из модели")
                print("Используем все числовые колонки...")
                expected_features = df.select_dtypes(include=[np.number]).columns.tolist()
        except:
            expected_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Проверка наличия всех необходимых признаков
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        print(f"⚠️  Отсутствующие признаки ({len(missing_features)}): {list(missing_features)[:5]}...")
        # Создаем недостающие признаки со значением 0
        for feat in missing_features:
            df[feat] = 0
        print("  Созданы с нулевыми значениями")
    
    # Выбираем только нужные признаки
    X_test = df[expected_features].copy()
    
    # Применяем scaler если есть
    if scaler is not None:
        print(f"  Применение нормализации (scaler)...")
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
    
    print(f"✅ Подготовлено {len(expected_features)} признаков для {len(X_test):,} клиентов")
    
    # Предсказание
    print("\n🤖 Выполнение предсказаний...")
    try:
        # Проверяем, есть ли базовые модели для стекинга
        if 'models' in model_dict and model_dict['models']:
            base_models = model_dict['models']
            print(f"  Используется стекинг из {len(base_models)} базовых моделей")
            
            # Получаем предсказания от базовых моделей
            base_predictions = []
            for i, base_model in enumerate(base_models):
                if hasattr(base_model, 'predict_proba'):
                    preds = base_model.predict_proba(X_test)[:, 1]
                else:
                    preds = base_model.predict(X_test)
                base_predictions.append(preds)
            
            # Создаем мета-признаки
            X_meta = np.column_stack(base_predictions)
            print(f"  Мета-признаки: {X_meta.shape}")
            
            # Предсказание мета-моделью
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_meta)[:, 1]
            else:
                probabilities = model.predict(X_meta)
        else:
            # Прямое предсказание
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_test)[:, 1]
            else:
                probabilities = model.predict(X_test)
        
        print(f"✅ Предсказания выполнены для {len(probabilities):,} клиентов")
    except Exception as e:
        print(f"❌ Ошибка при предсказании: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Создание результатов
    print("\n📝 Формирование результатов...")
    results = pd.DataFrame({
        'customer_ref': customer_ids,
        'probability': probabilities
    })
    
    # Сортировка по вероятности (от высокой к низкой)
    results = results.sort_values('probability', ascending=False).reset_index(drop=True)
    
    # Сохранение результатов
    output_file = test_dir / 'predictions.csv'
    results.to_csv(output_file, index=False)
    print(f"✅ Результаты сохранены: {output_file}")
    
    # Анализ результатов
    print("\n" + "=" * 80)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    print(f"\n🎯 Всего клиентов: {len(results):,}")
    print(f"\n📈 Статистика вероятностей дефолта:")
    print(f"  Минимум:     {results['probability'].min():.6f}")
    print(f"  Максимум:    {results['probability'].max():.6f}")
    print(f"  Среднее:     {results['probability'].mean():.6f}")
    print(f"  Медиана:     {results['probability'].median():.6f}")
    print(f"  Std:         {results['probability'].std():.6f}")
    
    print(f"\n📊 Квантили:")
    for q in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        val = results['probability'].quantile(q)
        print(f"  {int(q*100):2d}%:          {val:.6f}")
    
    print(f"\n🎲 Распределение по порогам:")
    print(f"{'Порог':<12} {'Дефолтов':<12} {'Процент':<12}")
    print("-" * 40)
    
    thresholds = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    for threshold in thresholds:
        count = (results['probability'] >= threshold).sum()
        pct = count / len(results) * 100
        print(f"{threshold:<12.4f} {count:<12,} {pct:<12.2f}%")
    
    # Топ-20 рискованных клиентов
    print(f"\n🔴 ТОП-20 САМЫХ РИСКОВАННЫХ КЛИЕНТОВ:")
    print(f"{'Rank':<6} {'Customer ID':<15} {'Вероятность':<15}")
    print("-" * 40)
    for idx, row in results.head(20).iterrows():
        print(f"{idx+1:<6} {row['customer_ref']:<15} {row['probability']:<15.6f}")
    
    # Топ-20 самых безопасных клиентов
    print(f"\n🟢 ТОП-20 САМЫХ БЕЗОПАСНЫХ КЛИЕНТОВ:")
    print(f"{'Rank':<6} {'Customer ID':<15} {'Вероятность':<15}")
    print("-" * 40)
    for idx, row in results.tail(20).iterrows():
        rank = len(results) - idx
        print(f"{rank:<6} {row['customer_ref']:<15} {row['probability']:<15.6f}")
    
    # Рекомендации
    print("\n" + "=" * 80)
    print("💡 РЕКОМЕНДАЦИИ ПО ИСПОЛЬЗОВАНИЮ")
    print("=" * 80)
    
    median_prob = results['probability'].median()
    mean_prob = results['probability'].mean()
    
    if mean_prob < 0.10:
        recommended_threshold = 0.05
    elif mean_prob < 0.20:
        recommended_threshold = 0.10
    else:
        recommended_threshold = 0.20
    
    high_risk = (results['probability'] >= recommended_threshold).sum()
    high_risk_pct = high_risk / len(results) * 100
    
    print(f"\n1️⃣ Рекомендуемый порог для отклонения заявок: {recommended_threshold:.4f}")
    print(f"   При этом пороге будет отклонено: {high_risk:,} заявок ({high_risk_pct:.2f}%)")
    
    print(f"\n2️⃣ Средняя вероятность дефолта в портфеле: {mean_prob:.4f} ({mean_prob*100:.2f}%)")
    
    if mean_prob > 0.15:
        print("   ⚠️  Высокий уровень риска в портфеле!")
    elif mean_prob > 0.10:
        print("   ⚠️  Умеренный уровень риска")
    else:
        print("   ✅ Приемлемый уровень риска")
    
    print(f"\n3️⃣ Распределение риска:")
    low_risk = (results['probability'] < 0.10).sum()
    medium_risk = ((results['probability'] >= 0.10) & (results['probability'] < 0.30)).sum()
    high_risk_final = (results['probability'] >= 0.30).sum()
    
    print(f"   Низкий риск (<10%):     {low_risk:,} клиентов ({low_risk/len(results)*100:.1f}%)")
    print(f"   Средний риск (10-30%):  {medium_risk:,} клиентов ({medium_risk/len(results)*100:.1f}%)")
    print(f"   Высокий риск (>30%):    {high_risk_final:,} клиентов ({high_risk_final/len(results)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("=" * 80)
    print(f"\n📁 Результаты сохранены в: {output_file}")
    print(f"📊 Всего обработано: {len(results):,} клиентов")
    print()

if __name__ == '__main__':
    main()
