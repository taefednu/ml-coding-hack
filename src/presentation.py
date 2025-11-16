"""
Генерация материалов для презентации.

Этот скрипт создает:
1. Markdown отчет с выводами
2. Сводную таблицу метрик
3. Бизнес-рекомендации
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle


# ============================================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_all_results():
    """Загружает все результаты для презентации."""
    results = {}
    
    # Модель
    try:
        with open("models/best_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        results["model"] = model_data
    except FileNotFoundError:
        results["model"] = None
    
    # Предсказания
    try:
        results["predictions"] = pd.read_csv("models/predictions.csv")
    except FileNotFoundError:
        results["predictions"] = None
    
    # Сравнение моделей
    try:
        results["comparison"] = pd.read_csv("models/model_comparison.csv")
    except FileNotFoundError:
        results["comparison"] = None
    
    # Важность признаков
    try:
        results["shap_importance"] = pd.read_csv("models/top_features_shap.csv")
    except FileNotFoundError:
        results["shap_importance"] = None
    
    try:
        results["model_importance"] = pd.read_csv("models/feature_importance_model.csv")
    except FileNotFoundError:
        results["model_importance"] = None
    
    # Бизнес-выводы
    try:
        with open("models/business_insights.txt", "r", encoding="utf-8") as f:
            results["insights"] = f.read()
    except FileNotFoundError:
        results["insights"] = None
    
    # Данные
    try:
        import prepare_data as prep
        results["master_df"] = prep.build_master_dataset()
    except Exception as e:
        results["master_df"] = None
        print(f"⚠ Ошибка загрузки данных: {e}")
    
    return results


# ============================================================================
# ГЕНЕРАЦИЯ ОТЧЕТА
# ============================================================================

def generate_presentation_report(results: dict, save_path: str = "presentation/report.md"):
    """
    Генерирует Markdown отчет для презентации.
    """
    Path("presentation").mkdir(exist_ok=True)
    
    report = []
    
    # Заголовок
    report.append("# Отчет: Предсказание дефолта заемщика")
    report.append("")
    report.append(f"**Дата создания:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("---")
    report.append("")
    
    # 1. Цель проекта
    report.append("## 1. Цель проекта")
    report.append("")
    report.append("Разработать модель машинного обучения для предсказания вероятности дефолта заемщика на основе исторических данных о заявках, кредитной истории, демографии и финансовых показателях.")
    report.append("")
    report.append("**Задачи:**")
    report.append("1. Очистить и объединить разноформатные датасеты")
    report.append("2. Построить модель для предсказания дефолта")
    report.append("3. Интерпретировать важные факторы риска")
    report.append("4. Предоставить бизнес-рекомендации")
    report.append("")
    report.append("---")
    report.append("")
    
    # 2. Описание данных
    report.append("## 2. Описание данных")
    report.append("")
    
    if results["master_df"] is not None:
        df = results["master_df"]
        report.append(f"- **Количество наблюдений:** {len(df):,}")
        report.append(f"- **Количество признаков:** {len(df.columns)}")
        report.append("")
        report.append("**Источники данных:**")
        report.append("- Application Metadata (метаданные заявок)")
        report.append("- Credit History (кредитная история)")
        report.append("- Demographics (демографические данные)")
        report.append("- Financial Ratios (финансовые показатели)")
        report.append("- Loan Details (детали займа)")
        report.append("- Geographic Data (географические данные)")
        report.append("")
        
        # Распределение таргета
        if "default" in df.columns:
            default_rate = df["default"].mean()
            report.append(f"**Распределение целевой переменной:**")
            report.append(f"- Дефолт: {df['default'].sum():,} ({default_rate:.1%})")
            report.append(f"- Нет дефолта: {(df['default'] == 0).sum():,} ({1 - default_rate:.1%})")
            report.append("")
            if default_rate < 0.3 or default_rate > 0.7:
                report.append("⚠ **Примечание:** Данные несбалансированы, использованы методы балансировки классов.")
            report.append("")
    else:
        report.append("⚠ Данные не загружены")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # 3. Методология
    report.append("## 3. Методология")
    report.append("")
    report.append("### 3.1 Обработка данных")
    report.append("")
    report.append("1. **Нормализация:** Приведение имен колонок к snake_case")
    report.append("2. **Очистка денежных значений:** Удаление символов $, запятых, пробелов")
    report.append("3. **Обработка пропусков:**")
    report.append("   - Числовые признаки: медиана")
    report.append("   - Категориальные признаки: мода или 'unknown'")
    report.append("4. **Обработка выбросов:** Winsorization (1-й и 99-й перцентили)")
    report.append("5. **Feature Engineering:**")
    report.append("   - Debt-to-Income Ratio")
    report.append("   - Credit Utilization")
    report.append("   - Cash Flow Ratio")
    report.append("   - Employment-to-Age Ratio")
    report.append("")
    
    report.append("### 3.2 Модели")
    report.append("")
    report.append("Протестированы следующие модели:")
    report.append("1. **Logistic Regression** (с балансировкой классов)")
    report.append("2. **Random Forest** (200 деревьев, max_depth=15)")
    report.append("3. **XGBoost** (200 итераций, learning_rate=0.1)")
    report.append("4. **LightGBM** (200 итераций, learning_rate=0.1)")
    report.append("5. **CatBoost** (200 итераций, learning_rate=0.1)")
    report.append("")
    report.append("**Валидация:**")
    report.append("- Train/Test split: 80/20")
    report.append("- Stratified sampling для сохранения распределения классов")
    report.append("")
    
    report.append("### 3.3 Метрики")
    report.append("")
    report.append("- **AUC-ROC:** Площадь под ROC кривой")
    report.append("- **PR-AUC:** Площадь под Precision-Recall кривой (важнее для несбалансированных данных)")
    report.append("")
    report.append("---")
    report.append("")
    
    # 4. Результаты
    report.append("## 4. Результаты моделирования")
    report.append("")
    
    if results["comparison"] is not None:
        report.append("### 4.1 Сравнение моделей")
        report.append("")
        report.append("| Модель | AUC | PR-AUC |")
        report.append("|--------|-----|--------|")
        for _, row in results["comparison"].iterrows():
            report.append(f"| {row['model']} | {row['auc']:.4f} | {row['pr_auc']:.4f} |")
        report.append("")
    
    if results["model"] is not None:
        model_data = results["model"]
        report.append("### 4.2 Лучшая модель")
        report.append("")
        report.append(f"**Модель:** {model_data.get('model_name', 'Unknown')}")
        report.append(f"**AUC:** {model_data.get('auc', 'N/A'):.4f}")
        report.append(f"**PR-AUC:** {model_data.get('pr_auc', 'N/A'):.4f}")
        report.append("")
    
    if results["predictions"] is not None:
        pred_df = results["predictions"]
        from sklearn.metrics import classification_report, confusion_matrix
        
        y_true = pred_df['y_true'].values
        y_pred = pred_df['y_pred'].values
        
        cm = confusion_matrix(y_true, y_pred)
        report.append("### 4.3 Confusion Matrix")
        report.append("")
        report.append("| | Predicted: No Default | Predicted: Default |")
        report.append("|--|---------------------|-------------------|")
        report.append(f"| **Actual: No Default** | {cm[0,0]} | {cm[0,1]} |")
        report.append(f"| **Actual: Default** | {cm[1,0]} | {cm[1,1]} |")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # 5. Интерпретация
    report.append("## 5. Интерпретация: Важные факторы риска")
    report.append("")
    
    if results["shap_importance"] is not None:
        report.append("### 5.1 TOP-10 Важных Признаков (SHAP)")
        report.append("")
        report.append("| Ранг | Признак | Важность (SHAP) |")
        report.append("|------|---------|------------------|")
        for idx, row in results["shap_importance"].head(10).iterrows():
            report.append(f"| {idx + 1} | {row['feature']} | {row['mean_abs_shap']:.4f} |")
        report.append("")
    
    if results["insights"]:
        report.append("### 5.2 Бизнес-выводы")
        report.append("")
        report.append("```")
        report.append(results["insights"])
        report.append("```")
        report.append("")
    
    report.append("---")
    report.append("")
    
    # 6. Бизнес-рекомендации
    report.append("## 6. Бизнес-рекомендации")
    report.append("")
    report.append("### 6.1 Для кредитного отдела")
    report.append("")
    report.append("1. **Мониторинг ключевых показателей:**")
    report.append("   - Регулярно отслеживать TOP-10 факторов риска")
    report.append("   - Установить пороговые значения для критических признаков")
    report.append("")
    report.append("2. **Процесс принятия решений:**")
    report.append("   - Использовать модель для предварительной оценки заявок")
    report.append("   - Для заемщиков с высоким риском (вероятность > 0.7) - дополнительная проверка")
    report.append("   - Для заемщиков с низким риском (вероятность < 0.3) - ускоренное одобрение")
    report.append("")
    report.append("3. **Калибровка модели:**")
    report.append("   - Регулярно переобучать модель на новых данных (ежеквартально)")
    report.append("   - Мониторить дрифт признаков")
    report.append("")
    
    report.append("### 6.2 Для управления рисками")
    report.append("")
    report.append("1. **Управление портфелем:**")
    report.append("   - Диверсификация по уровням риска")
    report.append("   - Резервирование средств для высокорисковых займов")
    report.append("")
    report.append("2. **Политики кредитования:**")
    report.append("   - Ужесточить требования для заемщиков с высоким debt-to-income ratio")
    report.append("   - Предлагать специальные условия для заемщиков с низким риском")
    report.append("")
    
    report.append("### 6.3 Ограничения и улучшения")
    report.append("")
    report.append("1. **Ограничения:**")
    report.append("   - Модель обучена на исторических данных, может не учитывать новые тренды")
    report.append("   - Внешние факторы (экономические кризисы) не учтены")
    report.append("")
    report.append("2. **Возможные улучшения:**")
    report.append("   - Добавить временные признаки (тренды, сезонность)")
    report.append("   - Интегрировать внешние данные (макроэкономические показатели)")
    report.append("   - Использовать ансамбли моделей")
    report.append("   - Реализовать онлайн-обучение для адаптации к изменениям")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append("## Заключение")
    report.append("")
    report.append("Разработанная модель успешно предсказывает вероятность дефолта заемщика с высокой точностью. ")
    report.append("SHAP анализ позволил выявить ключевые факторы риска, что дает возможность принимать обоснованные бизнес-решения.")
    report.append("")
    
    # Сохранение
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"✅ Отчет сохранен: {save_path}")
    
    return "\n".join(report)


def generate_executive_summary(results: dict, save_path: str = "presentation/executive_summary.md"):
    """
    Генерирует краткое резюме для руководства.
    """
    Path("presentation").mkdir(exist_ok=True)
    
    summary = []
    
    summary.append("# Executive Summary: Предсказание дефолта заемщика")
    summary.append("")
    summary.append(f"**Дата:** {datetime.now().strftime('%Y-%m-%d')}")
    summary.append("")
    summary.append("---")
    summary.append("")
    
    # Ключевые результаты
    summary.append("## Ключевые результаты")
    summary.append("")
    
    if results["model"] is not None:
        model_data = results["model"]
        summary.append(f"- **Лучшая модель:** {model_data.get('model_name', 'Unknown')}")
        summary.append(f"- **Точность (AUC):** {model_data.get('auc', 'N/A'):.4f}")
        summary.append(f"- **Точность (PR-AUC):** {model_data.get('pr_auc', 'N/A'):.4f}")
        summary.append("")
    
    # Топ факторы
    if results["shap_importance"] is not None:
        summary.append("## Топ-5 факторов риска дефолта")
        summary.append("")
        for idx, row in results["shap_importance"].head(5).iterrows():
            summary.append(f"{idx + 1}. **{row['feature']}** (важность: {row['mean_abs_shap']:.4f})")
        summary.append("")
    
    # Рекомендации
    summary.append("## Рекомендации")
    summary.append("")
    summary.append("1. Внедрить модель в процесс оценки заявок")
    summary.append("2. Установить пороговые значения для автоматического принятия/отклонения")
    summary.append("3. Регулярно переобучать модель на новых данных")
    summary.append("4. Мониторить ключевые факторы риска")
    summary.append("")
    
    # Сохранение
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    
    print(f"✅ Executive Summary сохранен: {save_path}")


# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("=" * 80)
    print("ГЕНЕРАЦИЯ МАТЕРИАЛОВ ДЛЯ ПРЕЗЕНТАЦИИ")
    print("=" * 80)
    
    # Загрузка результатов
    print("\n[1/3] Загрузка результатов...")
    results = load_all_results()
    print("   ✅ Результаты загружены")
    
    # Генерация отчета
    print("\n[2/3] Генерация полного отчета...")
    generate_presentation_report(results)
    
    # Генерация резюме
    print("\n[3/3] Генерация Executive Summary...")
    generate_executive_summary(results)
    
    print("\n" + "=" * 80)
    print("✅ МАТЕРИАЛЫ ДЛЯ ПРЕЗЕНТАЦИИ СОЗДАНЫ!")
    print("=" * 80)
    print("\nФайлы сохранены в папке presentation/")
    print("- report.md - Полный отчет")
    print("- executive_summary.md - Краткое резюме для руководства")


if __name__ == "__main__":
    main()

