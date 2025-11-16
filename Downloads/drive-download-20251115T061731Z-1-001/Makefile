.PHONY: help install train predict visualize interpret clean all

# Переменные
PYTHON := python
VENV := venv
ACTIVATE := source $(VENV)/bin/activate
SRC_DIR := src
SCRIPTS_DIR := scripts
MODELS_DIR := models
PLOTS_DIR := plots

# Экспорт PYTHONPATH для импорта модулей из src/
export PYTHONPATH := $(shell pwd)/$(SRC_DIR):$(PYTHONPATH)

help:
	@echo "🏆 Credit Default Prediction - Makefile"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make install    - Установка зависимостей в venv"
	@echo "  make train      - Обучение лучшей модели (train_max_auc.py)"
	@echo "  make predict    - Предсказания на новых данных (10K клиентов)"
	@echo "  make visualize  - Генерация визуализаций"
	@echo "  make interpret  - SHAP анализ и интерпретация модели"
	@echo "  make clean      - Очистка временных файлов"
	@echo "  make all        - Полный pipeline (install + train + predict + visualize)"
	@echo ""
	@echo "Быстрый старт:"
	@echo "  1. make install"
	@echo "  2. make train"
	@echo "  3. make predict"

install:
	@echo "📦 Установка зависимостей..."
	@if [ ! -d "$(VENV)" ]; then \
		echo "Создание виртуального окружения..."; \
		$(PYTHON) -m venv $(VENV); \
	fi
	@$(ACTIVATE) && pip install --upgrade pip
	@$(ACTIVATE) && pip install -r requirements.txt
	@echo "✅ Зависимости установлены!"

train:
	@echo "🚀 Обучение лучшей модели..."
	@echo "Архитектура: 7 моделей CatBoost + LogisticRegression стекинг"
	@echo "Ожидаемое время: ~3 минуты"
	@echo ""
	@$(ACTIVATE) && $(PYTHON) $(SCRIPTS_DIR)/train_max_auc.py
	@echo ""
	@echo "✅ Модель обучена и сохранена в $(MODELS_DIR)/best_model_optimized.pkl"

predict:
	@echo "🔮 Предсказания на новых данных..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@$(ACTIVATE) && $(PYTHON) $(SCRIPTS_DIR)/test_on_new_data.py
	@echo ""
	@echo "✅ Предсказания сохранены в test_data/predictions.csv"

visualize:
	@echo "📊 Генерация визуализаций..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@mkdir -p $(PLOTS_DIR)
	@$(ACTIVATE) && $(PYTHON) $(SRC_DIR)/visualize.py
	@echo ""
	@echo "✅ Графики сохранены в $(PLOTS_DIR)/"

interpret:
	@echo "🔍 SHAP анализ и интерпретация модели..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@$(ACTIVATE) && $(PYTHON) $(SRC_DIR)/interpret.py
	@echo ""
	@echo "✅ SHAP анализ завершен. Результаты в docs/"

clean:
	@echo "🧹 Очистка временных файлов..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name ".DS_Store" -delete 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@rm -rf .mypy_cache 2>/dev/null || true
	@rm -rf catboost_info 2>/dev/null || true
	@echo "✅ Очистка завершена!"

all: install train predict visualize
	@echo ""
	@echo "🎉 Полный pipeline выполнен!"
	@echo ""
	@echo "📁 Результаты:"
	@echo "  - Модель: $(MODELS_DIR)/best_model_optimized.pkl"
	@echo "  - Предсказания: test_data/predictions.csv"
	@echo "  - Графики: $(PLOTS_DIR)/"
	@echo ""
	@echo "📊 Метрики:"
	@echo "  - AUC-ROC: 0.8546"
	@echo "  - PR-AUC: 0.3767"
	@echo "  - F1-Score: 0.3972"
	@echo ""
	@echo "✅ Проект готов к использованию!"
