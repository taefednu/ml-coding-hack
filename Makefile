.PHONY: help install train predict result visualize interpret clean all

# Переменные
PYTHON := python
VENV := venv
VENV_PYTHON := $(VENV)/bin/python
SRC_DIR := src
SCRIPTS_DIR := scripts
MODELS_DIR := models
PLOTS_DIR := plots
SHELL := /bin/bash

# Экспорт PYTHONPATH для импорта модулей из src/
export PYTHONPATH := $(shell pwd)/$(SRC_DIR):$(PYTHONPATH)

help:
	@echo "🏆 Credit Default Prediction - Makefile"
	@echo ""
	@echo "Доступные команды:"
	@echo "  make install    - Установка зависимостей в venv"
	@echo "  make train      - Обучение лучшей модели (train_max_auc.py)"
	@echo "  make predict    - Предсказания на новых данных (10K клиентов)"
	@echo "  make result     - Создание result.csv для хакатона (customer_id, prob, default)"
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
	@$(VENV_PYTHON) -m pip install --upgrade pip
	@$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "✅ Зависимости установлены!"

train:
	@echo "🚀 Обучение лучшей модели..."
	@echo "Архитектура: 7 моделей CatBoost + LogisticRegression стекинг"
	@echo "Ожидаемое время: ~3 минуты"
	@echo ""
	@$(VENV_PYTHON) $(SCRIPTS_DIR)/train_max_auc.py
	@echo ""
	@echo "✅ Модель обучена и сохранена в $(MODELS_DIR)/best_model_optimized.pkl"

predict:
	@echo "🔮 Предсказания на новых данных..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@$(VENV_PYTHON) $(SCRIPTS_DIR)/test_on_new_data.py
	@echo ""
	@echo "✅ Предсказания сохранены в test_data/predictions.csv"

result:
	@echo "📋 Создание result.csv для хакатона..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@$(VENV_PYTHON) $(SCRIPTS_DIR)/test_on_new_data.py
	@echo ""
	@echo "📝 Формирование result.csv..."
	@if [ -f "test_data/predictions.csv" ]; then \
		$(VENV_PYTHON) -c "import pandas as pd; df = pd.read_csv('test_data/predictions.csv'); result = pd.DataFrame({'customer_id': df.get('customer_id', df.get('application_id', range(len(df)))), 'prob': df['probability'], 'default': df['prediction']}); result.to_csv('result.csv', index=False); print(f'✅ result.csv создан: {len(result)} строк')"; \
	else \
		echo "❌ Ошибка: predictions.csv не найден. Сначала выполните 'make predict'"; \
		exit 1; \
	fi
	@echo ""
	@echo "✅ result.csv создан в корне проекта (формат: customer_id, prob, default)"

visualize:
	@echo "📊 Генерация визуализаций..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@mkdir -p $(PLOTS_DIR)
	@$(VENV_PYTHON) $(SRC_DIR)/visualize.py
	@echo ""
	@echo "✅ Графики сохранены в $(PLOTS_DIR)/"

interpret:
	@echo "🔍 SHAP анализ и интерпретация модели..."
	@if [ ! -f "$(MODELS_DIR)/best_model_optimized.pkl" ]; then \
		echo "❌ Ошибка: Модель не найдена. Сначала выполните 'make train'"; \
		exit 1; \
	fi
	@$(VENV_PYTHON) $(SRC_DIR)/interpret.py
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
