# Mini-Upstart PD Scoring Stack (Uzbekistan)

> MVP кредитного AI-скоринга уровня Upstart / Zest AI / VantageScore 4.0, построенный за 2 дня хакатона: от инвентаризации данных до FastAPI `/score`, Champion–Challenger моделей, PD-калибровки, scorecard 300–900, SHAP/fairness отчётности и тестового покрытия.

## Почему это похоже на мировые решения
- **Инспирация**: лучшие практики Upstart, Zest AI, VantageScore 4.0, Equifax NeuroDecision, Kaggle AMEX/BNPL решений.
- **Архитектура**: чёткое разделение Champion (интерпретируемая Logistic+WOE) и Challenger (CatBoost/LightGBM с CV и лёгким тюнингом) + калибровка PD + scorecard layer (PDO=50, 300–900).
- **Feature Factory**: модуль `src/feature_eng.py` агрегирует поведенческие, трендовые, event-фичи, сегментные нормировки и rolling-метрики в стиле Kaggle winners.
- **Explainability/Fairness**: SHAP / feature importances / reason codes (`src/explainability.py`) + fairness-разрезы (`src/fairness.py`) с экспортом JSON-репортов.
- **MLOps mindset**: reproducible Makefile, configs YAML, артефакты (`models/`, `artifacts/`), drift (PSI), метрики, FastAPI сервис и тесты.

## Day-1 / Day-2 план (поддерживается кодом)
**Day 1**: `make inventory`, `make eda`, выбор таргета/дат, базовый препроцессинг, Champion Logistic+WOE (`scripts/train.py`), черновой CatBoost/LGBM.

**Day 2**: расширенный feature engineering, CV+random tuning Challenger, калибровка PD, 300–900 scorecard (`models/scorecard.yaml`), SHAP и fairness отчёты (`artifacts/*_explainability.json`, `artifacts/fairness_report.json`), FastAPI `/score`, финальные `train/evaluate/predict` скрипты и тесты.

## Структура
```
configs/default.yaml      # единый конфиг: пути, таргет, сплиты, фичи, модели, XAI, fairness
src/
  ├─ data_loading.py      # инвентаризация (CSV/Parquet/JSONL/XML/XLSX), кодировки, схемы
  ├─ preprocessing.py     # ColumnTransformer, таргет-энкодер, валюта/даты
  ├─ feature_eng.py       # агрегаты, rolling, тренды, события риска, сегментные нормировки
  ├─ woe_iv.py            # монотонный биннинг, WOE/IV
  ├─ modeling.py          # Champion/Challenger, TimeSeries+Group CV, легкий тюнинг
  ├─ calibration.py       # Platt vs Isotonic, выбор лучшего
  ├─ scorecard.py         # points-to-odds, score 300–900, экспорт YAML/JSON
  ├─ metrics.py           # ROC/PR/KS/Gini/LogLoss/Brier/ECE, lift/gain
  ├─ drift.py             # PSI train↔valid↔OOT
  ├─ explainability.py    # SHAP, feature importance, reason codes
  ├─ fairness.py          # AUC/bad-rate/score по группам
  └─ utils.py
scripts/
  ├─ eda_report.py        # Markdown+plots EDA, кандидат таргета/дат
  ├─ train.py             # ingest → FE → Champion/Challenger → tuning → calibration → XAI → scorecard
  ├─ evaluate.py          # OOT-метрики, reliability, fairness JSON
  └─ predict.py           # батч-скоринг (CSV/Parquet/JSONL)
api/
  ├─ app.py               # FastAPI `/healthz`, `/score`
  └─ schemas.py
models/, artifacts/       # артефакты (champion.pkl, challenger.pkl, calibrator, scorecard, метрики, shap)
```

## Данные, таргет и сплиты
- **RAW**: `data/ml_coding_hackathon/*` (UTF-8/CP1251, `,`/`;`/`\t`, Parquet, JSONL, XML, XLSX). `make inventory` генерирует `artifacts/data_inventory.json` + Markdown EDA (`artifacts/reports/eda_report.md`).
- **Таргет**: по возможности `default_flag`/`label`. Если нет — `bad_90 = 1`, если `overdue_90d > 0` или `dpd_90 >= 1` (логика фиксируется в README/config). Плейсхолдер `<<TARGET_DEF>>` заменяется после выбора.
- **Сплиты**: строгие по времени: `train <= 2023-06-30`, `valid ∈ (2023-06-30; 2023-09-30]`, `OOT > 2023-09-30`. Внутри моделей используем `TimeSeriesSplit` + Group по `<<CLIENT_ID_COL>>`, чтобы клиент не попадал в разные фолды.

## Feature Engineering (мини Upstart/VantageScore)
- **Ratios**: DTI, utilization, debt-service, custom ratios из конфига.
- **Aggregations**: mean/max/min/std для платежей/балансов/поступлений.
- **Rolling windows**: 3/6/12 периодов для средних и максимумов.
- **Trended data**: прирост/наклон по доходу/балансу на клиента (если есть временные ряды).
- **Risk events**: флаги high utilization, количество просрочек 30/60/90+, реструктуризаций.
- **Segment normalization**: доход/балансы относительно среднего по региону/типу занятости.
- **Config-driven**: `configs/default.yaml::feature_engineering` описывает маппинги, чтобы быстро адаптироваться под реальные колонки.

## Champion vs Challenger
| Модель | Назначение | Файл | Особенности |
|--------|------------|------|-------------|
| **Champion** | Регуляторно-дружественный baseline | `models/champion_model.pkl` | Logistic Regression + WOE/IV с монотонными биннами. Экспорт коэффициентов и bins (`artifacts/champion_explainability.json`). |
| **Challenger** | ML-мотор AUC | `models/challenger_model.pkl` | CatBoost (приоритет) / LightGBM с class weights, TimeSeries+Group-CV, лёгкое random tuning (`modeling.py`). SHAP/feature importance сохраняются (`artifacts/challenger_shap.json`). |
| **Production best** | Используется для PD/score | `models/best_model.pkl` | Совпадает с Challenger, если он лучше Champion. |

## Калибровка и Scorecard 300–900
- `src/calibration.py`: Platt vs Isotonic, выбирается лучшая по Brier/LogLoss на валидации и проверяется на OOT. Калибратор сохраняется в `models/calibrator.pkl`.
- `src/scorecard.py`: points-to-odds (PDO=50, BaseOdds=20, BaseScore=650) → функции `pd_to_score()` и `score_to_pd()`. Scorecard (`models/scorecard.yaml`) содержит Offset/Factor, коэффициенты Champion и бин-таблицы для отчётности, совместимо с требованием 300–900.

## Explainability & Fairness
- `src/explainability.py`:
  - Reason codes Champion (top-k факторов на клиента, woe×coef).
  - SHAP/feature importance для Challenger (CatBoost builtin SHAP, fallback на gain). Сохраняется JSON для дашборда.
- `src/fairness.py`: подсчёт AUC/bad-rate/avg-score по чувствительным признакам (`configs/default.yaml::fairness.sensitive_cols`). Колонки исключаются из обучения, но попадают в отчёт `artifacts/fairness_report.json`.
- **PSI & stability**: `src/drift.py` считает PSI между train/valid/OOT, результаты — `artifacts/psi.json`.

## Как воспроизвести
```bash
make venv
make inventory   # автоинвентаризация и markdown-EDA
make eda         # расширенный отчёт с графиками (fallback на matplotlib)
make train       # Champion + Challenger + calibration + scorecard + XAI + артефакты
make evaluate    # OOT-метрики, reliability, fairness
make predict INPUT=data/new_apps.csv OUTPUT=artifacts/predictions.csv
make api         # FastAPI на http://localhost:8080
make tests       # pytest (data loading, leakage, metrics, API)
```

## API и батч-скоринг
- **FastAPI** (`api/app.py`):
  - `GET /healthz`
  - `POST /score` ⇒ `{ "features": {...} } → { "pd": 0.042, "score": 715, "model": "catboost" }`
  ```bash
  curl -X POST http://localhost:8080/score \
       -H 'Content-Type: application/json' \
       -d '{"features": {"age": 35, "monthly_income": 12000000, "dpd_90": 0}}'
  ```
- **Batch** (`scripts/predict.py`): `python scripts/predict.py --config configs/default.yaml --input data/applications.parquet --output artifacts/applications_scored.csv`.

## Тесты
- `tests/test_data_loading.py` — схемы, кодировки, multi-format ingest.
- `tests/test_leakage.py` — target encoding без утечек.
- `tests/test_metrics.py` — ROC/Gini/PSI sanity.
- `tests/test_api.py` — контракт API (health + score).

## Roadmap 2.0 (после хакатона)
- **Модели**: TabNet/Transformer/FTT на транзакциях, Equifax NeuroDecision-style NN, GNN по связям клиентов, мульти-хэд ансамбли.
- **Data Graph**: динамические поведенческие графы, внешние бюро и телко-данные, pseudo-labeling.
- **MLOps**: MLflow Tracking + Registry, Dagster/Airflow пайплайн, CI/CD + infra-as-code, прод-мониторинг drift/fairness, alerting.
- **Explainability**: интерактивные dashboards (Plotly Dash/Streamlit), контрастивные explanations.
- **Fairness & Governance**: automated bias mitigation, reject inference, compliance toolkit (audit trails, policy checks).
- **Notebooks**: `notebooks/eda.ipynb`, `notebooks/shap_report.ipynb` (описаны, но не включены — можно добавить после хакатона).

## Допущения и плейсхолдеры
- `<<CLIENT_ID_COL>>`, `<<APPLICATION_ID_COL>>`, `<<DATE_COLS>>`, `<<TARGET_DEF>>`, `<<SENSITIVE_COLS>>`, `<<COST_MATRIX_JSON>>`, `<<MONOTONE_FEATURES_JSON>>` — заменить на реальные значения после финального EDA.
- Sensitive-фичи удаляются из обучения автоматически; используются только для fairness отчёта.
- Части feature engineering/SHAP зависят от фактически доступных временнЫх рядов — архитектура готова, доработка сводится к настройке конфигов.

## Использование ИИ-ассистента
Часть boilerplate и документации была ускорена с помощью Codex/GPT. Все решения/код ревьюились инженером, соответствуют требованиям хакатона (обязательный раздел об ИИ-респектации выполнен).
