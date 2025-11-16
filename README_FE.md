# Feature Engineering & Preprocessing Framework

This repository contains a production-style credit scoring preprocessing stack
designed to squeeze the maximum signal from the hackathon data sources. The
framework mirrors tooling used at top-tier lenders (Upstart/Zest/AMEX) while
remaining fully reproducible and notebook-friendly.

## Architecture

- `src/id_normalization.py` ― harmonises inconsistent ID columns into
  `customer_ref` so all downstream logic works off a single key.
- `src/merging.py` ― assembles the master customer table, performs left joins
  across application/demographics/loan/ratios and builds aggregated credit
  history features (rolling stats, delinquency patterns, payment ratios).
- `src/feature_eng.py` ― houses the feature factory: ratios, temporal trends,
  volatility, behavioural flags, time-since-event features, macro time signals,
  segment normalisation and more. Everything is config-driven through
  `FeatureSpec`.
- `src/preprocessing.py` ― advanced preprocessing (winsorisation, log transforms,
  missingness flags, rare-category collapsing, frequency/target/WOE encoding,
  optional scaling) powered by `AdvancedPreprocessor`.
- `src/woe_iv.py` ― monotonic binning + WOE transformer plus IV, correlation,
  mutual information and stability-selection helpers for Champion models.
- `src/fe_pipeline.py` ― orchestration entry-point exposing
  `preprocess_and_generate_features` which wires ID normalisation, merging,
  feature engineering and preprocessing together.
- `scripts/build_master_table.py` ― CLI to read raw files from
  `data/train` (default) or specified directory, run the full pipeline and persist the final
  modelling dataset (CSV or Parquet).

## Usage

```python
from src.fe_pipeline import preprocess_and_generate_features
from src.merging import MergeConfig
from src.preprocessing import PreprocessingConfig

dataset, artifacts = preprocess_and_generate_features(
    application_df,
    demographics_df=demographics_df,
    loan_df=loan_df,
    ratios_df=ratios_df,
    history_df=credit_history_df,
    merge_config=MergeConfig(),
    preprocessing_config=PreprocessingConfig(
        categorical_encoding="frequency",
        enable_woe=True,
    ),
    return_artifacts=True,
)
```

`dataset` is ready for CatBoost/LightGBM (target column `default` retained).
`artifacts` contains the fitted `AdvancedPreprocessor`, merge coverage metrics,
class weights for imbalanced training and the untransformed master table.

## Feature Families

- **Ratios & utilisation:** DTI, payment-to-due, utilisation, past-due ratios.
- **Aggregations:** Mean/max/min/std per customer for balances, payments, dues.
- **Rolling windows:** 3/6/12 month rolling means to capture recent dynamics.
- **Trends:** Absolute and relative changes over time (balance/utilisation).
- **Risk events:** DPD 30/60/90+ counts, shares, chronic delinquency flags.
- **Volatility:** Standard deviation and coefficient of variation for balances
  and payments (credit instability signal).
- **Behavioural flags:** Configurable thresholds (credit hungry, high
  utilisation) with optional rolling windows.
- **Time since events:** Days since last delinquency or other risk flag.
- **Macro/context:** Hour/day-of-week sine/cosine encodings, weekend/night
  indicators for fraud/risk control.
- **Segment normalisation:** Income vs. regional mean ratios for peer-relative
  stability.
- **Advanced Behavioral Features:** DPD counters (3/6/12 months), load trends (DTI/debt_service_ratio slopes), volatility (std/coefficient of variation), stress indicators (proportion of months with DPD > 0).
- **Default Detection Features:** Specialized features for default detection — risk combinations (extreme_risk_combo, critical_risk_combo), feature interactions (credit_history × debt_stress, utilization × debt), risk scoring (default_risk_score 0-10), behavioral flags (late_night_application, support_intensive_low_engagement), triple interactions (young + unstable + high debt).
- **Smart Interactions:** Automatic generation of interactions between top-K features (product, ratio, difference) filtered by Spearman correlation from `feature_strength.json`.
- **WOE/IV bundle:** Optional monotonic WOE features plus IV/correlation/MI
  ranking utilities for Champion scorecards.

## Preprocessing Highlights

- Target-aware imputations with missingness indicators on all numeric fields.
- Winsorization (1–99%) + log1p transforms for skewed distributions.
- Rare-category consolidation + flexible encoders (frequency, target, one-hot).
- Optional WOE appenders for interpretable scorecards.
- Automatic class weight computation (no SMOTE) for boosting downstream.

## CLI Workflow

```
python scripts/build_master_table.py \
    --data-dir data/train \
    --output artifacts/master_table.parquet \
    --categorical-encoding frequency \
    --enable-woe
```

This reads the five raw files, constructs the master dataset and stores the
result in `artifacts/master_table.parquet`. Swap the output suffix to `.csv`
to obtain a CSV export instead.

## Next Steps

- Feed the saved dataset directly into `src/modeling.py` or custom experiments.
- Use `artifacts.preprocessor` to transform validation / inference batches.
- Leverage `src/woe_iv.py` stability selection helpers to prune redundant
  features before deploying a Champion scorecard.
