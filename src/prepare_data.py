import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_PATH = Path("data_sets")  # поменяй на свою папку при необходимости


def to_snake(name: str) -> str:
    """
    Приводит имена колонок к snake_case.
    """
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    name = name.replace(" ", "_").replace("-", "_")
    return name.lower()


def clean_money(series: pd.Series) -> pd.Series:
    """
    Преобразует строковый денежный формат в float.
    Убирает $, запятые, пробелы. 'None', 'nan', 'null' и пустые -> NaN.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r'[\$,]', '', regex=True)
        .str.strip()
    )

    cleaned = cleaned.replace(
        ['None', 'none', 'NaN', 'nan', 'NULL', 'null', ''],
        np.nan
    )

    # безопасно переводим в числа: всё странное -> NaN
    return pd.to_numeric(cleaned, errors="coerce")


def load_application_metadata() -> pd.DataFrame:
    df = pd.read_csv(BASE_PATH / "application_metadata.csv")

    # Переименуем ID и приведём к snake_case
    df = df.rename(columns={"customer_ref": "customer_id"})
    df.columns = [to_snake(c) for c in df.columns]
    
    # Удаляем шумовые колонки
    noise_cols = [c for c in df.columns if 'noise' in c.lower() or 'random' in c.lower()]
    if noise_cols:
        df = df.drop(columns=noise_cols)

    # Приводим типы
    df["customer_id"] = df["customer_id"].astype(int)

    # Бинарные флаги
    for col in ["has_mobile_app", "paperless_billing", "default"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Числовые
    num_cols = [
        "application_hour",
        "application_day_of_week",
        "account_open_year",
        "num_login_sessions",
        "num_customer_service_calls",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Категориальные чистим от пробелов
    cat_cols = ["preferred_contact", "referral_code", "account_status_code"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan})
            )

    return df


def load_credit_history() -> pd.DataFrame:
    # имя файла у тебя: credit_hystory.csv
    df = pd.read_csv(BASE_PATH / "credit_hystory.csv")

    df = df.rename(columns={"customer_number": "customer_id"})
    df.columns = [to_snake(c) for c in df.columns]

    df["customer_id"] = df["customer_id"].astype(int)

    # Числовые колонки
    for col in df.columns:
        if col == "customer_id":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(BASE_PATH / "demographics.csv")

    df = df.rename(columns={"cust_id": "customer_id"})
    df.columns = [to_snake(c) for c in df.columns]

    df["customer_id"] = df["customer_id"].astype(int)

    # Чистим annual_income
    if "annual_income" in df.columns:
        df["annual_income"] = clean_money(df["annual_income"])

    # Числовые поля
    num_cols = ["age", "employment_length", "num_dependents"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Нормализуем employment_type
    if "employment_type" in df.columns:
        df["employment_type"] = (
            df["employment_type"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("-", "_")
            .str.replace(" ", "_")
        )

    return df


def load_financial_ratios() -> pd.DataFrame:
    df = pd.read_json(BASE_PATH / "financial_ratios.jsonl", lines=True)

    df = df.rename(columns={"cust_num": "customer_id"})
    df.columns = [to_snake(c) for c in df.columns]

    df["customer_id"] = df["customer_id"].astype(int)

    money_cols = [
        "monthly_income",
        "existing_monthly_debt",
        "monthly_payment",
        "revolving_balance",
        "credit_usage_amount",
        "available_credit",
        "total_monthly_debt_payment",
        "total_debt_amount",
        "monthly_free_cash_flow",
    ]
    for col in money_cols:
        if col in df.columns:
            df[col] = clean_money(df[col])

    # Остальные числовые
    for col in df.columns:
        if col in ["customer_id"] + money_cols:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_loan_details() -> pd.DataFrame:
    df = pd.read_excel(BASE_PATH / "loan_details.xlsx")

    df.columns = [to_snake(c) for c in df.columns]

    # Пытаемся найти customer_id, если есть
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(int)

    # Чистим денежные поля (все object-поля, где есть $, запятые или точки)
    for col in df.columns:
        if df[col].dtype == "object":
            # пробуем преобразовать как деньги
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .str.strip()
            )
            # если после чистки больше половины строк — числа, считаем это денежным
            is_num = pd.to_numeric(cleaned, errors="coerce")
            if is_num.notna().mean() > 0.5:
                df[col] = is_num
            # иначе оставляем как есть (категория/текст)

    return df


def load_geographic_data() -> pd.DataFrame:
    tree = ET.parse(BASE_PATH / "geographic_data.xml")
    root = tree.getroot()

    rows = []
    for cust in root.findall("customer"):
        row = {child.tag: child.text for child in cust}
        rows.append(row)

    df = pd.DataFrame(rows)

    # customer_id и snake_case
    df = df.rename(columns={"id": "customer_id"})
    df.columns = [to_snake(c) for c in df.columns]

    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype(int)

    # Числовые колонки (кроме state)
    num_cols = [
        "regional_unemployment_rate",
        "regional_median_income",
        "regional_median_rent",
        "housing_price_index",
        "cost_of_living_index",
        "previous_zip_code",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clip_outliers(df: pd.DataFrame, exclude_cols=None) -> pd.DataFrame:
    """
    Простая обработка выбросов:
    обрезаем все числовые признаки по [1-й перцентиль, 99-й перцентиль].
    """
    if exclude_cols is None:
        exclude_cols = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q1, upper=q99)
    return df


def build_master_dataset() -> pd.DataFrame:
    app = load_application_metadata()
    credit = load_credit_history()
    demo = load_demographics()
    ratios = load_financial_ratios()
    loan = load_loan_details()
    geo = load_geographic_data()

    # начинаем с application_metadata как "ядра"
    master = app.copy()

    master = master.merge(credit, on="customer_id", how="left", suffixes=("", "_credit"))
    master = master.merge(demo, on="customer_id", how="left", suffixes=("", "_demo"))
    master = master.merge(ratios, on="customer_id", how="left", suffixes=("", "_ratios"))
    master = master.merge(geo, on="customer_id", how="left", suffixes=("", "_geo"))

    # loan_details: если нет customer_id, джойним по индексу
    if "customer_id" in loan.columns:
        master = master.merge(loan, on="customer_id", how="left", suffixes=("", "_loan"))
    else:
        loan_reset = loan.reset_index().rename(columns={"index": "row_id"})
        master_reset = master.reset_index().rename(columns={"index": "row_id"})
        master = master_reset.merge(loan_reset, on="row_id", how="left", suffixes=("", "_loan"))
        master = master.drop(columns=["row_id"])

    # Обработка выбросов (не трогаем таргет)
    master = clip_outliers(master, exclude_cols=["default", "customer_id"])

    return master


if __name__ == "__main__":
    master_df = build_master_dataset()
    print("Master shape:", master_df.shape)
    print("Default rate:")
    print(master_df["default"].value_counts(normalize=True))

    # Сохраним результат
    master_df.to_csv(BASE_PATH / "master_dataset_clean.csv", index=False)
    print("Saved to master_dataset_clean.csv")