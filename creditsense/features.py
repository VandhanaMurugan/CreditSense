"""Raw applicant attributes -> model features. Identical to pipeline/step1 (cleaning fitted on the training split only)."""
import numpy as np
import pandas as pd

PAST_DUE = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTimes90DaysLate", "NumberOfTime60-89DaysPastDueNotWorse"]
RAW_INPUTS = ["RevolvingUtilizationOfUnsecuredLines", "age", "NumberOfTime30-59DaysPastDueNotWorse",
              "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate",
              "NumberOfOpenCreditLinesAndLoans", "NumberRealEstateLoansOrLines"]
OPTIONAL_INPUTS = ["DebtRatio", "MonthlyIncome", "NumberOfDependents"]   # accepted, not used by the selected features


def fit_cleaning(train_raw: pd.DataFrame) -> dict:
    p = {}
    for col in PAST_DUE:
        p[f"{col}_sentinel_cap"] = float(train_raw[col][train_raw[col] < 90].max())
    p["MonthlyIncome_median"] = float(train_raw["MonthlyIncome"].median())
    p["NumberOfDependents_median"] = float(train_raw["NumberOfDependents"].median())
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        p[f"{col}_p995"] = float(train_raw[col].quantile(0.995))
    return p


def apply_cleaning(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    for col in OPTIONAL_INPUTS:
        if col not in df: df[col] = np.nan
    for col in PAST_DUE:
        df[col] = np.where(df[col] >= 90, p[f"{col}_sentinel_cap"], df[col])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(p["MonthlyIncome_median"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(p["NumberOfDependents_median"])
    for col in ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]:
        df[col] = df[col].fillna(0.0) if col == "DebtRatio" else df[col]
        df[col] = np.where(df[col] > p[f"{col}_p995"], p[f"{col}_p995"], df[col])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalPastDueEvents"] = (df["NumberOfTime30-59DaysPastDueNotWorse"] + df["NumberOfTime60-89DaysPastDueNotWorse"]
                                + df["NumberOfTimes90DaysLate"])
    df["HasPastDueHistory"] = (df["TotalPastDueEvents"] > 0).astype(int)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["DebtToIncomeProxy"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["CreditLinesPerOpenLoan"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["NumberRealEstateLoansOrLines"] + 1)
    df["UtilizationBucket"] = pd.cut(df["RevolvingUtilizationOfUnsecuredLines"], bins=[-0.01, 0.3, 0.7, 1.0, np.inf],
                                     labels=[0, 1, 2, 3]).astype(int)
    return df


def to_model_frame(raw: pd.DataFrame, cleaning: dict, features: list) -> pd.DataFrame:
    return engineer_features(apply_cleaning(raw, cleaning))[features]
