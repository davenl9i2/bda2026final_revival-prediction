"""
Post feature-engineering cleanup for two columns with bad source values.
Overwrites train_features.csv / test_features.csv in place (utf-8-sig).

1. age_at_t0          : values outside [10, 100] -> NaN (negative / >100 = bad Birthday)
2. days_since_register: negative values -> NaN (register date after t0 = snapshot artifact)

All other 55 features untouched. F5 ratios are NOT winsorized here (left for pre-SHAP).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE / "output"
FILES = ["train_features.csv", "test_features.csv"]


def _report(name, s):
    return (f"    {name:24s} min={s.min():.2f}  max={s.max():.2f}  "
            f"missing={s.isna().mean() * 100:.2f}%")


def clean(path):
    print(f"\n=== {path.name} ===")
    df = pd.read_csv(path, encoding="utf-8-sig")

    print("  BEFORE:")
    print(_report("age_at_t0", df["age_at_t0"]))
    print(_report("days_since_register", df["days_since_register"]))

    # 1. age_at_t0 outside [10, 100] -> NaN
    df["age_at_t0"] = df["age_at_t0"].where(
        (df["age_at_t0"] >= 10) & (df["age_at_t0"] <= 100), np.nan
    )
    # 2. days_since_register negative -> NaN
    df["days_since_register"] = df["days_since_register"].where(
        df["days_since_register"] >= 0, np.nan
    )

    print("  AFTER:")
    print(_report("age_at_t0", df["age_at_t0"]))
    print(_report("days_since_register", df["days_since_register"]))

    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  overwritten -> {path}  ({len(df):,} rows, {len(df.columns)} cols)")


def main():
    for f in FILES:
        clean(OUT_DIR / f)
    print("\nDone. Only age_at_t0 / days_since_register modified.")


if __name__ == "__main__":
    main()
