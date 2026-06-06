"""
clean_2022.py
==============

Derivation script that produces ``heart_2022_cleaned_final.csv`` from the
Kaggle BRFSS-2022 source file ``heart_2022_no_nans.csv``.

Source on Kaggle:
    https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
    File: heart_2022_no_nans.csv  (245,901 rows × 40 columns)

Output:
    heart_2022_cleaned_final.csv  (236,922 rows × 18 columns)

Run:
    python clean_2022.py --input heart_2022_no_nans.csv --output heart_2022_cleaned_final.csv

The 2022 BRFSS cycle uses a different schema than the 2020 cycle, so this
script harmonises it to the 18-column 2020 schema. The transformations
applied, in order:

  1.  Construct HeartDisease target as logical OR of HadHeartAttack and
      HadAngina (the 2020 cycle uses a single 'HeartDisease' flag that
      covers both).

  2.  Collapse the 4-level SmokerStatus into a binary Smoking field:
      'Yes' for any currently smoking respondent
            (both 'Current smoker - now smokes every day'
             and 'Current smoker - now smokes some days')
      'No'  for everyone else
            ('Former smoker' and 'Never smoked').

  3.  Rename columns to match the 2020 schema:
        AlcoholDrinkers       → AlcoholDrinking
        HadStroke             → Stroke
        PhysicalHealthDays    → PhysicalHealth
        MentalHealthDays      → MentalHealth
        DifficultyWalking     → DiffWalking
        RaceEthnicityCategory → Race
        HadDiabetes           → Diabetic
        PhysicalActivities    → PhysicalActivity
        GeneralHealth         → GenHealth
        SleepHours            → SleepTime
        HadAsthma             → Asthma
        HadKidneyDisease      → KidneyDisease
        HadSkinCancer         → SkinCancer

  4.  Normalise AgeCategory labels:
        'Age 65 to 69'    → '65-69'
        'Age 80 or older' → '80 or older'

  5.  Select and reorder to the final 18 columns of the 2020 schema.

  6.  Drop exact duplicate rows.

  7.  Remove BMI outliers (BMI < 10 or BMI > 70) as physiologically
      implausible values.
"""

import argparse
import pandas as pd


COLUMN_RENAME = {
    "AlcoholDrinkers":       "AlcoholDrinking",
    "HadStroke":             "Stroke",
    "PhysicalHealthDays":    "PhysicalHealth",
    "MentalHealthDays":      "MentalHealth",
    "DifficultyWalking":     "DiffWalking",
    "RaceEthnicityCategory": "Race",
    "HadDiabetes":           "Diabetic",
    "PhysicalActivities":    "PhysicalActivity",
    "GeneralHealth":         "GenHealth",
    "SleepHours":            "SleepTime",
    "HadAsthma":             "Asthma",
    "HadKidneyDisease":      "KidneyDisease",
    "HadSkinCancer":         "SkinCancer",
}

FINAL_COLUMNS = [
    "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke",
    "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory",
    "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime",
    "Asthma", "KidneyDisease", "SkinCancer",
]


def harmonise_2022(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw 2022 schema (40 cols) to harmonised 2020 schema (18 cols)."""

    # Step 1: HeartDisease = HadHeartAttack OR HadAngina
    df = df.copy()
    df["HeartDisease"] = df.apply(
        lambda r: "Yes" if (r["HadHeartAttack"] == "Yes"
                            or r["HadAngina"] == "Yes") else "No",
        axis=1,
    )

    # Step 2: Smoking = "Yes" if currently smoking, "No" otherwise
    df["Smoking"] = df["SmokerStatus"].apply(
        lambda v: "Yes" if isinstance(v, str) and "Current smoker" in v else "No"
    )

    # Step 3: Column renames
    df = df.rename(columns=COLUMN_RENAME)

    # Step 4: Normalise AgeCategory format
    df["AgeCategory"] = (
        df["AgeCategory"]
        .str.replace("Age ", "", regex=False)
        .str.replace(" to ", "-", regex=False)
    )

    # Step 5: Select and reorder
    df = df[FINAL_COLUMNS]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate and remove BMI outliers."""
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"  Removed {before - len(df):,} duplicate rows.")

    before = len(df)
    df = df[(df["BMI"] >= 10) & (df["BMI"] <= 70)].copy()
    print(f"  Removed {before - len(df):,} BMI outliers (<10 or >70).")

    return df


def main():
    ap = argparse.ArgumentParser(
        description="Derive heart_2022_cleaned_final.csv from heart_2022_no_nans.csv"
    )
    ap.add_argument("--input", default="heart_2022_no_nans.csv",
                    help="Kaggle source file (heart_2022_no_nans.csv)")
    ap.add_argument("--output", default="heart_2022_cleaned_final.csv",
                    help="Output file path")
    args = ap.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"  Source: {len(df):,} rows × {df.shape[1]} columns")

    print("Harmonising to 2020 schema...")
    df = harmonise_2022(df)
    print(f"  After harmonisation: {len(df):,} rows × {df.shape[1]} columns")

    print("Cleaning (deduplicate + BMI outliers)...")
    df = clean(df)
    print(f"  Final: {len(df):,} rows × {df.shape[1]} columns")

    print(f"Writing {args.output}...")
    df.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
