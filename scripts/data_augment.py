import os
import pandas as pd
import numpy as np

# Paths
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'Dataset')
SRC_CSV = os.path.join(DATA_DIR, 'survey lung cancer.csv')
OUT_CSV = os.path.join(DATA_DIR, 'survey_lung_cancer_500.csv')

TARGET_ROWS = 500


def jitter_age(age: int) -> int:
    # small random jitter within plausible range
    jitter = np.random.randint(-3, 4)
    new_age = int(np.clip(age + jitter, 21, 87))
    return new_age


def random_flip(val: int, p: float = 0.1) -> int:
    # values are 1/2; with prob p, flip between 1 and 2
    if np.random.rand() < p:
        return 1 if val == 2 else 2
    return val


def coerce_to_int(value, column: str) -> int:
    """Coerce mixed string/numeric values to 1/2 integers with sensible defaults."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        v = value.strip().upper()
        if column == 'GENDER':
            if v in ('M', 'MALE', '1'):
                return 1
            if v in ('F', 'FEMALE', '2'):
                return 2
        if column == 'LUNG_CANCER':
            if v in ('YES', 'Y', '1', 'TRUE'):
                return 1
            if v in ('NO', 'N', '2', 'FALSE'):
                return 2
        # generic 1/2 mapping
        if v == '1':
            return 1
        if v == '2':
            return 2
    # fallback default
    return 1


def synthesize_row(row: pd.Series) -> pd.Series:
    new_row = row.copy()
    # Age: slight jitter
    new_row['AGE'] = jitter_age(int(coerce_to_int(row['AGE'], 'AGE')))

    # For binary/categorical 1-2 columns, flip with small prob
    cat_cols = [
        'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
        'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY', 'WHEEZING',
        'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'LUNG_CANCER'
    ]
    for c in cat_cols:
        if c in new_row:
            new_row[c] = random_flip(int(coerce_to_int(row[c], c)), p=0.1)

    return new_row


def main():
    np.random.seed(42)
    df = pd.read_csv(SRC_CSV)

    # Normalize known string categories to integers upfront
    if df['GENDER'].dtype == object:
        df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 2, 'm': 1, 'f': 2}).fillna(df['GENDER'])
    if df['LUNG_CANCER'].dtype == object:
        df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 2, 'Yes': 1, 'No': 2}).fillna(df['LUNG_CANCER'])

    if len(df) >= TARGET_ROWS:
        df_out = df.iloc[:TARGET_ROWS].copy()
    else:
        needed = TARGET_ROWS - len(df)
        # sample rows with replacement to synthesize
        base = df.copy()
        sampled = base.sample(n=needed, replace=True, random_state=42).reset_index(drop=True)
        synth = sampled.apply(synthesize_row, axis=1)
        df_out = pd.concat([df, synth], ignore_index=True)

    # Shuffle to avoid all synthetic at the end
    df_out = df_out.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df_out)} rows to {OUT_CSV}")


if __name__ == '__main__':
    main()
