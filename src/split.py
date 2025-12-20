import pandas as pd
import numpy as np

INPUT = "data/pivot_data.parquet"
TRAIN_OUT = "data/pivot_train.parquet"
TEST_OUT = "data/pivot_test.parquet"

TEST_RATIO = 0.2  # 20% ide do testu

# Nastavenie seedu pre reprodukovateľnosť
np.random.seed(42)

print(f"Loading {INPUT}...")
df = pd.read_parquet(INPUT)
print(f"Original shape: {df.shape}")

# --- ANALÝZA DÁT (len pre info) ---
stds = df.std(numeric_only=True)
zero_std_cols = stds[stds == 0].index
if len(zero_std_cols) > 0:
    print(f"Dropping {len(zero_std_cols)} constant columns (std=0)...")
    df = df.drop(columns=zero_std_cols)

# --- RANDOM SPLIT ---
# 1. Vytvoríme náhodnú masku pre testovaciu sadu
#    (Namiesto jednoduchého rezu na konci)
mask = np.random.rand(len(df)) < (1 - TEST_RATIO)

df_train = df[mask]
df_test = df[~mask]

# 2. Zoradíme dáta späť podľa času (indexu)
#    Toto je kľúčové pre GRU! Model potrebuje vidieť dáta v poradí.
#    V train sete budú "diery" (chýbajúce riadky, ktoré sú v teste), ale časová postupnosť zostane.
df_train = df_train.sort_index()
df_test = df_test.sort_index()

print(f"Train shape: {df_train.shape}")
print(f"Test shape:  {df_test.shape}")

df_train.to_parquet(TRAIN_OUT)
df_test.to_parquet(TEST_OUT)

print("Saved (Random Split with Time Order preserved).")