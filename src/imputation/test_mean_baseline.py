import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader

# Importy z tvojho projektu
from utils import prepare_data, remove_outliers_iqr
from dataset import ImputationDataset
import model_param

# Nastavenia (musia byť rovnaké ako v test.py)
SEQ_LEN = model_param.SEQ_LEN
ART_RATE = 0.15
torch.manual_seed(42)
np.random.seed(42)

# 1. Načítanie Scalera (z trénovania)
scaler_path = os.path.join("models", "imputation", "scaler.pkl")
if not os.path.exists(scaler_path):
    print("Scaler not found! Run train.py first.")
    exit()

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# 2. Výpočet priemerov z TRÉNOVACÍCH dát (Baseline)
# Je dôležité použiť priemer z trénovacej sady, nie z testovacej (data leakage).
print("Calculating training means...")
df_train = pd.read_parquet("data/pivot_train.parquet")
df_train = remove_outliers_iqr(df_train) # Aplikujeme rovnaké čistenie ako pri tréningu

# Pridanie časových príznakov (musí byť identické ako v utils.py)
time_index = pd.to_datetime(df_train.index)
day_of_year = time_index.dayofyear.to_numpy()
hour_of_day = time_index.hour.to_numpy()
df_train["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.0)
df_train["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.0)
df_train["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0)
df_train["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0)

# Výpočet priemeru (ignorujeme NaN)
train_means = df_train.mean().to_numpy().reshape(1, -1)

# Škálovanie priemerov (aby sme boli v rovnakom priestore ako test.py a scaler)
# Scaler očakáva 2D pole, preto reshape
train_means_scaled = scaler.transform(train_means)

# 3. Príprava Testovacích dát
print("Loading test data...")
# fit_scaler=False, lebo používame už načítaný scaler
X, M, y, target_masks, _, mask, df, _ = prepare_data(
    "data/pivot_test.parquet", SEQ_LEN, scaler=scaler, fit_scaler=False, verbose=False
)

# 4. Evaluácia (Mean Imputation)
test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

all_preds_scaled, all_targs_scaled, all_flags_art = [], [], []

print("Running Mean Imputation Baseline...")
for X_batch, M_batch, y_batch, tm_batch in test_loader:
    # Generovanie rovnakej masky ako pri GRU (vďaka fixnému seedu)
    present = (tm_batch == 0)
    rand = torch.rand_like(present.float())
    art_mask = (rand < ART_RATE) & present
    
    # PREDIKCIA: Vždy len priemer (broadcastnutý na veľkosť batchu)
    batch_size = y_batch.shape[0]
    # Každý riadok predikcie je rovnaký vektor priemerov
    preds = np.repeat(train_means_scaled, batch_size, axis=0)
    
    all_preds_scaled.append(preds)
    all_targs_scaled.append(y_batch.numpy())
    all_flags_art.append(art_mask.numpy().astype(np.uint8))

# Spojenie výsledkov
preds_scaled = np.concatenate(all_preds_scaled, axis=0)
targs_scaled = np.concatenate(all_targs_scaled, axis=0)
flags_art = np.concatenate(all_flags_art, axis=0)

# 5. Inverzná transformácia (späť na reálne jednotky)
print("Inverting scale...")
preds_orig = scaler.inverse_transform(preds_scaled)
targs_orig = scaler.inverse_transform(targs_scaled)

# 6. Výpočet metrík
print("\n" + "="*60)
print(f"{'Feature Name':<30} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
print("-" * 60)

results = []
for col, feature_name in enumerate(df.columns):
    # Vyberieme len tie body, ktoré boli umelo zamaskované
    col_mask = flags_art[:, col] == 1
    
    if not np.any(col_mask):
        continue
        
    y_true = targs_orig[col_mask, col]
    y_pred = preds_orig[col_mask, col]
    
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"{feature_name[:30]:<30} | {rmse:8.4f} | {mae:8.4f} | {r2:8.4f}")
    results.append({"Feature": feature_name, "RMSE": rmse, "MAE": mae, "R2": r2})

# Uloženie
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("mean_imputation_metrics.txt", sep="\t", index=False)
print("="*60)
print(f"Average R2 Score: {metrics_df['R2'].mean():.4f}")
print("Saved metrics to mean_imputation_metrics.txt")