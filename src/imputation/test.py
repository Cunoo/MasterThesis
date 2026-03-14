from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from utils import prepare_data
from model import GRU_Imputation
from dataset import ImputationDataset
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import model_param as model_param
from test_log import append_test_log, TEST_LOG_PATH

# Nastavenia
SEQ_LEN = model_param.SEQ_LEN
ART_RATE = 0.15  # Pomer umelo maskovaných hodnôt pre test presnosti
torch.manual_seed(42) # Fixný seed pre reprodukovateľnosť testu
np.random.seed(42)
EXCLUDED_FEATURES = {"sin_doy", "cos_doy", "sin_hour", "cos_hour"}

# 1. Načítanie dát
print("Loading data and preparing sequences...")

scaler_path = os.path.join("models", "imputation", "scaler.pkl")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(
        f"Scaler not found at {scaler_path}. Run train.py first to generate it."
    )

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Načítanie testovacích dát (fit_scaler=False, použijeme načítaný scaler)
X, M, y, target_masks, _, mask, df, seq_to_orig_idx = prepare_data(
    "data/pivot_test.parquet", SEQ_LEN, scaler=scaler, fit_scaler=False, verbose=False
)

# 2. Inicializácia a načítanie modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X.shape[2]
model = GRU_Imputation(input_size=input_size, hidden_size=model_param.hidden_size, dropout=model_param.dropout)

model_path = 'models/imputation/imputation_model_gru_without_attention.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    print("Error: Model file not found!")
    exit()

model = model.to(device)
model.eval()

# 3. Príprava DataLoader-a
test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# 4. Inferenčný cyklus (Artificial Masking Test)
all_preds_scaled, all_targs_scaled, all_flags_art = [], [], []

print("Running inference on test set with artificial masking...")
with torch.no_grad():
    for X_batch, M_batch, y_batch, tm_batch in test_loader:
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        y_batch = y_batch.to(device)
        tm_batch = tm_batch.to(device)

        # Vytvorenie umelej masky len na pozíciách, ktoré v realite poznáme (tm_batch == 0)
        present = (tm_batch == 0)
        rand = torch.rand_like(present.float())
        art_mask = (rand < ART_RATE) & present

        X_masked = X_batch.clone()
        M_masked = M_batch.clone()
        
        # Aplikácia masky: X nastavíme na 0 (priemer), M nastavíme na 1
        if art_mask.any():
            X_masked[:, -1, :] = torch.where(art_mask, torch.zeros_like(X_masked[:, -1, :]), X_masked[:, -1, :])
            M_masked[:, -1, :] = torch.where(art_mask, torch.ones_like(M_masked[:, -1, :]), M_masked[:, -1, :])

        outputs = model(X_masked, M_masked)

        all_preds_scaled.append(outputs.cpu().numpy())
        all_targs_scaled.append(y_batch.cpu().numpy())
        all_flags_art.append(art_mask.cpu().numpy().astype(np.uint8))

# Spojenie výsledkov
preds_scaled = np.concatenate(all_preds_scaled, axis=0)
targs_scaled = np.concatenate(all_targs_scaled, axis=0)
flags_art = np.concatenate(all_flags_art, axis=0)

# 5. INVERZNÁ TRANSFORMÁCIA (Návrat k reálnym jednotkám)
print("Inverting scale to real-world units...")

# Bezpečná manuálna inverzia pre StandardScaler
if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
    # x_orig = x_scaled * std + mean
    preds_orig = preds_scaled * scaler.scale_ + scaler.mean_
    targs_orig = targs_scaled * scaler.scale_ + scaler.mean_
else:
    # Fallback na štandardnú metódu (ak by to bol iný scaler)
    print("Warning: Scaler attributes not found, using inverse_transform...")
    preds_orig = scaler.inverse_transform(preds_scaled)
    targs_orig = scaler.inverse_transform(targs_scaled)

# 6. Výpočet metrík na reálnych hodnotách
print("\n" + "="*60)
print(f"{'Feature Name':<30} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
print("-" * 60)

results = []
for col, feature_name in enumerate(df.columns):
    if feature_name in EXCLUDED_FEATURES:
        continue

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

# Uloženie metrík do súboru
metrics_df = pd.DataFrame(results)
metrics_df = metrics_df[~metrics_df["Feature"].isin(EXCLUDED_FEATURES)].reset_index(drop=True)
append_test_log(
    log_path=TEST_LOG_PATH,
    model_name=model.__class__.__name__,
    model_file=os.path.basename(model_path),
    metrics_df=metrics_df,
)
print(f"Test log saved to {TEST_LOG_PATH}")

print("="*60)
if not metrics_df.empty:
    print(f"Average R2 Score: {metrics_df['R2'].mean():.4f}")
else:
    print("Average R2 Score: N/A (no features after exclusion)")
print("Saved metrics to gru_metrics_real_units_imputation_model_gru_without_attention.txt")

# 7. Vizualizácia výsledkov
# Vyberieme 3 zaujímavé stĺpce (nie časové features na konci)
cols_to_plot = [0, 5, 10] 
# Uistíme sa, že indexy sú v rozsahu
cols_to_plot = [c for c in cols_to_plot if c < len(df.columns)]

plt.figure(figsize=(18, 6))

for i, col in enumerate(cols_to_plot):
    idx = flags_art[:, col] == 1
    if idx.sum() == 0:
        continue
    
    plt.subplot(1, 3, i+1)
    # Zobraziť len prvých 100 bodov pre prehľadnosť
    plot_limit = 100
    y_t = targs_orig[idx, col][:plot_limit]
    y_p = preds_orig[idx, col][:plot_limit]
    
    plt.scatter(np.arange(len(y_t)), y_t, color='tab:blue', label='Actual', alpha=0.7, s=30)
    plt.scatter(np.arange(len(y_p)), y_p, color='tab:orange', label='Imputed', alpha=0.7, s=30)
    
    plt.title(f'Feature: {df.columns[col]}')
    plt.xlabel('Sample Index (Masked points)')
    plt.ylabel('Real Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('Model Performance on Artificially Masked Data (Real Units)', fontsize=16)
plt.savefig('imputation_test_results_imputation_model_gru_without_attention.png')
# plt.show() # Odkomentujte ak bežíte lokálne s GUI
print("Plot saved to imputation_test_results_imputation_model_gru_without_attention.png")