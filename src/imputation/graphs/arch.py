
    
# visualization_imputation_specific_features.py
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import sys
import torch

# Pridaj cestu k modulom - ideme o level vyššie na imputation/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # graphs/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # imputation/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # src/

from utils import prepare_data
from model import GRU_Imputation
from dataset import ImputationDataset
from torch.utils.data import DataLoader
import model_param as model_param

# Nastavenia
SEQ_LEN = model_param.SEQ_LEN
ART_RATE = 0.15
torch.manual_seed(42)
np.random.seed(42)
EXCLUDED_FEATURES = {"sin_doy", "cos_doy", "sin_hour", "cos_hour"}

# DEFINUJ STLPCE KTORE CHCES VIZUALIZOVAT
FEATURES_TO_PLOT = [
    "SSA 4 Schacht ec (1) 120cm mS/cm",
    "SSA 4 Schacht UMP (1) 120cm %",
    "SSA 4 Schacht UMP (1) 75cm %"
]

# Nastavenie ciest - ideme z graphs/ na koreň
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCALER_PATH = os.path.join(root_path, "models", "imputation", "scaler.pkl")
DATA_PATH = os.path.join(root_path, "data", "pivot_test.parquet")
MODEL_PATH = os.path.join(root_path, "models", "imputation", "imputation_model__gru_and_attention.pth")
print("Loading data and preparing sequences...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X, M, y, target_masks, _, mask, df, seq_to_orig_idx = prepare_data(
    DATA_PATH, SEQ_LEN, scaler=scaler, fit_scaler=False, verbose=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X.shape[2]
model = GRU_Imputation(input_size=input_size, hidden_size=model_param.hidden_size, dropout=model_param.dropout)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


all_preds_scaled, all_targs_scaled, all_flags_art = [], [], []

print("Running inference...")
with torch.no_grad():
    for X_batch, M_batch, y_batch, tm_batch in test_loader:
        X_batch, M_batch, y_batch, tm_batch = X_batch.to(device), M_batch.to(device), y_batch.to(device), tm_batch.to(device)
        
        present = (tm_batch == 0)
        rand = torch.rand_like(present.float())
        art_mask = (rand < ART_RATE) & present
        
        X_masked = X_batch.clone()
        M_masked = M_batch.clone()
        if art_mask.any():
            X_masked[:, -1, :] = torch.where(art_mask, torch.zeros_like(X_masked[:, -1, :]), X_masked[:, -1, :])
            M_masked[:, -1, :] = torch.where(art_mask, torch.ones_like(M_masked[:, -1, :]), M_masked[:, -1, :])
        
        outputs = model(X_masked, M_masked)
        all_preds_scaled.append(outputs.cpu().numpy())
        all_targs_scaled.append(y_batch.cpu().numpy())
        all_flags_art.append(art_mask.cpu().numpy().astype(np.uint8))

preds_scaled = np.concatenate(all_preds_scaled, axis=0)
targs_scaled = np.concatenate(all_targs_scaled, axis=0)
flags_art = np.concatenate(all_flags_art, axis=0)

# 5. Inverzia
preds_orig = preds_scaled * scaler.scale_ + scaler.mean_
targs_orig = targs_scaled * scaler.scale_ + scaler.mean_

# 6. Nájdenie stĺpcov podľa názvu
cols_to_plot = []
for feature_name in FEATURES_TO_PLOT:
    if feature_name in df.columns:
        col_idx = df.columns.get_loc(feature_name)
        cols_to_plot.append((col_idx, feature_name))
    else:
        print(f"Warning: Feature '{feature_name}' not found in dataframe!")

# 7. Vizualizácia
plt.figure(figsize=(18, 6))

for i, (col, feature_name) in enumerate(cols_to_plot):
    idx = flags_art[:, col] == 1
    if idx.sum() == 0:
        continue
    
    plt.subplot(1, 3, i+1)
    plot_limit = 500
    y_t = targs_orig[idx, col][:plot_limit]
    y_p = preds_orig[idx, col][:plot_limit]
    
    plt.scatter(np.arange(len(y_t)), y_t, color='tab:blue', label='Actual', alpha=0.7, s=30)
    plt.scatter(np.arange(len(y_p)), y_p, color='tab:orange', label='Imputed', alpha=0.7, s=30)
    
    plt.title(f'{feature_name}')
    plt.xlabel('Sample Index (Masked points)')
    plt.ylabel('Real Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.suptitle('Imputation Results - Specific Features', fontsize=16)
plt.tight_layout()
plt.savefig('imputation_visualization_specific_features_gru_and_attention.png', dpi=150)
print("Plot saved to imputation_visualization_specific_features_gru_and_attention.png")