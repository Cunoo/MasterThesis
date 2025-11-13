from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from utils import prepare_data
from model import GRU_Imputation
from dataset import ImputationDataset
from torch.utils.data import DataLoader
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

SEQ_LEN = 24
ART_RATE = 0.10  # fraction of artificially masked known targets at the last time step

# Data
X, M, y, target_masks, scaler, mask, df = prepare_data("data/pivot_data.parquet", SEQ_LEN)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRU_Imputation(input_size=X.shape[2], hidden_size=512, num_layers=3, dropout=0.3)
model.load_state_dict(torch.load('models/imputation/imputation_model_gru.pth', map_location=device))
model = model.to(device)
model.eval()

# Loader
test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate test loss on natural missing
test_loss = 0.0
test_batches = 0
criterion = nn.HuberLoss(delta=0.05)
with torch.no_grad():
    for X_batch, M_batch, y_batch, tm_batch in test_loader:
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        y_batch = y_batch.to(device)
        tm_batch = tm_batch.to(device)

        outputs = model(X_batch, M_batch)
        miss_mask = tm_batch  # 1 = missing at the last time step
        if miss_mask.sum() > 0:
            loss = criterion(outputs[miss_mask == 1], y_batch[miss_mask == 1])
            test_loss += loss.item()
            test_batches += 1

avg_test_loss = test_loss / test_batches if test_batches > 0 else float('inf')
print(f"Test Loss (natural missing): {avg_test_loss:.6f}")

# NATURAL missing metrics
all_predictions, all_targets, all_masks = [], [], []
with torch.no_grad():
    for X_batch, M_batch, y_batch, tm_batch in test_loader:
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        y_batch = y_batch.to(device)
        tm_batch = tm_batch.to(device)
        outputs = model(X_batch, M_batch)
        all_predictions.append(outputs.cpu().numpy())
        all_targets.append(y_batch.cpu().numpy())
        all_masks.append(tm_batch.cpu().numpy())

predictions = np.concatenate(all_predictions, axis=0)
targets = np.concatenate(all_targets, axis=0)
masks = np.concatenate(all_masks, axis=0)

print("\n=== Metrics on NATURAL missing values (original NaNs) ===")
for col, feature_name in enumerate(df.columns):
    col_mask = masks[:, col] == 1
    if not np.any(col_mask):
        continue
    y_true = targets[col_mask, col]
    y_pred = predictions[col_mask, col]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{col:02d}] {feature_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

# ARTIFICIAL missing metrics
all_preds_art, all_targs_art, all_flags_art = [], [], []
with torch.no_grad():
    for X_batch, M_batch, y_batch, tm_batch in test_loader:
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        y_batch = y_batch.to(device)
        tm_batch = tm_batch.to(device)

        present = (tm_batch == 0)  # positions where the target is known
        rand = torch.rand_like(present.float())
        art_mask = (rand < ART_RATE) & present  # artificially missing at the last time step

        X_masked = X_batch.clone()
        M_masked = M_batch.clone()
        if art_mask.any():
            # convention: missing => X=0, M=1
            X_masked[:, -1, :] = torch.where(art_mask, torch.zeros_like(X_masked[:, -1, :]), X_masked[:, -1, :])
            M_masked[:, -1, :] = torch.where(art_mask, torch.ones_like(M_masked[:, -1, :]), M_masked[:, -1, :])

        outputs = model(X_masked, M_masked)

        all_preds_art.append(outputs.cpu().numpy())
        all_targs_art.append(y_batch.cpu().numpy())
        all_flags_art.append(art_mask.cpu().numpy().astype(np.uint8))

preds_art = np.concatenate(all_preds_art, axis=0)
targs_art = np.concatenate(all_targs_art, axis=0)
flags_art = np.concatenate(all_flags_art, axis=0)

print("\n=== Metrics on ARTIFICIALLY masked, known values (last step) ===")
for col, feature_name in enumerate(df.columns):
    col_mask = flags_art[:, col] == 1
    if not np.any(col_mask):
        continue
    y_true = targs_art[col_mask, col]
    y_pred = preds_art[col_mask, col]
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{col:02d}] {feature_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

# Visualization for 3 columns (NATURAL missing)
cols_to_plot = [3, 14, 69]
plt.figure(figsize=(18, 5))
for i, col in enumerate(cols_to_plot):
    idx = masks[:, col] == 1  # only where it was naturally missing
    if idx.sum() == 0:
        continue
    plt.subplot(1, 3, i+1)
    plt.scatter(np.arange(idx.sum()), targets[idx, col], color='tab:blue', label='Actual', alpha=0.7, s=20)
    plt.scatter(np.arange(idx.sum()), predictions[idx, col], color='tab:orange', label='Imputed', alpha=0.7, s=20)
    plt.title(f'Column {df.columns[col]}')
    plt.xlabel('Missing Value Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.tight_layout()

plt.suptitle('Actual vs Imputed Values for Missing Data (3 Columns)', fontsize=16, y=1.05)
plt.show()


cols_to_plot_art = [3, 14, 69]
plt.figure(figsize=(18, 5))
for i, col in enumerate(cols_to_plot_art):
    idx = flags_art[:, col] == 1  # umelo skryté
    if idx.sum() == 0:
        continue
    plt.subplot(1, 3, i+1)
    plt.scatter(np.arange(idx.sum()), targs_art[idx, col], color='tab:blue', label='Actual', alpha=0.7, s=20)
    plt.scatter(np.arange(idx.sum()), preds_art[idx, col], color='tab:orange', label='Imputed', alpha=0.7, s=20)
    plt.title(f'Artificially masked {df.columns[col]}')
    plt.xlabel('Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.tight_layout()
plt.suptitle('Actual vs Imputed (Artificially Masked)', fontsize=16, y=1.05)
plt.show()