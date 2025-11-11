from matplotlib import pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import prepare_data
from model import GRU_Imputation
from dataset import ImputationDataset

SEQ_LEN = 24
X, M, y, target_masks, scaler, mask, df = prepare_data("data/pivot_data.parquet", SEQ_LEN)

# Load the best model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GRU_Imputation(input_size=X.shape[2], hidden_size=256, num_layers=3, dropout=0.3)
model.load_state_dict(torch.load('models/imputation/imputation_model_gru.pth'))
model = model.to(device)
model.eval()

# Create dataset and dataloader for testing
test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Evaluate the model
test_loss = 0.0
test_batches = 0
criterion = torch.nn.MSELoss()

with torch.no_grad():
    for batch_data in test_loader:
        X_batch, M_batch, y_batch, tm_batch = batch_data
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        y_batch = y_batch.to(device)
        tm_batch = tm_batch.to(device)

        outputs = model(X_batch, M_batch)
        missing_mask = tm_batch

        if missing_mask.sum() > 0:
            loss = criterion(outputs[missing_mask == 1], y_batch[missing_mask == 1])
            test_loss += loss.item()
            test_batches += 1

# Calculate average test loss
avg_test_loss = test_loss / test_batches if test_batches > 0 else float('inf')
print(f"Test Loss: {avg_test_loss:.6f}")


all_predictions = []
all_targets = []
all_masks = []

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

# Visualization for 3 columns (change indices as needed)
cols_to_plot = [63, 52, 74]
plt.figure(figsize=(18, 5))
for i, col in enumerate(cols_to_plot):
    idx = masks[:, col] == 1  # Only where value was missing
    if idx.sum() == 0:
        continue
    plt.subplot(1, 3, i+1)
    plt.scatter(np.arange(idx.sum()), targets[idx, col], color='tab:blue', label='Actual', alpha=0.7, s=20)
    plt.scatter(np.arange(idx.sum()), predictions[idx, col], color='tab:orange', label='Imputed', alpha=0.7, s=20)
    plt.title(f'Column {col}')
    plt.xlabel('Missing Value Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.tight_layout()

plt.suptitle('Actual vs Imputed Values for Missing Data (3 Columns)', fontsize=16, y=1.05)
plt.show()