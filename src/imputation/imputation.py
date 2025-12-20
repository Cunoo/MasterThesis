import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import prepare_data
from model import GRU_Imputation
from dataset import ImputationDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from matplotlib import pyplot as plt
import random
import model_param
SEQ_LEN = model_param.SEQ_LEN

# Load and prepare data
print("Loading and preparing data...")
X, M, y, target_masks, scaler, mask, df, seq_to_orig_idx = prepare_data("data/pivot_data.parquet", SEQ_LEN, impute_mode=True)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
input_size = X.shape[2]
model = GRU_Imputation(input_size=input_size, hidden_size=model_param.hidden_size, num_layers=model_param.num_layers, dropout=model_param.dropout)
model.load_state_dict(torch.load('models/imputation/imputation_model_gru.pth', map_location=device))
model = model.to(device)
model.eval()

# Prepare data for imputation
test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Collect all imputed values
all_predictions = []

print("Running inference...")
with torch.no_grad():
    for batch_idx, (X_batch, M_batch, y_batch, tm_batch) in enumerate(tqdm(test_loader, desc="Imputing")):
        X_batch = X_batch.to(device)
        M_batch = M_batch.to(device)
        outputs = model(X_batch, M_batch)
        all_predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(all_predictions, axis=0)

# Read original data (bez časových features)
df_original = pd.read_parquet("data/pivot_data.parquet")

# Create a copy to impute (na df s časovými features!)
df_imputed = df.copy()

# Track which values were imputed for visualization
imputed_mask = np.zeros((len(df), len(df.columns)), dtype=bool)

# For each prediction, replace missing values with imputed values
print("Filling missing values...")
mask_np = mask.values
for seq_idx, original_idx in enumerate(tqdm(seq_to_orig_idx, desc="Processing")):
    if original_idx < len(df):
        # Find which values were missing in the original sequence
        for col in range(df.shape[1]):
            if mask_np[original_idx, col] == 1:  # Bolo chýbajúce (z pôvodného masku)
                # Scale back the prediction to original scale
                imputed_value = scaler.inverse_transform(
                    predictions[seq_idx].reshape(1, -1)
                )[0, col]
                df_imputed.iloc[original_idx, col] = imputed_value
                imputed_mask[original_idx, col] = True

# Drop časové features pred uložením (aby sa zhodoval s pôvodným)
df_imputed_clean = df_imputed.drop(columns=['sin_doy', 'cos_doy', 'sin_hour', 'cos_hour'])

# Save imputed dataset
print("Saving imputed dataset...")
df_imputed_clean.to_parquet("data/pivot_data_imputed.parquet")
print(f"Imputed dataset saved to: data/pivot_data_imputed.parquet")

# Print statistics
print(f"\nOriginal missing values: {df_original.isna().sum().sum()}")
print(f"Imputed dataset missing values: {df_imputed_clean.isna().sum().sum()}")
print(f"Values imputed: {imputed_mask.sum()}")

print(f"\nPredictions stats:")
print(f"Min: {predictions.min()}")
print(f"Max: {predictions.max()}")
print(f"Mean: {predictions.mean()}")
print(f"Unique values count: {len(np.unique(predictions[:, 0]))}")
print(f"First prediction sample: {predictions[0, :5]}")

# Visualization - random column (z pôvodných stĺpcov, nie časových features)
original_cols = [i for i, col in enumerate(df.columns) if col not in ['sin_doy', 'cos_doy', 'sin_hour', 'cos_hour']]
random_col = random.choice(original_cols)
col_name = df.columns[random_col]

print(f"\nPlotting column: {col_name}")

plt.figure(figsize=(16, 5))

# Extract data for this column
original_values = df.iloc[:, random_col].to_numpy().astype(float)
imputed_values = df_imputed.iloc[:, random_col].to_numpy().astype(float)
col_imputed_mask = imputed_mask[:, random_col]

# Plot original data (blue)
plt.plot(np.arange(len(original_values)), original_values, 
         color='tab:blue', label='Original data', alpha=0.7, linewidth=1)

# Plot imputed data (orange dots where data was missing)
imputed_indices = np.where(col_imputed_mask)[0]
plt.scatter(imputed_indices, imputed_values[imputed_indices], 
           color='tab:orange', label='Imputed values', alpha=0.8, s=30, zorder=5)

# Statistics
print(f"\nMissing values by row index:")
missing_by_row = df_imputed_clean.isna().sum(axis=1)
print(missing_by_row[missing_by_row > 0].head(30))

print(f"\nFirst {SEQ_LEN-1} rows cannot be imputed (no history):")
print(f"Expected remaining NaN in first {SEQ_LEN-1} rows: ~{df_original.iloc[:SEQ_LEN-1].isna().sum().sum()}")

print(f"\nLast row NaN count: {df_original.iloc[-1].isna().sum()}")

plt.xlabel('Time index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title(f'Column: {col_name} - Original vs Imputed Data', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('imputation_visualization.png', dpi=100)
print(f"✓ Visualization saved to: imputation_visualization.png")
plt.show()
