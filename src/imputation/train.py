import random
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from utils import create_imputation_sequences, prepare_data, random_mask, remove_outliers_iqr
from model import GRU_Imputation
from dataset import ImputationDataset
import os

os.makedirs("models", exist_ok=True)

# Reproducible seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

df = pd.read_parquet("data/pivot_data.parquet")
SEQ_LEN = 24
X, M, y, target_masks, scaler, mask, df = prepare_data("data/pivot_data.parquet", SEQ_LEN)

print(f"X shape: {X.shape}")  # (samples, seq_len, features)
print(f"M shape: {M.shape}")  # (samples, seq_len, features)
print(f"y shape: {y.shape}")  # (samples, features) - target to impute
print(f"target_masks shape: {target_masks.shape}")  # (samples, features) - which targets are missing

print("Sequence shapes:")
print(f"X shape: {X.shape}")  # (samples, seq_len, features)
print(f"M shape: {M.shape}")  # (samples, seq_len, features) - mask (1 = missing, 0 = present)
print(f"y shape: {y.shape}")  # (samples, features)

# Train / test split (chronological, no shuffle)
X_train, X_test, M_train, M_test, y_train, y_test, tm_train, tm_test = train_test_split(
    X, M, y, target_masks, test_size=0.2, random_state=42, shuffle=False
)

# Train / validation split from training portion
X_train_final, X_val, M_train_final, M_val, y_train_final, y_val, tm_train_final, tm_val = train_test_split(
    X_train, M_train, y_train, tm_train, test_size=0.2, random_state=42, shuffle=False
)

# Additional random artificial masking for denoising pretraining
X_train_masked, added_mask = random_mask(X_train_final, missing_rate=0.1, seed=42)
M_train_final = np.maximum(M_train_final, added_mask)

print("Final data splits:")
print(f"Training samples: {len(X_train_final)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

print("Checking mask statistics:")
print(f"Total missing values in mask: {M.sum()}")
print(f"Missing values in last timestep: {M[:, -1, :].sum()}")
print(f"Percentage of missing values: {(M.sum() / M.size) * 100:.2f}%")

# Step-by-step mask verification
print("=== MASK VERIFICATION ===")

# 1. Original DataFrame missing values
print("1. Original DataFrame:")
print(f"   Total NaN values: {df.isna().sum().sum()}")
print(f"   DataFrame shape: {df.shape}")
print(f"   Total possible values: {df.shape[0] * df.shape[1]}")

# 2. Flat mask DataFrame check
print("\n2. Mask Array:")
print(f"   Mask shape: {mask.shape}")
print(f"   Unique values in mask: {np.unique(mask.values)}")
print(f"   Count of 1s (missing): {(mask == 1).sum().sum()}")
print(f"   Count of 0s (present): {(mask == 0).sum().sum()}")
print(f"   Total mask size: {mask.size}")

# 3. Sequence mask tensor check
print("\n3. Sequence Arrays:")
print(f"   M (sequence mask) shape: {M.shape}")
print(f"   M total 1s (missing): {(M == 1).sum()}")
print(f"   M total 0s (present): {(M == 0).sum()}")
print(f"   M total size: {M.size}")

# 4. Target masks (last step missing indicators)
print("\n4. Target Masks:")
print(f"   target_masks shape: {target_masks.shape}")
print(f"   target_masks 1s (missing): {(target_masks == 1).sum()}")
print(f"   target_masks 0s (present): {(target_masks == 0).sum()}")

# 5. Expected number of rolling sequences
expected_sequences = len(df) - SEQ_LEN + 1
print(f"\n5. Expected vs Actual:")
print(f"   Expected sequences: {expected_sequences}")
print(f"   Actual sequences: {len(X)}")
print(f"   Sequence length: {SEQ_LEN}")
print(f"   Features: {df.shape[1]}")

# 6. Size consistency check
print(f"\n6. Size Verification:")
print(f"   X.shape[0] * X.shape[1] * X.shape[2] = {X.shape[0]} * {X.shape[1]} * {X.shape[2]} = {X.shape[0] * X.shape[1] * X.shape[2]}")
print(f"   This should equal M.size = {M.size}")
print(f"   Match: {X.shape[0] * X.shape[1] * X.shape[2] == M.size}")

# After artificial masking
print("=== RANDOM MASKING (TRAIN SET) ===")
print(f"X_train_masked shape: {X_train_masked.shape}")
print(f"NaN in X_train_masked: {np.isnan(X_train_masked).sum()}")
print(f"M_train_final shape: {M_train_final.shape}")
print(f"Number of 1s (missing) in M_train_final: {(M_train_final == 1).sum()}")
print(f"Number of 0s (present) in M_train_final: {(M_train_final == 0).sum()}")
print(f"Percentage missing in M_train_final: {(M_train_final == 1).sum() / M_train_final.size * 100:.2f}%")

# Initialize model
input_size = X.shape[2]  # Number of features
model = GRU_Imputation(input_size=input_size, hidden_size=512, num_layers=3, dropout=0.3)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model input size: {input_size}")
print(f"Model: {model}")

model = model.to(device)

# Sanity checks for NaNs
print("NaN in y_train_final:", np.isnan(y_train_final).sum())
print("NaN in X_train_masked:", np.isnan(X_train_masked).sum())
print("NaN in M_train_final:", np.isnan(M_train_final).sum())

# Dataset objects
train_dataset = ImputationDataset(X_train_masked, M_train_final, y_train_final, tm_train_final)
val_dataset = ImputationDataset(X_val, M_val, y_val, tm_val)
test_dataset = ImputationDataset(X_test, M_test, y_test, tm_test)

# Data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training setup (Huber loss for robustness)
criterion = nn.HuberLoss(delta=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Denoising training parameters
denoise_rate = 0.10       # probability of artificially masking known targets at last step
loss_nat_weight = 0.5     # weight for natural missing vs artificial missing in loss
epochs = 25
patience = 3
min_delta = 1e-6

print("Starting training...")
print(f"Training on {len(train_dataset)} samples")
print(f"Validating on {len(val_dataset)} samples")
print(f"Batch size: {batch_size}")
print(f"Early stopping patience: {patience} epochs")
print("=" * 60)

def train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, epochs, patience, min_delta, model_path='models/imputation/imputation_model_gru.pth'
):
    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses, lrs = [], [], []

    for epoch in range(epochs):
        model.train()
        ep_train_loss = 0.0
        batches_train = 0
        for X_batch, M_batch, y_batch, tm_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            X_batch = X_batch.to(device)
            M_batch = M_batch.to(device)
            y_batch = y_batch.to(device)
            tm_batch = tm_batch.to(device)          # 1 = naturally missing at last step

            # Artificial mask over known targets (tm_batch == 0)
            present = (tm_batch == 0)               # [B, F]
            rand = torch.rand_like(present.float())
            art_mask = (rand < denoise_rate) & present  # [B, F] artificially missing

            # Prepare inputs: set artificially missing to zero; set mask to 1
            X_input = X_batch.clone()
            M_input = M_batch.clone()
            # convention: missing => X=0, M=1
            if art_mask.any():
                X_input[:, -1, :] = torch.where(art_mask, torch.zeros_like(X_input[:, -1, :]), X_input[:, -1, :])
                M_input[:, -1, :] = torch.where(art_mask, torch.ones_like(M_input[:, -1, :]), M_input[:, -1, :])

            optimizer.zero_grad()
            outputs = model(X_input, M_input)

            has_nat = (tm_batch == 1).any()
            has_art = art_mask.any()

            # Skip batch with no supervised targets
            if not (has_nat or has_art):
                continue

            # Loss on naturally missing targets
            loss_nat = criterion(outputs[tm_batch == 1], y_batch[tm_batch == 1]) if has_nat else torch.tensor(0.0, device=device)
            # Loss on artificially masked targets
            loss_art = criterion(outputs[art_mask], y_batch[art_mask]) if has_art else torch.tensor(0.0, device=device)

            # Combined loss
            loss = loss_nat_weight * loss_nat + (1 - loss_nat_weight) * loss_art

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_train_loss += loss.item()
            batches_train += 1

        avg_train = ep_train_loss / batches_train if batches_train else float('inf')

        # Validation (only natural missing positions)
        model.eval()
        ep_val_loss = 0.0
        batches_val = 0
        with torch.no_grad():
            for X_batch, M_batch, y_batch, tm_batch in tqdm(val_loader, desc="Validation", leave=False):
                X_batch = X_batch.to(device)
                M_batch = M_batch.to(device)
                y_batch = y_batch.to(device)
                tm_batch = tm_batch.to(device)

                outputs = model(X_batch, M_batch)
                miss_mask = tm_batch
                if miss_mask.sum() > 0:
                    vloss = criterion(outputs[miss_mask == 1], y_batch[miss_mask == 1])
                    ep_val_loss += vloss.item()
                    batches_val += 1

        avg_val = ep_val_loss / batches_val if batches_val else float('inf')
        scheduler.step(avg_val)
        lr = optimizer.param_groups[0]['lr']

        # Early stopping check
        if avg_val < best_loss - min_delta:
            best_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"Epoch {epoch+1}: new best val {best_loss:.6f} saved")
            print(f'Train Loss: {avg_train:.6f} (batches: {batches_train}/{len(train_loader)})')
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: no improvement ({patience_counter}/{patience})")

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        lrs.append(lr)

        if patience_counter >= patience or lr < 1e-7:
            print("Early stopping.")
            break

    return train_losses, val_losses, lrs, best_loss

train_losses, val_losses, learning_rates, best_loss = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, epochs, patience, min_delta
)

# Plot loss curves
plt.figure(figsize=(7,4))
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Val')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss curves (denoising)')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print("Done. Best val loss:", best_loss)