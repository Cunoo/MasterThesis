from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from utils import create_imputation_sequences, prepare_data, remove_outliers_iqr
from model import GRU_Imputation
from dataset import ImputationDataset

# Load data
df = pd.read_parquet("data/pivot_data.parquet")
import os
os.makedirs("models", exist_ok=True)

SEQ_LEN = 24

X, M, y, target_masks, scaler, mask, df = prepare_data("data/pivot_data.parquet", SEQ_LEN)


print(f"X shape: {X.shape}")  # (samples, seq_len, features)
print(f"M shape: {M.shape}")  # (samples, seq_len, features) 
print(f"y shape: {y.shape}")  # (samples, features) - target to impute
print(f"target_masks shape: {target_masks.shape}")  # (samples, features) - what to impute

print("Sequence shapes:")
print(f"X shape: {X.shape}")  # (samples, seq_len, features)
print(f"M shape: {M.shape}")  # (samples, seq_len, features) - mask
print(f"y shape: {y.shape}")  # (samples, features)

X_train, X_test, M_train, M_test, y_train, y_test, tm_train, tm_test = train_test_split(
    X, M, y, target_masks, test_size=0.2, random_state=42, shuffle=False
)

X_train_final, X_val, M_train_final, M_val, y_train_final, y_val, tm_train_final, tm_val = train_test_split(
    X_train, M_train, y_train, tm_train, test_size=0.2, random_state=42, shuffle=False
)

print("Final data splits:")
print(f"Training samples: {len(X_train_final)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

print("Checking mask statistics:")
print(f"Total missing values in mask: {M.sum()}")
print(f"Missing values in last timestep: {M[:, -1, :].sum()}")
print(f"Percentage of missing values: {(M.sum() / M.size) * 100:.2f}%")

# Verify mask statistics step by step
print("=== MASK VERIFICATION ===")

# 1. Original DataFrame missing values
print("1. Original DataFrame:")
print(f"   Total NaN values: {df.isna().sum().sum()}")
print(f"   DataFrame shape: {df.shape}")
print(f"   Total possible values: {df.shape[0] * df.shape[1]}")

# 2. Mask array check
print("\n2. Mask Array:")
print(f"   Mask shape: {mask.shape}")
print(f"   Unique values in mask: {np.unique(mask.values)}")
print(f"   Count of 1s (missing): {(mask == 1).sum().sum()}")
print(f"   Count of 0s (present): {(mask == 0).sum().sum()}")
print(f"   Total mask size: {mask.size}")

# 3. Sequence arrays check
print("\n3. Sequence Arrays:")
print(f"   M (sequence mask) shape: {M.shape}")
print(f"   M total 1s (missing): {(M == 1).sum()}")
print(f"   M total 0s (present): {(M == 0).sum()}")
print(f"   M total size: {M.size}")

# 4. Target masks check
print("\n4. Target Masks:")
print(f"   target_masks shape: {target_masks.shape}")
print(f"   target_masks 1s (missing): {(target_masks == 1).sum()}")
print(f"   target_masks 0s (present): {(target_masks == 0).sum()}")

# 5. Calculate expected values
expected_sequences = len(df) - SEQ_LEN + 1
print(f"\n5. Expected vs Actual:")
print(f"   Expected sequences: {expected_sequences}")
print(f"   Actual sequences: {len(X)}")
print(f"   Sequence length: {SEQ_LEN}")
print(f"   Features: {df.shape[1]}")

# 6. Verify the multiplication
print(f"\n6. Size Verification:")
print(f"   X.shape[0] * X.shape[1] * X.shape[2] = {X.shape[0]} * {X.shape[1]} * {X.shape[2]} = {X.shape[0] * X.shape[1] * X.shape[2]}")
print(f"   This should equal M.size = {M.size}")
print(f"   Match: {X.shape[0] * X.shape[1] * X.shape[2] == M.size}")

# Initialize model
input_size = X.shape[2]  # Number of features (75)
model = GRU_Imputation(input_size=input_size, hidden_size=256, num_layers=3, dropout=0.3)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model input size: {input_size}")
print(f"Model: {model}")

model = model.to(device)

# Create datasets
train_dataset = ImputationDataset(X_train_final, M_train_final, y_train_final, tm_train_final)
val_dataset = ImputationDataset(X_val, M_val, y_val, tm_val)
test_dataset = ImputationDataset(X_test, M_test, y_test, tm_test)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training parameters
epochs = 15
best_loss = float('inf')
patience = 3  # Early stopping patience
patience_counter = 0  # Counter for patience
min_delta = 1e-6  # Minimum improvement to count as progress


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
    train_losses = []
    val_losses = []
    learning_rates = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        num_train_batches = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', leave=False)
        for batch_data in train_pbar:
            X_batch, M_batch, y_batch, tm_batch = batch_data
            X_batch = X_batch.to(device)
            M_batch = M_batch.to(device)
            y_batch = y_batch.to(device)
            tm_batch = tm_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch, M_batch)
            missing_mask = tm_batch

            if missing_mask.sum() > 0:
                loss = criterion(outputs[missing_mask == 1], y_batch[missing_mask == 1])
            else:
                continue

            if torch.isnan(loss):
                print(f"NaN loss detected in epoch {epoch+1}, skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_train_batches += 1
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'missing': f'{missing_mask.sum().item()}'
            })

        # Validation
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False)
        with torch.no_grad():
            for batch_data in val_pbar:
                X_batch, M_batch, y_batch, tm_batch = batch_data
                X_batch = X_batch.to(device)
                M_batch = M_batch.to(device)
                y_batch = y_batch.to(device)
                tm_batch = tm_batch.to(device)

                outputs = model(X_batch, M_batch)
                missing_mask = tm_batch

                if missing_mask.sum() > 0:
                    loss = criterion(outputs[missing_mask == 1], y_batch[missing_mask == 1])
                    val_loss += loss.item()
                    num_val_batches += 1
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'missing': f'{missing_mask.sum().item()}'
                    })

        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else float('inf')
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if avg_val_loss < best_loss - min_delta:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved! Val Loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter}/{patience} epochs")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        learning_rates.append(current_lr)

        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {avg_train_loss:.6f} (batches: {num_train_batches}/{len(train_loader)})')
        print(f'Val Loss: {avg_val_loss:.6f} (batches: {num_val_batches}/{len(val_loader)})')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'Best Val Loss: {best_loss:.6f}')
        print(f'Patience: {patience_counter}/{patience}')
        print(f'Model saved: {"YES" if patience_counter == 0 else "NO"}')
        print('=' * 60)

        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} epochs")
            break
        if current_lr < 1e-7:
            print("Learning rate too low, stopping training")
            break

        print(f"Training completed!")
        print(f"Total epochs: {epoch + 1}")
        print(f"Best validation loss: {best_loss:.6f}")
        print(f"Final learning rate: {current_lr:.2e}")

    return train_losses, val_losses, learning_rates, best_loss

train_losses, val_losses, learning_rates, best_loss = train_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler,
    device, epochs, patience, min_delta
)
    
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
plt.close()
print("Loss curve saved as loss_curve.png")

