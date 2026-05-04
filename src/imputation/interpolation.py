import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Importy z projektu
from utils import prepare_data
from dataset import ImputationDataset
import model_param

# ============================================================================
# KONFIGURÁCIA
# ============================================================================

SEQ_LEN = model_param.SEQ_LEN
ART_RATE = 0.15  # Pomer maskovaných hodnôt pre test
EXCLUDED_FEATURES = {"sin_doy", "cos_doy", "sin_hour", "cos_hour"}
torch.manual_seed(42)
np.random.seed(42)

METHODS = ['linear', 'forward_fill']  # Testované metódy
RESULTS_DIR = "models/imputation"
RESULTS_FILE = os.path.join(RESULTS_DIR, "interpolation_results.txt")

# ============================================================================
# HELPER FUNKCIE
# ============================================================================

def append_result_log(log_path: str, method: str, metrics: dict) -> None:
    """Zapis výsledkov do logu"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] Method: {method}\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\n")


def interpolate_sequence(X_batch: np.ndarray, method: str) -> np.ndarray:
    """
    Interpoluj sekvenciu podľa zvolenej metódy
    X_batch: shape (seq_len, features)
    """
    df_batch = pd.DataFrame(X_batch)
    
    if method == 'linear':
        df_interp = df_batch.interpolate(method='linear', axis=0)
    elif method == 'forward_fill':
        df_interp = df_batch.ffill(axis=0)
    else:
        raise ValueError(f"Neznáma metóda: {method}")
    
    # Ošetri chýbajúce hodnoty na začiatku/konci.
    # Ak je cely stlpec v sekvencii NaN, ostane NaN aj po bfill/ffill,
    # preto davame fallback 0.0 (v scaled priestore je to priemer).
    df_interp = df_interp.bfill(axis=0).ffill(axis=0).fillna(0.0)
    
    return df_interp.values


def get_interpolation_predictions(
    X: np.ndarray,
    M: np.ndarray,
    y: np.ndarray,
    target_masks: np.ndarray,
    flags_art: np.ndarray,
    method: str,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vytvor interpolované predikcie v rovnakom režime ako GRU test.py:
    - DataLoader(batch_size=256, shuffle=False)
    - umelé maskovanie len na poslednom timestepe podľa tm_batch
    - evaluácia na rovnakých pozíciách (flags_art)
    """
    test_dataset = ImputationDataset(X, M, y, target_masks)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_preds_scaled, all_targs_scaled, all_flags_art = [], [], []
    cursor = 0

    for X_batch, M_batch, y_batch, tm_batch in tqdm(
        test_loader, desc=f"Interpolácia [{method}]", leave=False
    ):
        # Použi predgenerovanú masku (rovnakú pre všetky metódy).
        bs = X_batch.shape[0]
        art_mask_np = flags_art[cursor:cursor + bs]
        cursor += bs

        X_np = X_batch.numpy()
        M_np = M_batch.numpy()
        y_np = y_batch.numpy()

        batch_preds = np.zeros_like(y_np)
        for j in range(X_np.shape[0]):
            seq = X_np[j].copy()
            # Obnov NaN pre prirodzene missing hodnoty.
            seq[M_np[j] == 1] = np.nan
            # Umele maskovanie len na poslednom timestepe.
            seq[-1, art_mask_np[j].astype(bool)] = np.nan

            seq_interp = interpolate_sequence(seq, method)
            batch_preds[j] = seq_interp[-1, :]

        all_preds_scaled.append(batch_preds)
        all_targs_scaled.append(y_np)
        all_flags_art.append(art_mask_np)

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    targs_scaled = np.concatenate(all_targs_scaled, axis=0)
    flags_art = np.concatenate(all_flags_art, axis=0)
    return preds_scaled, targs_scaled, flags_art


def build_gru_style_art_masks(target_masks: np.ndarray, art_rate: float, batch_size: int = 256) -> np.ndarray:
    """Generate artificial masks exactly like GRU test.py (single pass, last timestep features)."""
    test_dataset = ImputationDataset(
        np.zeros((target_masks.shape[0], 1, target_masks.shape[1]), dtype=np.float32),
        np.zeros((target_masks.shape[0], 1, target_masks.shape[1]), dtype=np.float32),
        np.zeros((target_masks.shape[0], target_masks.shape[1]), dtype=np.float32),
        target_masks,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    all_flags_art = []
    for _, _, _, tm_batch in test_loader:
        present = (tm_batch == 0)
        rand = torch.rand_like(present.float())
        art_mask = (rand < art_rate) & present
        all_flags_art.append(art_mask.numpy().astype(np.uint8))

    return np.concatenate(all_flags_art, axis=0)


def evaluate_method(
    preds_scaled: np.ndarray,
    targs_scaled: np.ndarray,
    flags_art: np.ndarray,
    scaler,
    feature_names,
    method: str,
) -> dict:
    """
    Evaluuj metódu na škálovaných aj originálnych dátach
    """
    # Bezpečná inverzia škálovania
    if not (hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_')):
        raise ValueError("Scaler bez mean_ a scale_")
    
    if preds_scaled.size == 0 or targs_scaled.size == 0 or flags_art.size == 0:
        print(f"Varovanie: Žiadne hodnoty na evaluáciu pre {method}")
        return {}

    # Inverzia: x_orig = x_scaled * std + mean
    preds_real = preds_scaled * scaler.scale_ + scaler.mean_
    targs_real = targs_scaled * scaler.scale_ + scaler.mean_

    # Hodnotíme len umelo maskované pozície + konečné hodnoty.
    art_mask = flags_art.astype(bool)
    finite_mask = (
        np.isfinite(targs_scaled)
        & np.isfinite(preds_scaled)
        & np.isfinite(targs_real)
        & np.isfinite(preds_real)
    )
    eval_mask = art_mask & finite_mask
    
    if not eval_mask.any():
        print(f"Varovanie: Žiadne umelo maskované hodnoty pre {method}")
        return {}
    
    # Metriky na škálovaných dátach
    rmse_scaled = root_mean_squared_error(targs_scaled[eval_mask], preds_scaled[eval_mask])
    mae_scaled = mean_absolute_error(targs_scaled[eval_mask], preds_scaled[eval_mask])
    r2_scaled = r2_score(targs_scaled[eval_mask], preds_scaled[eval_mask])
    
    # Metriky na originálnych dátach
    rmse_real = root_mean_squared_error(targs_real[eval_mask], preds_real[eval_mask])
    mae_real = mean_absolute_error(targs_real[eval_mask], preds_real[eval_mask])
    r2_real = r2_score(targs_real[eval_mask], preds_real[eval_mask])

    # Priemerne R2 cez features ako v GRU reporte (bez časových feature-ov)
    per_feature_r2 = []
    for col, feature_name in enumerate(feature_names):
        if feature_name in EXCLUDED_FEATURES:
            continue
        col_mask = eval_mask[:, col]
        if not np.any(col_mask):
            continue
        per_feature_r2.append(r2_score(targs_real[col_mask, col], preds_real[col_mask, col]))

    avg_r2_per_feature = float(np.mean(per_feature_r2)) if per_feature_r2 else float("nan")
    
    metrics = {
        'RMSE_scaled': rmse_scaled,
        'MAE_scaled': mae_scaled,
        'R2_scaled': r2_scaled,
        'RMSE_real': rmse_real,
        'MAE_real': mae_real,
        'R2_real': r2_real,
        'Avg_R2_per_feature_real': avg_r2_per_feature,
    }
    
    return metrics


# ============================================================================
# HLAVNÝ SKRIPT
# ============================================================================

def main():
    print("="*70)
    print("INTERPOLAČNÝ TEST - KOMPLETNÝ BASELINE")
    print("="*70)
    
    # Načítaj scaler z trénovania
    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"Scaler nenájdený na {scaler_path}")
        print("   Spustite train.py najskôr!")
        return
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Načítaj testovacie dáta
    print("\n1️Načítavanie testovacích dát...")
    X, M, y, target_masks, _, mask, df, _ = prepare_data(
        "data/pivot_test.parquet", SEQ_LEN, scaler=scaler, 
        fit_scaler=False, verbose=False
    )
    
    print(f"   ✓ X shape: {X.shape}")
    print(f"   ✓ y shape: {y.shape}")
    
    print("Maskovanie je nastavene identicky ako v GRU teste (last timestep, ART_RATE, batch=256).")
    shared_flags_art = build_gru_style_art_masks(target_masks, ART_RATE, batch_size=256)
    print(f"Spoločná maska pre všetky metódy: {int(shared_flags_art.sum())}")
    
    # Vymaž logovací súbor (nový zápis)
    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)
    
    # Vykonaj interpolácie
    print("Testovanie metód interpolácie...\n")
    
    all_results = {}
    
    for method in METHODS:
        print(f"   → {method.upper()}...")
        
        # Predikcie
        preds_scaled, targs_scaled, flags_art = get_interpolation_predictions(
            X, M, y, target_masks, shared_flags_art, method, batch_size=256
        )
        print(f"      Počet hodnotených hodnôt: {int(flags_art.sum())}")
        
        # Evaluácia
        metrics = evaluate_method(preds_scaled, targs_scaled, flags_art, scaler, df.columns, method)
        all_results[method] = metrics
        
        # Zapis do logu
        append_result_log(RESULTS_FILE, method, metrics)
        
        print(f"      RMSE: {metrics['RMSE_scaled']:.6f} (škálované)")
        print(f"      MAE:  {metrics['MAE_scaled']:.6f} (škálované)")
        print(f"      R²:   {metrics['R2_scaled']:.6f} (škálované)")
        print(f"      Avg R² per feature (real): {metrics['Avg_R2_per_feature_real']:.6f}")
        print()
    
    # Porovnanie
    print("POROVNANIE VÝSLEDKOV:\n")
    print(f"{'Metóda':<20} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
    print("-" * 56)
    
    for method in METHODS:
        metrics = all_results[method]
        print(f"{method:<20} {metrics['RMSE_scaled']:<12.6f} "
              f"{metrics['MAE_scaled']:<12.6f} {metrics['R2_scaled']:<12.6f}")
    
    # Najlepšia metóda
    best_method = min(all_results.keys(), 
                     key=lambda m: all_results[m]['RMSE_scaled'])
    print(f"Najlepšia metóda: {best_method.upper()}")
    print(f"RMSE: {all_results[best_method]['RMSE_scaled']:.6f}")
    
    print(f"Výsledky uložené do: {RESULTS_FILE}")


if __name__ == "__main__":
    main()