import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from tqdm import tqdm

from utils import prepare_data
import model_param

# Configuration
SEQ_LEN = model_param.SEQ_LEN
ART_RATE = 0.15
METHODS = ["linear", "forward_fill"]
RESULTS_DIR = "models/imputation"
RESULTS_FILE = os.path.join(RESULTS_DIR, "interpolation_results_in_sequence.txt")


def append_result_log(log_path: str, method: str, metrics: dict) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] Method: {method}\n")
        for key, val in metrics.items():
            f.write(f"  {key}: {val:.6f}\n")
        f.write("\n")


def interpolate_sequence(seq: np.ndarray, method: str) -> np.ndarray:
    df = pd.DataFrame(seq)

    if method == "linear":
        out = df.interpolate(method="linear", axis=0)
    elif method == "forward_fill":
        out = df.ffill(axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Fill boundaries and all-NaN columns in the window.
    out = out.bfill(axis=0).ffill(axis=0).fillna(0.0)
    return out.values


def build_in_sequence_mask(M: np.ndarray, art_rate: float, seed: int = 42) -> np.ndarray:
    """Build one shared mask over the whole sequence [N, T, F], only on known values."""
    rng = np.random.default_rng(seed)
    present = (M == 0)
    flags = (rng.random(M.shape) < art_rate) & present

    # Ensure at least one masked value per sample when possible.
    for i in range(flags.shape[0]):
        if flags[i].any():
            continue
        present_idx = np.argwhere(present[i])
        if present_idx.size == 0:
            continue
        pick = int(rng.integers(0, present_idx.shape[0]))
        t, f = present_idx[pick].tolist()
        flags[i, t, f] = True

    return flags.astype(np.uint8)


def get_interpolation_predictions(
    X: np.ndarray,
    M: np.ndarray,
    flags_art: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return flattened preds/targets/feature_index for masked positions across sequence."""
    preds_parts = []
    targs_parts = []
    feat_idx_parts = []

    for i in tqdm(range(X.shape[0]), desc=f"Interpolation [{method}]", leave=False):
        seq = X[i].copy()

        # Restore natural missing values (X has zeros there after scaling pipeline).
        seq[M[i] == 1] = np.nan

        art_mask = flags_art[i].astype(bool)
        if not art_mask.any():
            continue

        true_vals = X[i][art_mask]
        feat_idx = np.where(art_mask)[1]

        # Hide artificial positions before interpolation.
        seq[art_mask] = np.nan

        seq_interp = interpolate_sequence(seq, method)
        pred_vals = seq_interp[art_mask]

        preds_parts.append(pred_vals)
        targs_parts.append(true_vals)
        feat_idx_parts.append(feat_idx)

    if not preds_parts:
        return np.array([]), np.array([]), np.array([], dtype=int)

    preds = np.concatenate(preds_parts, axis=0)
    targs = np.concatenate(targs_parts, axis=0)
    feat_idx = np.concatenate(feat_idx_parts, axis=0)
    return preds, targs, feat_idx


def evaluate_method(
    preds_scaled: np.ndarray,
    targs_scaled: np.ndarray,
    feat_idx: np.ndarray,
    scaler,
    method: str,
) -> dict:
    if preds_scaled.size == 0 or targs_scaled.size == 0 or feat_idx.size == 0:
        print(f"Warning: no evaluation points for {method}")
        return {}

    if not (hasattr(scaler, "mean_") and hasattr(scaler, "scale_")):
        raise ValueError("Scaler missing mean_ or scale_")

    preds_real = preds_scaled * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]
    targs_real = targs_scaled * scaler.scale_[feat_idx] + scaler.mean_[feat_idx]

    finite_mask = (
        np.isfinite(preds_scaled)
        & np.isfinite(targs_scaled)
        & np.isfinite(preds_real)
        & np.isfinite(targs_real)
    )
    if not finite_mask.any():
        print(f"Warning: no finite values for {method}")
        return {}

    rmse_scaled = root_mean_squared_error(targs_scaled[finite_mask], preds_scaled[finite_mask])
    mae_scaled = mean_absolute_error(targs_scaled[finite_mask], preds_scaled[finite_mask])
    r2_scaled = r2_score(targs_scaled[finite_mask], preds_scaled[finite_mask])

    rmse_real = root_mean_squared_error(targs_real[finite_mask], preds_real[finite_mask])
    mae_real = mean_absolute_error(targs_real[finite_mask], preds_real[finite_mask])
    r2_real = r2_score(targs_real[finite_mask], preds_real[finite_mask])

    return {
        "RMSE_scaled": rmse_scaled,
        "MAE_scaled": mae_scaled,
        "R2_scaled": r2_scaled,
        "RMSE_real": rmse_real,
        "MAE_real": mae_real,
        "R2_real": r2_real,
    }


def main() -> None:
    print("=" * 70)
    print("INTERPOLATION TEST - IN-SEQUENCE MASKING")
    print("=" * 70)

    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    if not os.path.exists(scaler_path):
        print(f"Scaler not found: {scaler_path}")
        print("Run train.py first.")
        return

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("\nLoading test data...")
    X, M, y, target_masks, _, mask, df, _ = prepare_data(
        "data/pivot_test.parquet", SEQ_LEN, scaler=scaler, fit_scaler=False, verbose=False
    )

    print(f"X shape: {X.shape}")
    print(f"M shape: {M.shape}")

    flags_art = build_in_sequence_mask(M, ART_RATE, seed=42)
    print(f"Masked points across sequence: {int(flags_art.sum())}")

    if os.path.exists(RESULTS_FILE):
        os.remove(RESULTS_FILE)

    print("\nTesting interpolation methods...\n")
    all_results = {}

    for method in METHODS:
        print(f" -> {method.upper()}...")
        preds_scaled, targs_scaled, feat_idx = get_interpolation_predictions(X, M, flags_art, method)
        print(f"    Evaluated points: {len(preds_scaled)}")

        metrics = evaluate_method(preds_scaled, targs_scaled, feat_idx, scaler, method)
        all_results[method] = metrics
        append_result_log(RESULTS_FILE, method, metrics)

        print(f"    RMSE: {metrics['RMSE_scaled']:.6f} (scaled)")
        print(f"    MAE:  {metrics['MAE_scaled']:.6f} (scaled)")
        print(f"    R2:   {metrics['R2_scaled']:.6f} (scaled)")
        print()

    print("RESULTS:\n")
    print(f"{'Method':<20} {'RMSE':<12} {'MAE':<12} {'R2':<12}")
    print("-" * 56)
    for method in METHODS:
        metrics = all_results[method]
        print(
            f"{method:<20} {metrics['RMSE_scaled']:<12.6f} "
            f"{metrics['MAE_scaled']:<12.6f} {metrics['R2_scaled']:<12.6f}"
        )

    best_method = min(all_results.keys(), key=lambda m: all_results[m]["RMSE_scaled"])
    print(f"\nBest method: {best_method.upper()}")
    print(f"RMSE: {all_results[best_method]['RMSE_scaled']:.6f}")
    print(f"Saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
