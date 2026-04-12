# visualization_imputation_specific_features.py
from matplotlib import pyplot as plt
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
from model import GRU_Imputation, GRUOnlyImputation, AttentionOnlyImputation
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
MODEL_SPECS = [
    {
        "name": "GRU + Attention",
        "path": os.path.join(root_path, "models", "imputation", "imputation_model__gru_and_attention.pth"),
        "builder": lambda input_size: GRU_Imputation(
            input_size=input_size,
            hidden_size=model_param.hidden_size,
            dropout=model_param.dropout,
        ),
    },
    {
        "name": "Only GRU",
        "path": os.path.join(root_path, "models", "imputation", "imputation_model_only_gru.pth"),
        "builder": lambda input_size: GRUOnlyImputation(
            input_size=input_size,
            hidden_size=model_param.hidden_size,
            num_layers=model_param.num_layers,
            dropout=model_param.dropout,
        ),
    },
    {
        "name": "Only Attention",
        "path": os.path.join(root_path, "models", "imputation", "imputation_model_only_attention_.pth"),
        "builder": lambda input_size: AttentionOnlyImputation(
            input_size=input_size,
            hidden_size=model_param.hidden_size,
            dropout=model_param.dropout,
        ),
    },
]

print("Loading data and preparing sequences...")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

X, M, y, target_masks, _, mask, df, seq_to_orig_idx = prepare_data(
    DATA_PATH, SEQ_LEN, scaler=scaler, fit_scaler=False, verbose=False
)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
input_size = X.shape[2]

test_dataset = ImputationDataset(X, M, y, target_masks)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


def run_inference_for_model(model):
    # Re-seed before each run so every model sees the same artificial masking pattern.
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    all_preds_scaled, all_targs_scaled, all_flags_art = [], [], []
    with torch.no_grad():
        for X_batch, M_batch, y_batch, tm_batch in test_loader:
            X_batch = X_batch.to(device)
            M_batch = M_batch.to(device)
            y_batch = y_batch.to(device)
            tm_batch = tm_batch.to(device)

            present = tm_batch == 0
            rand = torch.rand_like(present.float())
            art_mask = (rand < ART_RATE) & present

            X_masked = X_batch.clone()
            M_masked = M_batch.clone()
            if art_mask.any():
                X_masked[:, -1, :] = torch.where(
                    art_mask,
                    torch.zeros_like(X_masked[:, -1, :]),
                    X_masked[:, -1, :],
                )
                M_masked[:, -1, :] = torch.where(
                    art_mask,
                    torch.ones_like(M_masked[:, -1, :]),
                    M_masked[:, -1, :],
                )

            outputs = model(X_masked, M_masked)
            all_preds_scaled.append(outputs.cpu().numpy())
            all_targs_scaled.append(y_batch.cpu().numpy())
            all_flags_art.append(art_mask.cpu().numpy().astype(np.uint8))

    preds_scaled = np.concatenate(all_preds_scaled, axis=0)
    targs_scaled = np.concatenate(all_targs_scaled, axis=0)
    flags_art = np.concatenate(all_flags_art, axis=0)

    preds_orig = preds_scaled * scaler.scale_ + scaler.mean_
    targs_orig = targs_scaled * scaler.scale_ + scaler.mean_
    return preds_orig, targs_orig, flags_art


results_by_model = {}
for spec in MODEL_SPECS:
    if not os.path.exists(spec["path"]):
        print(f"Skipping {spec['name']}: file not found at {spec['path']}")
        continue

    print(f"Running inference for model: {spec['name']}")
    model = spec["builder"](input_size).to(device)
    try:
        state_dict = torch.load(spec["path"], map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Skipping {spec['name']}: incompatible checkpoint ({e})")
        continue
    model.eval()

    preds_orig, targs_orig, flags_art = run_inference_for_model(model)
    results_by_model[spec["name"]] = {
        "preds": preds_orig,
        "targs": targs_orig,
        "flags": flags_art,
    }

if not results_by_model:
    raise RuntimeError("No model checkpoint could be loaded. Check MODEL_SPECS paths.")

# 6. Nájdenie stĺpcov podľa názvu
cols_to_plot = []
for feature_name in FEATURES_TO_PLOT:
    if feature_name in df.columns:
        col_idx = df.columns.get_loc(feature_name)
        cols_to_plot.append((col_idx, feature_name))
    else:
        print(f"Warning: Feature '{feature_name}' not found in dataframe!")

# 7. Vizualizácia: jeden obrazok pre kazdy sensor/feature
plot_limit = 500
for col, feature_name in cols_to_plot:
    first_model_name = next(iter(results_by_model))
    common_flags = results_by_model[first_model_name]["flags"][:, col] == 1
    if common_flags.sum() == 0:
        print(f"Skipping feature '{feature_name}': no artificially masked points.")
        continue

    plt.figure(figsize=(14, 6))
    y_t = results_by_model[first_model_name]["targs"][common_flags, col][:plot_limit]
    x_axis = np.arange(len(y_t))

    plt.scatter(x_axis, y_t, color="black", label="Actual", alpha=0.65, s=18)

    for model_name, model_data in results_by_model.items():
        y_p = model_data["preds"][common_flags, col][:plot_limit]
        mae = np.mean(np.abs(y_t - y_p)) if len(y_t) > 0 else np.nan
        plt.scatter(
            x_axis,
            y_p,
            label=f"{model_name} (MAE={mae:.3f})",
            alpha=0.6,
            s=16,
        )

    plt.title(feature_name)
    plt.xlabel("Sample Index (Masked points)")
    plt.ylabel("Real Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    safe_name = "".join(ch if ch.isalnum() else "_" for ch in feature_name).strip("_")
    output_name = f"imputation_compare_{safe_name}.png"
    plt.savefig(output_name, dpi=150)
    plt.close()
    print(f"Saved: {output_name}")
    
    
    
