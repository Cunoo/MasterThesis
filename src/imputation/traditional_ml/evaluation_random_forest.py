# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib
# import os
# import sys
# import re
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# # Allow importing project utilities
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from utils import remove_outliers_iqr

# # Setup
# model_dir = "models/imputation/random_forest"
# df = pd.read_parquet("data/pivot_data.parquet")
# df = remove_outliers_iqr(df)
# orig_df = df.copy(deep=True)

# # Configuration: Select column indices to visualize
# COLUMN_INDICES = [0, 5, 10]  # Column indices (0-based)
# test_frac = 0.1

# # Get column names from indices
# all_columns = df.columns.tolist()
# selected_columns = [all_columns[i] for i in COLUMN_INDICES if i < len(all_columns)]

# print(f"Selected columns: {selected_columns}\n")

# # Prepare for metrics
# results = []

# # Create figure with subplots
# fig, axes = plt.subplots(len(selected_columns), 1, figsize=(14, 5 * len(selected_columns)))
# if len(selected_columns) == 1:
#     axes = [axes]

# for idx, col in enumerate(selected_columns):
#     ax = axes[idx]
#     safe_col = re.sub(r'[^A-Za-z0-9_.-]+', '_', col)
#     model_path = os.path.join(model_dir, f"rf_{safe_col}.joblib")
#     if not os.path.exists(model_path):
#         print(f"Model not found for column: {col}")
#         continue
#     pipe = joblib.load(model_path)

#     # Find known (non-missing) rows
#     real_missing = df[col].isna()
#     known_rows = df.index[~real_missing]
#     eval_n = int(len(known_rows) * test_frac)
#     if eval_n == 0:
#         continue

#     # Artificially mask 10% of known values
#     rng = np.random.RandomState(hash(col) % (2**32))
#     eval_rows = rng.choice(known_rows, size=eval_n, replace=False)
#     eval_rows = pd.Index(eval_rows)
#     df_masked = df.copy()
#     df_masked.loc[eval_rows, col] = np.nan

#     # Use only numeric features except the target
#     numeric_features = df.select_dtypes(include=np.number).columns.drop(col, errors='ignore').tolist()
#     train_rows = known_rows.difference(eval_rows)
#     valid_features = [c for c in numeric_features if not df.loc[train_rows, c].isna().all()]
#     if len(valid_features) == 0:
#         continue

#     # Predict on artificially masked rows
#     X_eval = df_masked.loc[eval_rows, valid_features]
#     y_pred = pipe.predict(X_eval)
#     y_true = np.array(orig_df.loc[eval_rows, col])

#     # Sort by index for better visualization
#     sort_idx = np.argsort(eval_rows)
#     y_true_sorted = y_true[sort_idx]
#     y_pred_sorted = y_pred[sort_idx]
#     x_pos = np.arange(len(eval_rows))

#     # Plot comparison
#     ax.plot(x_pos, y_true_sorted, 'o-', label='Real Data', linewidth=2, markersize=4, color='blue', alpha=0.7)
#     ax.plot(x_pos, y_pred_sorted, 's--', label='Predicted Data', linewidth=2, markersize=4, color='orange', alpha=0.7)

#     # Calculate metrics
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     results.append((col, rmse, mae, r2))

#     # Labels and title
#     ax.set_title(f"{col}\nRMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}", fontsize=12, fontweight='bold')
#     ax.set_xlabel("Sample Index (sorted)")
#     ax.set_ylabel("Value")
#     ax.legend(loc='upper right')
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(os.path.join(model_dir, "imputation_comparison_artificial_missing.png"), dpi=300, bbox_inches='tight')
# print(f"\nGraph saved to {os.path.join(model_dir, 'imputation_comparison_artificial_missing.png')}")
# plt.show()

# # Save metrics to txt
# with open(os.path.join(model_dir, "rf_metrics_artificial_missing.txt"), "w", encoding="utf-8") as f:
#     f.write("Column\tRMSE\tMAE\tR2\n")
#     for col, rmse, mae, r2 in results:
#         f.write(f"{col}\t{rmse:.4f}\t{mae:.4f}\t{r2:.4f}\n")
# print(f"Saved artificial missing metrics to {os.path.join(model_dir, 'rf_metrics_artificial_missing.txt')}")

import numpy as np
import pandas as pd
import joblib
import os
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys

# Allow importing project utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import remove_outliers_iqr

# Setup
model_dir = "models/imputation/random_forest"
df = pd.read_parquet("data/pivot_data.parquet")
df = remove_outliers_iqr(df)
orig_df = df.copy(deep=True)

test_frac = 0.1

results = []

for col in df.columns:
    safe_col = re.sub(r'[^A-Za-z0-9_.-]+', '_', col)
    model_path = os.path.join(model_dir, f"rf_{safe_col}.joblib")
    if not os.path.exists(model_path):
        print(f"Model not found for column: {col}")
        continue
    pipe = joblib.load(model_path)

    # Find known (non-missing) rows
    real_missing = df[col].isna()
    known_rows = df.index[~real_missing]
    eval_n = int(len(known_rows) * test_frac)
    if eval_n == 0:
        continue

    # Artificially mask 10% of known values
    rng = np.random.RandomState(hash(col) % (2**32))
    eval_rows = rng.choice(known_rows, size=eval_n, replace=False)
    eval_rows = pd.Index(eval_rows)
    df_masked = df.copy()
    df_masked.loc[eval_rows, col] = np.nan

    # Use only numeric features except the target
    numeric_features = df.select_dtypes(include=np.number).columns.drop(col, errors='ignore').tolist()
    train_rows = known_rows.difference(eval_rows)
    valid_features = [c for c in numeric_features if not df.loc[train_rows, c].isna().all()]
    if len(valid_features) == 0:
        continue

    # Predict on artificially masked rows
    X_eval = df_masked.loc[eval_rows, valid_features]
    y_pred = pipe.predict(X_eval)
    y_true = np.array(orig_df.loc[eval_rows, col])

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results.append((col, rmse, mae, r2))
    print(f"{col:40s} | RMSE={rmse:8.4f} | MAE={mae:8.4f} | R²={r2:7.4f}")

# Save metrics to txt
with open(os.path.join(model_dir, "rf_metrics_artificial_missing.txt"), "w", encoding="utf-8") as f:
    f.write("Column\tRMSE\tMAE\tR2\n")
    for col, rmse, mae, r2 in results:
        f.write(f"{col}\t{rmse:.4f}\t{mae:.4f}\t{r2:.4f}\n")
print(f"\nSaved artificial missing metrics to {os.path.join(model_dir, 'rf_metrics_artificial_missing.txt')}")