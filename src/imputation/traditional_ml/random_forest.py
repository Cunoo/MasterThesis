from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import joblib, re, json, os, sys

# Allow importing project utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import remove_outliers_iqr

# Create models directory
model_dir = "models/imputation/random_Forest"
os.makedirs(model_dir, exist_ok=True)

# Load and clean data using IQR outlier removal
print("Loading data...")
df_train = pd.read_parquet("data/pivot_train.parquet")
df_test = pd.read_parquet("data/pivot_test.parquet")

print("Removing outliers...")
df_train = remove_outliers_iqr(df_train)
df_test = remove_outliers_iqr(df_test)

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")

# Random Forest hyperparameters
rf_params = {
    "n_estimators": 50,           # Number of trees in the forest
    "max_depth": 7,               # Maximum tree depth
    "min_samples_split": 5,       # Minimum samples required to split
    "min_samples_leaf": 2,        # Minimum samples required at leaf node
    "criterion": "squared_error", # Loss function for regression
    "n_jobs": -1,                 # Use all available processors
    "random_state": 42            # Reproducibility seed (RF internals)
}

def impute_column(col, df_train, df_test, rf_params, model_dir):
    # Check if column exists in both
    if col not in df_train.columns or col not in df_test.columns:
        return None

    # Identify known values for training
    train_not_missing = ~df_train[col].isna()
    train_rows = df_train.index[train_not_missing]
    
    if len(train_rows) == 0:
        return None

    # Identify known values for evaluation
    test_not_missing = ~df_test[col].isna()
    eval_rows = df_test.index[test_not_missing]

    # Select only numeric features (exclude target column)
    # We need features that are present in both and numeric
    numeric_features = df_train.select_dtypes(include=np.number).columns.drop(col, errors='ignore')
    # Ensure these features exist in test as well
    numeric_features = [c for c in numeric_features if c in df_test.columns]
    
    if len(numeric_features) == 0:
        return None

    # TRAINING: Build training set from training rows
    X_train = df_train.loc[train_rows, numeric_features]
    y_train = df_train.loc[train_rows, col]

    # Remove features that are entirely NaN in training set
    valid_features = [c for c in numeric_features if not X_train[c].isna().all()]
    if len(valid_features) == 0:
        return None
    X_train = X_train[valid_features]

    # Create and train pipeline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**rf_params)),
    ])
    pipe.fit(X_train, y_train)

    # EVALUATION: Test on evaluation set (df_test known values)
    metrics = {}
    if len(eval_rows) > 0:
        X_eval = df_test.loc[eval_rows, valid_features]
        y_pred = pipe.predict(X_eval)
        y_true = df_test.loc[eval_rows, col]

        rmse_eval = root_mean_squared_error(y_true, y_pred)
        mae_eval = mean_absolute_error(y_true, y_pred)
        r2_eval = r2_score(y_true, y_pred)

        metrics = {"rmse": float(rmse_eval), "mae": float(mae_eval), "r2": float(r2_eval)}
    else:
        metrics = {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}

    # IMPUTATION: Predict missing values in both sets
    # Train set missing
    train_missing_mask = df_train[col].isna()
    if train_missing_mask.any():
        X_missing_train = df_train.loc[train_missing_mask, valid_features]
        imputed_values_train = pipe.predict(X_missing_train)
        df_train.loc[train_missing_mask, col] = imputed_values_train
    
    # Test set missing
    test_missing_mask = df_test[col].isna()
    if test_missing_mask.any():
        X_missing_test = df_test.loc[test_missing_mask, valid_features]
        imputed_values_test = pipe.predict(X_missing_test)
        df_test.loc[test_missing_mask, col] = imputed_values_test

    # Save trained model
    safe_col = re.sub(r'[^A-Za-z0-9_.-]+', '_', col)
    model_path = os.path.join(model_dir, f"rf_{safe_col}.joblib")
    joblib.dump(pipe, model_path)

    return {
        "column": col,
        "train_n": len(train_rows),
        "eval_n": len(eval_rows),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "model_path": model_path
    }

# Process all columns
results = []
# Iterate over columns present in both
common_columns = [c for c in df_train.columns if c in df_test.columns]

pbar = tqdm(total=len(common_columns), desc="Imputing columns")
for col in common_columns:
    r = impute_column(col, df_train, df_test, rf_params, model_dir)
    if r is not None:
        results.append(r)
        pbar.set_postfix({
            "col": col[:30],
            "RMSE": f"{r['rmse']:.4f}" if not pd.isna(r["rmse"]) else "N/A",
            "MAE": f"{r['mae']:.4f}" if not pd.isna(r["mae"]) else "N/A",
            "R²": f"{r['r2']:.4f}" if not pd.isna(r["r2"]) else "N/A"
        })
        tqdm.write(f"{col:40s} | RMSE={r['rmse']:8.4f} | MAE={r['mae']:8.4f} | R²={r['r2']:7.4f}")
    pbar.update(1)
pbar.close()

# Save evaluation metrics to CSV
metrics_path = os.path.join(model_dir, "masked_metrics.csv")
metrics_df = pd.DataFrame(results)
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved to {metrics_path}")


# Print summary
print(f"\n{'='*60}")
print("Imputation Summary")
print(f"{'='*60}")
print(f"Total columns processed: {len(results)}")
print("\nDetailed Metrics:")
print(metrics_df[["column", "train_n", "eval_n", "rmse", "mae", "r2"]].to_string())
print(f"{'='*60}")
