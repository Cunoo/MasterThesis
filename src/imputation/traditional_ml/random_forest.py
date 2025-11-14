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
model_dir = "models/imputation/random_forest"
os.makedirs(model_dir, exist_ok=True)

# Load and clean data using IQR outlier removal
df = pd.read_parquet("data/pivot_data.parquet")
df = remove_outliers_iqr(df)
orig_df = df.copy(deep=True)

print(f"Data shape: {df.shape}")
print(f"Missing values per column:\n{df.isna().sum()}\n")

# Random Forest hyperparameters
rf_params = {
    "n_estimators": 50,           # Number of trees in the forest
    "max_depth": 7,               # Maximum tree depth
    "min_samples_split": 5,       # Minimum samples required to split
    "min_samples_leaf": 2,        # Minimum samples required at leaf node
    "criterion": "squared_error", # Loss function for regression
    "n_jobs": -1,                 # Use all available processors
    "random_state": 42            # Reproducibility seed
}

test_frac = 0.1  # Reserve 10% of known values for testing

def impute_column(col, df_local, orig_df_local, rf_params, test_frac, model_dir):
    """
    Impute missing values in a column using Random Forest with cross-validation.
    
    Strategy:
    1. Split known values into training (90%) and evaluation (10%) sets
    2. Train RF model on training set only
    3. Evaluate on held-out evaluation set
    4. Predict missing values using original features
    
    Args:
        col: Column name to impute
        df_local: DataFrame with original data
        orig_df_local: Clean original DataFrame (for ground truth)
        rf_params: Random Forest hyperparameters dictionary
        test_frac: Fraction of known values to reserve for evaluation (0.1 = 10%)
        model_dir: Directory path to save trained models
    
    Returns:
        Dictionary containing imputation results or None if column has no missing values
    """
    # Identify missing values in current column
    real_missing = df_local[col].isna()
    real_missing_count = int(real_missing.sum())
    
    # Skip columns with no missing values
    if real_missing_count == 0:
        return None

    # Get indices of all known (non-missing) values
    known_rows = df_local.index[~real_missing]
    eval_n = int(len(known_rows) * test_frac)

    # STRICT TRAIN/TEST SPLIT: Separate before any masking
    # This prevents data leakage between train and evaluation sets
    if eval_n > 0:
        # Use column name hash as seed for reproducible splits
        rng = np.random.RandomState(hash(col) % (2**32))
        eval_rows = rng.choice(known_rows, size=eval_n, replace=False)
        train_rows = known_rows.difference(eval_rows)  # Strictly separated!
    else:
        eval_rows = np.array([], dtype=df_local.index.dtype)
        train_rows = known_rows

    # Select only numeric features (exclude target column)
    numeric_features = df_local.select_dtypes(include=np.number).columns.drop(col, errors='ignore')
    if len(numeric_features) == 0:
        return None

    # TRAINING: Build training set from training rows only
    X_train = df_local.loc[train_rows, numeric_features]
    y_train = df_local.loc[train_rows, col]

    # Remove features that are entirely NaN (would cause imputer to fail)
    valid_features = [c for c in numeric_features if not X_train[c].isna().all()]
    if len(valid_features) == 0:
        return None
    X_train = X_train[valid_features]

    # Create and train pipeline: impute missing features → train Random Forest
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # Impute missing feature values with median
        ("rf", RandomForestRegressor(**rf_params)),      # Train Random Forest regressor
    ])
    pipe.fit(X_train, y_train)

    # EVALUATION: Test on held-out evaluation set with original features
    metrics = {}
    if len(eval_rows) > 0:
        # Get evaluation features (original, not masked)
        X_eval = df_local.loc[eval_rows, valid_features]
        y_pred = pipe.predict(X_eval)
        # Get ground truth values from original dataframe
        y_true = orig_df_local.loc[eval_rows, col]

        # Calculate regression metrics
        rmse_eval = root_mean_squared_error(y_true, y_pred)
        mae_eval = mean_absolute_error(y_true, y_pred)
        r2_eval = r2_score(y_true, y_pred)
        
        metrics = {
            "rmse": float(rmse_eval),
            "mae": float(mae_eval),
            "r2": float(r2_eval)
        }
    else:
        # No evaluation set available
        metrics = {
            "rmse": float('nan'),
            "mae": float('nan'),
            "r2": float('nan')
        }

    # IMPUTATION: Predict missing values using entire dataset features
    X_missing = df_local.loc[real_missing, valid_features]
    imputed_values = pipe.predict(X_missing)

    # Save trained model for future use
    safe_col = re.sub(r'[^A-Za-z0-9_.-]+', '_', col)  # Sanitize column name for filename
    model_path = os.path.join(model_dir, f"rf_{safe_col}.joblib")
    joblib.dump(pipe, model_path)

    return {
        "column": col,
        "imputed_values": imputed_values,
        "imputed_index": df_local.index[real_missing],
        "real_missing": real_missing_count,
        "eval_n": int(eval_n),
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "rf_params": json.dumps(rf_params),
        "model_path": model_path
    }

# Process all columns with progress bar
results = []
pbar = tqdm(total=len(df.columns), desc="Imputing columns")
for col in df.columns:
    r = impute_column(col, df, orig_df, rf_params, test_frac, model_dir)
    if r is not None:
        results.append(r)
        # Update progress bar with current metrics
        pbar.set_postfix({
            "col": col[:30],
            "RMSE": f"{r['rmse']:.4f}" if not pd.isna(r['rmse']) else "N/A",
            "MAE": f"{r['mae']:.4f}" if not pd.isna(r['mae']) else "N/A",
            "R²": f"{r['r2']:.4f}" if not pd.isna(r['r2']) else "N/A"
        })
        # Print detailed per-column results
        tqdm.write(f"{col:40s} | RMSE={r['rmse']:8.4f} | MAE={r['mae']:8.4f} | R²={r['r2']:7.4f}")
    pbar.update(1)
pbar.close()

# Fill original dataframe with imputed values
for r in results:
    if len(r["imputed_values"]) > 0:
        df.loc[r["imputed_index"], r["column"]] = r["imputed_values"]

# Save evaluation metrics to CSV
metrics_path = os.path.join(model_dir, "masked_metrics.csv")
metrics_df = pd.DataFrame([{
    "column": r["column"],
    "real_missing": r["real_missing"],
    "eval_n": r["eval_n"],
    "rmse": r["rmse"],
    "mae": r["mae"],
    "r2": r["r2"]
} for r in results])
metrics_df.to_csv(metrics_path, index=False)
print(f"\nMetrics saved to {metrics_path}")

# Save imputed dataframe to parquet format
imputed_path = os.path.join(model_dir, "imputed_data.parquet")
df.to_parquet(imputed_path)
print(f"Imputed data saved to {imputed_path}")

# Print summary statistics
print(f"\n{'='*60}")
print(f"Imputation Summary")
print(f"{'='*60}")
print(f"Total columns processed: {len(results)}")
print(f"Models saved: {len([r for r in results])}")
print(f"\nDetailed Metrics:")
print(metrics_df[["column", "real_missing", "eval_n", "rmse", "mae", "r2"]].to_string())
print(f"{'='*60}")