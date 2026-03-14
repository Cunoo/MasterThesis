from datetime import datetime
import os

import pandas as pd


TEST_LOG_PATH = os.path.join("models", "imputation", "test_log.txt")


def append_test_log(
    log_path: str,
    model_name: str,
    model_file: str,
    metrics_df: pd.DataFrame,
) -> None:
    """
    Appends test results to a log file.
    metrics_df must contain columns: Feature, RMSE, MAE, R2
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Separator if file already has content
    if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"=== TEST RUN [{ts}] | model={model_name} | file={model_file} ===\n")
        f.write(f"{'Feature':<35} | {'RMSE':<10} | {'MAE':<10} | {'R2':<10}\n")
        f.write("-" * 72 + "\n")

        for _, row in metrics_df.iterrows():
            f.write(
                f"{str(row['Feature'])[:35]:<35} | "
                f"{row['RMSE']:10.4f} | "
                f"{row['MAE']:10.4f} | "
                f"{row['R2']:10.4f}\n"
            )

        # Average row
        avg_rmse = metrics_df['RMSE'].mean()
        avg_mae  = metrics_df['MAE'].mean()
        avg_r2   = metrics_df['R2'].mean()
        f.write("-" * 72 + "\n")
        f.write(
            f"{'AVERAGE':<35} | "
            f"{avg_rmse:10.4f} | "
            f"{avg_mae:10.4f} | "
            f"{avg_r2:10.4f}\n"
        )