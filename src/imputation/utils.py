import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def create_imputation_sequences(data, mask, seq_len):
    xs, ms, ys, target_masks = [], [], [], []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]  # Input sequence
        m = mask[i:i+seq_len]  # Mask sequence
        
        # Target is the SAME timestep as the last input, not next timestep
        y = data[i+seq_len-1]  # Last timestep of sequence
        target_mask = mask[i+seq_len-1]  # Mask for that timestep
        
        xs.append(x)
        ms.append(m)
        ys.append(y)
        target_masks.append(target_mask)
    
    return np.array(xs), np.array(ms), np.array(ys), np.array(target_masks)


def remove_outliers_iqr(df, factor=1.5):
    df_out = df.copy()
    for col in df_out.columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR

        df_out[col] = np.where(df_out[col] < lower, lower, df_out[col])
        df_out[col] = np.where(df_out[col] > upper, upper, df_out[col])
    return df_out


def prepare_data(filepath, seq_len):
    df = pd.read_parquet(filepath)
    print("df before cleaning:")
    print(df.isna().sum())
    print(f"Before cleaning - NaN: {df.isna().sum().sum()}")
    print(f"Before cleaning - Inf: {np.isinf(df.values).sum()}")
    mask = df.isna().astype(int)
    df_filled = df.copy()
    df_filled = df_filled.interpolate(method='linear', limit=3, limit_direction='both')
    df_filled = df_filled.ffill(limit=5)
    df_filled = df_filled.bfill(limit=5)
    df_filled = df_filled.fillna(df_filled.mean())
    df_filled = df_filled.fillna(0)
    df_filled = df_filled.replace([np.inf, -np.inf], 0)
    df_filled = remove_outliers_iqr(df_filled, factor=1.5)
    print(f"After cleaning - NaN: {df_filled.isna().sum().sum()}")
    print(f"After cleaning - Inf: {np.isinf(df_filled.values).sum()}")
    
    print("Data cleaning successful!")


    print("Columns in df_filled:")
    print(df_filled.columns.tolist())
    print("DataFrame info:")
    print(df_filled.info())
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_filled)
    df_scaled = pd.DataFrame(scaled_values, columns=df_filled.columns, index=df_filled.index)
    data = df_scaled.values
    mask_data = mask.values
    X, M, y, target_masks = create_imputation_sequences(data, mask_data, seq_len)
    return X, M, y, target_masks, scaler, mask, df