import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def create_imputation_sequences(data, mask, seq_len):
    xs, ms, ys, target_masks = [], [], [], []
    for i in range(len(data) - seq_len + 1):
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


def prepare_data(filepath, seq_len, scaler=None, fit_scaler: bool = True, impute_mode: bool = False, verbose: bool = True):
    df = pd.read_parquet(filepath)
    df = df.replace([np.inf, -np.inf], np.nan)
    #df = remove_outliers_iqr(df.copy(), factor=1.5)
    # Časové features
    time_index = pd.to_datetime(df.index)
    day_of_year = time_index.dayofyear.to_numpy()
    hour_of_day = time_index.hour.to_numpy()
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.0)
    df["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24.0)
    
    df_model = df.copy()
    df_stats = remove_outliers_iqr(df.copy(), factor=1.5)
    
    #After remove outliers take it and also predict as "imputation"
    clipped_mask = df_model != df_stats
    df_model[clipped_mask] = np.nan
    df = df_model
    
    # Mask = kde sú NaN
    mask = df.isna().astype(int)
    values = df.to_numpy(dtype=float)

    if fit_scaler:
        # Použijeme StandardScaler
        scaler = StandardScaler()
        
        # StandardScaler nezvláda NaN pri fit().
        # Vypočítame mean a std manuálne ignorujúc NaN a nastavíme ich do scalera.
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        
        # Ošetrenie nulovej smerodajnej odchýlky (konštantné stĺpce)
        std[std == 0] = 1.0
        
        scaler.mean_ = mean
        scaler.scale_ = std
        scaler.var_ = std ** 2
        scaler.n_samples_seen_ = np.sum(~np.isnan(values), axis=0)
        
    else:
        if scaler is None:
            raise ValueError("prepare_data(..., fit_scaler=False) requires passing an existing scaler")

    # Transformácia
    # StandardScaler.transform() tiež nezvláda NaN, musíme to obísť
    # (x - mean) / std
    
    # Manuálna transformácia pre rýchlosť a podporu NaN
    scaled_values = (values - scaler.mean_) / scaler.scale_
    
    # Nahradíme NaN nulou (čo je priemer v štandardizovaných dátach)
    # Toto je pre neurónovú sieť bezpečné, lebo 0 = priemer.
    scaled_values = np.nan_to_num(scaled_values, nan=0.0)
    
    df_scaled = pd.DataFrame(scaled_values, columns=df.columns, index=df.index)

    # Kontrola rozsahu
    if verbose and fit_scaler:
        print("Min after scaling:", df_scaled.min().min())
        print("Max after scaling:", df_scaled.max().max())
        print("Mean after scaling (should be ~0):", df_scaled.mean().mean())
        print("Std after scaling (should be ~1):", df_scaled.std().mean())

    data = df_scaled.values
    mask_data = mask.values

    # Vytvorenie sekvencií pre GRU/ANN
    X, M, y, target_masks = create_imputation_sequences(data, mask_data, seq_len)

    sequence_to_original_idx = np.arange(seq_len - 1, len(data))

    return X, M, y, target_masks, scaler, mask, df, sequence_to_original_idx



def random_mask(data, missing_rate=0.1, seed=None):
    np.random.seed(seed)
    mask = np.random.rand(*data.shape) < missing_rate
    data_masked = data.copy()
    data_masked[mask] = 0.0
    return data_masked, mask.astype(int)