import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

df = pd.read_parquet('data/pivot_data_hourly.parquet')

print(df.index)

df = df.dropna()
X = df.shift(1).iloc[1:]  # previous readings (inputs)
Y = df.iloc[1:]           # current readings (targets)

print(X.head())
print(Y.head())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
Y_train_scaled = scaler.fit_transform(Y_train)
X_test_scaled = scaler.transform(X_test)
Y_train_scaled = scaler.transform(Y_test)
Y_test_scaled = scaler.transform(Y_test)
print("Scaled training inputs shape:", X_train_scaled.shape)
print("Scaled training targets shape:", Y_train_scaled.shape)

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_dataset = TimeSeriesDataset(X_train_scaled, Y_train_scaled)
test_dataset = TimeSeriesDataset(X_test_scaled, Y_test_scaled)

