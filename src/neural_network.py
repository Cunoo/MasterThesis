import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

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
Y_test_scaled = scaler.transform(Y_test)
print("Scaled training inputs shape:", X_train_scaled.shape)
print("Scaled training targets shape:", Y_train_scaled.shape)


import numpy as np

SEQ_LEN = 24  # e.g., use previous 24 hours

def create_sequences(X, Y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(Y[i+seq_len])
    return np.array(xs), np.array(ys)

X_seq, Y_seq = create_sequences(X_train_scaled, Y_train_scaled, SEQ_LEN)
X_test_seq, Y_test_seq = create_sequences(X_test_scaled, Y_test_scaled, SEQ_LEN)

np.save("data/X_test_seq.npy", X_test_seq)
np.save("data/Y_test_seq.npy", Y_test_seq)

# Print shapes
print("X_seq shape:", X_seq.shape)
print("Y_seq shape:", Y_seq.shape)

# Print first sequence and target
print("First input sequence:\n", X_seq[0])
print("First target:\n", Y_seq[0])

class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_dataset = TimeSeriesDataset(X_seq, Y_seq)
test_dataset = TimeSeriesDataset(X_test_seq, Y_test_seq)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=X_seq.shape[2], output_size=Y_seq.shape[1], hidden_size=512, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
    
model = LSTMModel(input_size=X_seq.shape[2], output_size=Y_seq.shape[1])
print("Neural network:", model)
model = model.to(device)

print(f"Using device: {device}")
print(model)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 15

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss/len(train_loader)}")
print("Training complete.")

if(torch.save(model.state_dict(), "models/neural_networks/lstm_model.pth")):
    print("Model saved as lstm_model.pth")
else:
    print("Error saving the model.")

model.eval()
test_losses = []
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_losses.append(loss.item())
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

print(f"Average test loss: {np.mean(test_losses)}")


# Concatenate list of arrays to a single array
actuals = np.concatenate(actuals, axis=0)
predictions = np.concatenate(predictions, axis=0)


# Calculate MSE for each sensor column
mse_per_sensor = np.mean((actuals - predictions) ** 2, axis=0)

# Find the column (sensor) with the highest loss
max_loss_idx = np.argmax(mse_per_sensor)
max_loss_value = mse_per_sensor[max_loss_idx]

print(f"Sensor column with highest loss: {Y_test.columns[max_loss_idx]} (Loss: {max_loss_value:.4f})")

from sklearn.metrics import r2_score
# If you want to print all losses per column:
for idx, loss in enumerate(mse_per_sensor):
    rmse = np.sqrt(loss)
    r2 = r2_score(actuals[:, idx], predictions[:, idx])
    print(f"{Y.columns[idx]}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")


