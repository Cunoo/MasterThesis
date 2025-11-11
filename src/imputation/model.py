import torch
import torch.nn as nn

class GRU_Imputation(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super(GRU_Imputation, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, features)
        # Apply mask if provided (set missing values to 0)
        if mask is not None:
            # Where mask=1 (missing), set to 0
            x = x * (1 - mask)
        
        # GRU forward pass
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size)
        
        # Take the last time step
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        last_output = self.layer_norm(last_output)
        
        # Apply dropout and get final output
        output = self.dropout(last_output)
        output = self.fc(output)  # (batch_size, input_size)
        
        return output