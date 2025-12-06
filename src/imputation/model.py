import torch
import torch.nn as nn

class GRU_Imputation(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, bidirectional=False):
        super(GRU_Imputation, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # GRU
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0,
                          bidirectional=bidirectional)
        
        # Attention - prispôsob veľkosť
        gru_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.attn = nn.Sequential(
            nn.Linear(gru_output_size, gru_output_size),
            nn.Tanh(),
            nn.Linear(gru_output_size, 1, bias=False)
        )
        self.layer_norm = nn.LayerNorm(gru_output_size)
        self.fc = nn.Linear(gru_output_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * (1 - mask)

        gru_out, _ = self.gru(x)  # (batch, seq_len, gru_output_size)
        attn_scores = self.attn(gru_out).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        context = (gru_out * attn_weights).sum(dim=1)

        context = self.layer_norm(context)
        output = self.dropout(context)
        output = self.fc(output)
        return output