import torch
import torch.nn as nn
class GRU_Imputation(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input to GRU = concatenation of values and missingness mask
        self.gru = nn.GRU(
            input_size=input_size * 2,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Temporal attention: produces a score for each timestep
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # Output head: imputation of the last timestep
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, mask):

        # 1) Concatenate values and mask along the feature dimension
        x_in = torch.cat([x, mask], dim=-1)  # (B, T, 2F)

        # 2) Encode the sequence with GRU
        gru_out, _ = self.gru(x_in)  # (B, T, H)

        # 3) Compute attention scores for each timestep
        attn_scores = self.attn(gru_out).squeeze(-1)  # (B, T)

        # 4) Mask timesteps that are fully missing (all features missing)
        fully_missing = (mask.sum(dim=-1) == mask.shape[-1])  # (B, T)
        attn_scores = attn_scores.masked_fill(fully_missing, -1e9)

        # 5) Normalize attention scores into weights
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)

        # 6) Compute context vector as weighted sum of GRU outputs
        context = torch.sum(gru_out * attn_weights, dim=1)  # (B, H)

        # 7) Normalize and regularize the context representation
        context = self.layer_norm(context)
        context = self.dropout(context)

        # 8) Predict the imputed values for the last timestep
        out = self.fc(context)  # (B, F)

        return out
