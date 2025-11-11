import torch
from torch.utils.data import Dataset

class ImputationDataset(Dataset):
    def __init__(self, X, M, y, target_masks):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.M = torch.tensor(M, dtype=torch.float32)  # Input mask
        self.y = torch.tensor(y, dtype=torch.float32)  # Target values
        self.target_masks = torch.tensor(target_masks, dtype=torch.float32)  # Target mask
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.M[idx], self.y[idx], self.target_masks[idx]
    