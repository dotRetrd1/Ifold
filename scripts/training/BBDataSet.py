import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.nn.functional as F
from pathlib import Path

class BBDataset(Dataset):
    def __init__(self, data_dir, maxlen=350):
        self.data_dir = Path(data_dir)
        self.maxlen = maxlen
        self.files = list(self.data_dir.glob("*_features.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feat_path = self.files[idx]
        pdb_id = feat_path.stem.split('_')[0]
        dist_path = self.data_dir / f"{pdb_id}_dist.npy"

        features = torch.tensor(np.load(feat_path), dtype=torch.float32) #(N,4)
        distances = torch.tensor(np.load(dist_path), dtype=torch.float32) #(N,N)

        N = features.shape[0]

        #calc how much padding is needed
        pad_len = self.maxlen - N

        #bottom pad
        feat_padded = F.pad(features, (0, 0, 0, pad_len), "constant", 0)
    
        #bottom and right pad
        dist_padded = F.pad(distances, (0, pad_len, 0, pad_len), "constant", 0)

        #1d mask first
        mask_1d = torch.arange(self.maxlen) < N
        #N x N mask
        mask_2d = mask_1d.unsqueeze(0) & mask_1d.unsqueeze(1)

        return feat_padded, dist_padded, mask_2d
    