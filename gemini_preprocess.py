import os
import numpy as np
import math
import random
from tqdm import trange
from scipy import stats
import struct
from datetime import datetime
import torch
from torch.utils.data import Dataset

class GeminiDataset(Dataset):
    def __init__(self, file_path, input_size=168, predict_size=168, train=True):
        self.input_size = input_size
        self.predict_size = predict_size
        self.seq_len = input_size + predict_size
        
        # Read binary data
        with open(file_path, "rb") as f:
            self.raw_bytes = f.read()

        # Binary format: >Qdddd (Timestamp, Bid, BidQty, Ask, AskQty)
        # Big Endian
        dt = np.dtype([
            ('timestamp', '>u8'),
            ('bid', '>f8'),
            ('bid_qty', '>f8'),
            ('ask', '>f8'),
            ('ask_qty', '>f8')
        ])
        
        self.data = np.frombuffer(self.raw_bytes, dtype=dt)
        self.num_records = len(self.data)
        
        if self.num_records < self.seq_len:
            raise ValueError(f"File {file_path} is too small ({self.num_records} records) for sequence length {self.seq_len}")

        # Precompute features and covariates to save time during training
        self._precompute_data()

    def _precompute_data(self):
        # Extract columns
        timestamps = self.data['timestamp'].astype(np.float64) # Keep as float for calculations if needed, but we need int for datetime
        bid = self.data['bid'].astype(np.float32)
        bid_qty = self.data['bid_qty'].astype(np.float32)
        ask = self.data['ask'].astype(np.float32)
        ask_qty = self.data['ask_qty'].astype(np.float32)

        # 1. Feature Engineering (7 features)
        # 0: Bid Price
        # 1: Bid Qty
        # 2: Ask Price
        # 3: Ask Qty
        # 4: Mid Price
        mid_price = (bid + ask) / 2.0
        # 5: Spread
        spread = ask - bid
        # 6: Imbalance
        # Avoid division by zero
        total_qty = bid_qty + ask_qty
        total_qty[total_qty == 0] = 1.0 
        imbalance = (bid_qty - ask_qty) / total_qty

        # Normalize Features (Z-score)
        # Ideally we should compute mean/std on training set only and apply to val/test
        # For this implementation, we'll normalize based on the loaded file (assuming it's a day's data)
        # Note: Prices are non-stationary, so z-score might not be best, but acceptable for short-term windows.
        # Ideally we'd use log-returns for prices. Let's start with simple Z-score for now as it handles scales.
        
        self.features = np.stack([bid, bid_qty, ask, ask_qty, mid_price, spread, imbalance], axis=1)
        
        # 2. Covariate Generation (4 covariates)
        # Timestamps are in milliseconds
        # Convert to seconds for datetime
        timestamps_sec = self.data['timestamp'] / 1000.0
        
        # Vectorized datetime operations
        # Second of minute (0-59)
        cov_sec = (timestamps_sec % 60).astype(np.float32)
        
        # Minute of hour (0-59)
        cov_min = ((timestamps_sec / 60) % 60).astype(np.float32)
        
        # Hour of day (0-23)
        cov_hour = ((timestamps_sec / 3600) % 24).astype(np.float32)
        
        # Day of week (0-6)
        cov_day = (((timestamps_sec // 86400) + 4) % 7).astype(np.float32)

        self.covariates = np.stack([
            cov_sec / 60.0 - 0.5,
            cov_min / 60.0 - 0.5,
            cov_hour / 24.0 - 0.5,
            cov_day / 7.0 - 0.5
        ], axis=1)

    def normalize(self, mean=None, std=None):
        """
        Normalize features.
        If mean/std are provided, use them.
        If not, compute them from current data (be careful of leakage!).
        Returns mean, std used.
        """
        if mean is None:
            mean = np.mean(self.features, axis=0)
        if std is None:
            std = np.std(self.features, axis=0)
            
        std[std == 0] = 1.0 # Prevent div by zero
        self.features = (self.features - mean) / std
        return mean, std

    def __len__(self):
        return self.num_records - self.seq_len + 1

    def __getitem__(self, idx):
        # Window: [idx, idx + seq_len]
        # features: (seq_len, 7)
        # covariates: (seq_len, 4)
        
        x = self.features[idx : idx + self.seq_len]
        t = self.covariates[idx : idx + self.seq_len]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(t, dtype=torch.float32)

if __name__ == '__main__':
    # Test the dataset
    import sys
    
    # Create a dummy file if not exists for testing
    test_file = "test_data.bin"
    if not os.path.exists(test_file):
        print("Creating dummy test file...")
        with open(test_file, "wb") as f:
            for i in range(5000):
                ts = int(datetime.now().timestamp() * 1000) + i * 1000
                bid = 50000.0 + random.random() * 100
                bid_qty = 1.0 + random.random()
                ask = bid + 10.0 # Spread
                ask_qty = 1.0 + random.random()
                f.write(struct.pack(">Qdddd", ts, bid, bid_qty, ask, ask_qty))
    
    dataset = GeminiDataset(test_file)
    print(f"Dataset length: {len(dataset)}")
    x, t = dataset[0]
    print(f"Sample x shape: {x.shape}") # Should be (336, 7)
    print(f"Sample t shape: {t.shape}") # Should be (336, 4)
    print("Sample x[0]:", x[0])
    print("Sample t[0]:", t[0])



        