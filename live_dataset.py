import torch
from torch.utils.data import IterableDataset
import struct
import time
import os
import numpy as np
import json
from pathlib import Path
import glob

class OnlineScaler:
    """
    Computes running mean and std dev for features.
    Welford's online algorithm.
    """
    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features)
        self.M2 = np.zeros(n_features)
        
    def update(self, x):
        """Update stats with a new sample x (1D array)"""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        
    def transform(self, x):
        """Normalize x using current stats"""
        if self.n < 2:
            return x # Not enough data
        std = np.sqrt(self.M2 / (self.n - 1))
        # Prevent division by zero
        std[std == 0] = 1.0
        return (x - self.mean) / std
    
    def load_state_dict(self, state_dict):
        self.n = state_dict['n']
        self.mean = np.array(state_dict['mean'])
        self.M2 = np.array(state_dict['M2'])

    def state_dict(self):
        return {'n': self.n, 'mean': self.mean.tolist(), 'M2': self.M2.tolist()}

class LiveGeminiDataset(IterableDataset):
    def __init__(self, data_dir, symbol, input_size=168, predict_size=168):
        self.data_dir = Path(data_dir)
        self.symbol = symbol.lower()
        self.seq_len = input_size + predict_size
        self.scaler = OnlineScaler(7) # 7 features
        
        # Binary format: >Qdddd (Timestamp, Bid, BidQty, Ask, AskQty)
        # Size: 8 + 8 + 8 + 8 + 8 = 40 bytes
        self.record_size = 40 
        self.struct_fmt = ">Qdddd"
        
        self.buffer = []
        
        # Internal state for continuity
        self._last_read_file = None
        self._last_read_offset = 0
        self._waiting_msg_shown = False
        self._current_filename = None

    def _get_active_file(self):
        """
        Identify the file to read. 
        """
        status_path = self.data_dir / self.symbol / "recorder_status.json"
        
        # Try reading status first
        if status_path.exists():
            try:
                for _ in range(3):
                    try:
                        with open(status_path, 'r') as f:
                            status = json.load(f)
                            current_file = Path(status['current_file'])
                            if current_file.exists():
                                return current_file
                    except json.JSONDecodeError:
                        time.sleep(0.1)
            except:
                pass
        
        # Fallback: find latest binary file by name
        files = sorted(glob.glob(str(self.data_dir / self.symbol / "*.bin")))
        if files:
            return Path(files[-1])
        return None

    def _process_record(self, record_bytes):
        """Parse bytes into features (same logic as gemini_preprocess.py)"""
        if len(record_bytes) != self.record_size:
            return None, None
            
        ts, bid, bid_qty, ask, ask_qty = struct.unpack(self.struct_fmt, record_bytes)
        
        # Feature Engineering
        mid_price = (bid + ask) / 2.0
        spread = ask - bid
        total_qty = bid_qty + ask_qty
        if total_qty == 0: total_qty = 1.0
        imbalance = (bid_qty - ask_qty) / total_qty
        
        # Features: Bid, BidQty, Ask, AskQty, Mid, Spread, Imbalance
        features = np.array([bid, bid_qty, ask, ask_qty, mid_price, spread, imbalance], dtype=np.float32)
        
        # Update scaler
        self.scaler.update(features)
        
        # Normalize
        norm_features = self.scaler.transform(features)
        
        # Covariates
        ts_sec = ts / 1000.0
        
        cov_sec = (ts_sec % 60)
        cov_min = ((ts_sec / 60) % 60)
        cov_hour = ((ts_sec / 3600) % 24)
        cov_day = (((ts_sec // 86400) + 4) % 7)
        
        covariates = np.array([
            cov_sec / 60.0 - 0.5,
            cov_min / 60.0 - 0.5,
            cov_hour / 24.0 - 0.5,
            cov_day / 7.0 - 0.5
        ], dtype=np.float32)
        
        return norm_features, covariates

    def __iter__(self):
        while True:
            filename = self._get_active_file()
            while not filename:
                if not self._waiting_msg_shown:
                    print(f"Waiting for recorder data in {self.data_dir}/{self.symbol}...")
                    self._waiting_msg_shown = True
                time.sleep(1)
                filename = self._get_active_file()
                
            if self._current_filename != str(filename):
                print(f"Streaming from {filename}")
                self._current_filename = str(filename)
                self._waiting_msg_shown = False
            
            try:
                with open(filename, 'rb') as f:
                    # Seek to stored offset if same file
                    if self._last_read_file == str(filename):
                        f.seek(self._last_read_offset)
                    
                    while True:
                        chunk = f.read(self.record_size)
                        if not chunk:
                            time.sleep(0.1)
                            # Check rotation
                            new_filename = self._get_active_file()
                            if new_filename and new_filename.name != filename.name:
                                print(f"Switching to new file: {new_filename}")
                                break # Break inner loop to switch file
                            continue
                        
                        # Save state
                        self._last_read_offset = f.tell()
                        self._last_read_file = str(filename)
                        
                        x, t = self._process_record(chunk)
                        if x is None: continue
                        
                        self.buffer.append((x, t))
                        
                        # Yield sequence
                        if len(self.buffer) >= self.seq_len:
                            window_data = self.buffer[-self.seq_len:]
                            
                            x_seq = np.stack([item[0] for item in window_data])
                            t_seq = np.stack([item[1] for item in window_data])
                            
                            yield torch.tensor(x_seq, dtype=torch.float32), torch.tensor(t_seq, dtype=torch.float32)
                            
                            # Prune
                            if len(self.buffer) > self.seq_len * 2:
                                self.buffer = self.buffer[-self.seq_len:]
                                
            except PermissionError:
                print("File locked, retrying...")
                time.sleep(1)
            except Exception as e:
                print(f"Error reading file: {e}")
                time.sleep(1)
