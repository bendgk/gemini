import struct
import os
import random
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from live_dataset import LiveGeminiDataset
from continuous_infer import load_model

def create_large_dummy_data(file_path, num_records=10000):
    print(f"Creating dummy data with {num_records} records at {file_path}...")
    with open(file_path, "wb") as f:
        # Start time
        start_ts = int(datetime.now().timestamp() * 1000)
        
        bid = 50000.0
        
        for i in range(num_records):
            ts = start_ts + i * 1000 # 1 second intervals
            
            # Random walk
            bid += (random.random() - 0.5) * 10
            bid_qty = 1.0 + random.random()
            ask = bid + 0.5 + random.random()
            ask_qty = 1.0 + random.random()
            
            f.write(struct.pack(">Qdddd", ts, bid, bid_qty, ask, ask_qty))
            
    print("Done creating data.")

def verify_inference():
    symbol = "test_large"
    data_dir = "data_test_large"
    os.makedirs(os.path.join(data_dir, symbol), exist_ok=True)
    
    # Create dummy bin file
    bin_file = os.path.join(data_dir, symbol, "test.bin")
    create_large_dummy_data(bin_file, num_records=10000)
    
    # Create dummy status file so LiveGeminiDataset finds it
    import json
    with open(os.path.join(data_dir, symbol, "recorder_status.json"), "w") as f:
        json.dump({
            "timestamp": 0,
            "current_file": os.path.abspath(bin_file),
            "last_update": 0
        }, f)

    # Initialize Dataset
    print("Initializing dataset...")
    input_size = 8192
    predict_size = 1024
    
    dataset = LiveGeminiDataset(data_dir, symbol, input_size=input_size, predict_size=predict_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    # Load Model (cpu)
    device = torch.device("cpu")
    print("Initializing model...")
    # Pass dummy checkpoint path, it will initialize random weights
    model = load_model("dummy_ckpt.pth", device, input_size, predict_size)
    
    print("Running inference step...")
    for batch_data, batch_covariates in dataloader:
        data = batch_data.to(device)
        covariates = batch_covariates.to(device)
        
        x_enc = data[:, :input_size, :]
        t_enc = covariates[:, :input_size, :]
        
        target = data[:, input_size:, :]
        x_dec = torch.zeros_like(target).to(device)
        t_dec = covariates[:, input_size:, :]
        
        print(f"Input shape: {x_enc.shape}")
        print(f"Target shape: {target.shape}")
        
        with torch.no_grad():
            pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
            
        print(f"Prediction shape: {pred.shape}")
        print("Inference successful!")
        break

if __name__ == "__main__":
    verify_inference()
