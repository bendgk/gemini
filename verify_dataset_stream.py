
import torch
from torch.utils.data import DataLoader
from live_dataset import LiveGeminiDataset
import time

def verify_stream():
    symbol = "btcusd"
    data_dir = "data"
    print(f"Verifying stream for {symbol} in {data_dir}...")
    
    dataset = LiveGeminiDataset(data_dir, symbol, input_size=168, predict_size=168)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    
    print("Starting loop...")
    last_val = None
    count = 0
    
    for batch_data, batch_covariates in dataloader:
        # batch_data: [1, seq_len, 7]
        # Check last feature of last time step (Mid Price or Bid)
        # Feature 0 is Bid
        current_val = batch_data[0, -1, 0].item()
        
        # Check timestamp (implicit in sequence, or check change)
        
        print(f"Step {count}: Last Bid = {current_val:.6f}")
        
        if last_val is not None:
            if current_val == last_val:
                print("WARNING: Value unchanged from last step!")
            else:
                print("Value changed.")
        
        last_val = current_val
        count += 1
        
        if count >= 10:
            break

if __name__ == "__main__":
    verify_stream()
