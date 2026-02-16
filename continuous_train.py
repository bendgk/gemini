import argparse
import sys
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
import shutil

# Add parent directory to path to import gemini_preprocess and pyraformer
# Add parent directory to path to import gemini_preprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_dataset import LiveGeminiDataset

# Import Pyraformer
# We need to handle the case where pyraformer is a folder in the current directory
# but we want to import pyraformer.py FROM inside that folder, and also allow it to import 'layers' (sibling).
pyraformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyraformer')
if os.path.exists(pyraformer_path):
    # Insert at 0 to prioritize modules inside pyraformer/ over the folder itself in current dir
    sys.path.insert(0, pyraformer_path)

try:
    import pyraformer
    import layers # Check if layers is importable to confirm path setup
except ImportError:
    pass

def save_checkpoint(model, optimizer, scaler, epoch, step, loss, path):
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(), 
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, scaler, path, device):
    if os.path.exists(path):
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint and scaler:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        return checkpoint.get('epoch', 0), checkpoint.get('step', 0)
    return 0, 0

def update_status(symbol, step, loss, checkpoint_path):
    try:
        status_file = f"data/{symbol}/trainer_status.json"
        status = {
            "timestamp": time.time(),
            "step": step,
            "loss": loss,
            "latest_checkpoint": checkpoint_path
        }
        with open(status_file + ".tmp", 'w') as f:
            json.dump(status, f)
        os.replace(status_file + ".tmp", status_file)
    except Exception as e:
        print(f"Failed to update status: {e}")

def main():
    parser = argparse.ArgumentParser(description='Continuous Pyraformer Training')
    parser.add_argument('--symbol', type=str, default='btcusd', help='Symbol to train on')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (keep small for live updates)')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--checkpoint_itv', type=int, default=100, help='Steps between checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Starting continuous training for {args.symbol} on {device}")
    
    # 1. Setup Dataset
    input_size = 256
    predict_size = 256
    
    # We use batch_size 168 (seq len) in dataset but here batch_size is num sequences
    dataset = LiveGeminiDataset(args.data_dir, args.symbol, input_size=input_size, predict_size=predict_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0) # Must be 0 for IterableDataset simplicity
    
    # 2. Setup Model
    # Pyraformer implementation check
    if hasattr(pyraformer, 'Model'):
        model = pyraformer.Model(input_size=input_size, predict_size=predict_size).to(device)
    else:
        # Fallback if pyraformer is a package without implicit export
        from pyraformer.pyraformer import Model
        model = Model(input_size=input_size, predict_size=predict_size).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    # 3. Load Checkpoint
    checkpoint_dir = os.path.join(args.data_dir, args.symbol, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    latest_ckpt = os.path.join(checkpoint_dir, "latest.pth")
    epoch, step = load_checkpoint(model, optimizer, dataset.scaler, latest_ckpt, device)
    
    model.train()
    
    print("Waiting for data stream...")
    
    try:
        total_loss = 0
        count = 0
        
        # Infinite loop via DataLoader
        for batch_data, batch_covariates in dataloader:
            step += 1
            
            # Prepare data
            # batch_data: [batch, seq_len, 7]
            # batch_covariates: [batch, seq_len, 4]
            
            data = batch_data.to(device)
            covariates = batch_covariates.to(device)
            
            # Split Input/Target
            x_enc = data[:, :input_size, :]
            t_enc = covariates[:, :input_size, :]
            
            target = data[:, input_size:, :]
            # Zero placeholder for decoder input
            x_dec = torch.zeros_like(target).to(device)
            t_dec = covariates[:, input_size:, :]
            
            # Train Step
            optimizer.zero_grad()
            pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            
            # Logging
            current_loss = loss.item()
            total_loss += current_loss
            count += 1
            
            if step % 10 == 0:
                avg_loss = total_loss / count
                print(f"Step {step} | Loss: {current_loss:.6f} | Avg: {avg_loss:.6f}")
                total_loss = 0
                count = 0
                
            # Checkpointing
            if step % args.checkpoint_itv == 0:
                print(f"Saving checkpoint at step {step}...")
                save_checkpoint(model, optimizer, dataset.scaler, epoch, step, current_loss, latest_ckpt)
                update_status(args.symbol, step, current_loss, latest_ckpt)
                
                # Historic backup occasionally (every 10 checkpoints = 1000 steps)
                if step % (args.checkpoint_itv * 10) == 0:
                    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = os.path.join(checkpoint_dir, f"ckpt_{step}_{ts_str}.pth")
                    shutil.copy(latest_ckpt, backup_path)
                    print(f"Backed up checkpoint to {backup_path}")

    except KeyboardInterrupt:
        print("Training stopped by user. Saving checkpoint...")
        save_checkpoint(model, optimizer, dataset.scaler, epoch, step, 0, latest_ckpt)
        
if __name__ == '__main__':
    main()
