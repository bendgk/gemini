import argparse
import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Add parent directory to path to import gemini_preprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_preprocess import GeminiDataset
import pyraformer

def prepare_dataloader(args):
    """
    Load data and creating dataloaders.
    """
    print(f"Loading data from {args.data_path}...")
    # Initialize dataset
    # We use the default 168 input/predict size matching the Model definition
    full_dataset = GeminiDataset(args.data_path, input_size=168, predict_size=168)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, loss_fn, epoch, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, (data, covariates) in enumerate(progress_bar):
        # data: [batch, seq_len, 7]
        # covariates: [batch, seq_len, 4]
        
        # Pyraformer expects:
        # x_enc, t_enc (input)
        # x_dec, t_dec (prediction target, with specific masking usually)
        
        # In this implementation, we split the sequence:
        # Input part: [0 : input_size]
        # Prediction part: [input_size : input_size + predict_size]
        
        input_size = model.input_size
        predict_size = model.predict_size
        
        data = data.to(device)
        covariates = covariates.to(device)
        
        x_enc = data[:, :input_size, :]
        t_enc = covariates[:, :input_size, :]
        
        # For decoder input, commonly we use the last value of encoder input repeated or zeros,
        # plus the covariates for the prediction horizon.
        # But Pyraformer might expect the target sequence for teacher forcing?
        # Looking at forward: enc_output = encoder(x_enc, t_enc)
        # dec_enc = decoder(x_dec, t_dec, enc_output)
        # pred = predictor(dec_enc)
        
        # We need to construct x_dec.
        # Usually for transformer forecasting:
        # x_dec starts with a start token (last of enc) and then targets (masked) or just placeholders.
        # Let's use the actual future values as targets for loss, but construction of x_dec 
        # depends on model design. 
        # Code reference `pyraformer.py`: 
        # `dec_enc = self.decoder(x_dec, t_dec, enc_output)`
        
        # We'll assume x_dec should be the target sequence but we mask it/use it for teacher forcing if supported,
        # or just zeros if purely autoregressive/inference mode?
        # The model's `forward` in `pyraformer.py` doesn't show explicit masking logic inside `forward` except `get_subsequent_mask` in `__init__`.
        # `get_subsequent_mask` is passed to `self.decoder`.
        # So `x_dec` likely inputs all zeros or mean for the prediction part?
        # Or it expects the ground truth for training?
        
        # Standard approach: x_dec = zeros or last-value padded
        # And we compute loss against the actual future.
        
        target = data[:, input_size:, :] # [batch, predict_size, 7]
        
        # Prepare decoder input:
        # Some implementations use a "start token" + zeros
        # Here we'll try passing zeros for prediction range as a placeholder, 
        # or the "last observed value" repeated.
        # Let's use zeros for simplicity as done in many Informer/Pyraformer implementations.
        x_dec = torch.zeros_like(target).to(device)
        t_dec = covariates[:, input_size:, :]
        
        optimizer.zero_grad()
        
        # Forward
        # forward(self, x_enc, t_enc, x_dec, t_dec, pretrain)
        # pretrain=False usually for standard training? 
        # The model code has `if pretrain: pred = predictor(cat([enc, dec]))`.
        # If pretrain is for pretraining tasks (reconstruction), we might want True.
        # But for forecasting, likely False.
        # Let's try pretrain=False to get `pred = predictor(dec_enc)`
        
        pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
        
        # pred shape should be [batch, predict_size, enc_in] usually
        
        loss = loss_fn(pred, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(train_loader)

def eval_epoch(model, val_loader, loss_fn, epoch, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (data, covariates) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)):
            input_size = model.input_size
            
            data = data.to(device)
            covariates = covariates.to(device)
            
            x_enc = data[:, :input_size, :]
            t_enc = covariates[:, :input_size, :]
            
            target = data[:, input_size:, :]
            x_dec = torch.zeros_like(target).to(device)
            t_dec = covariates[:, input_size:, :]
            
            pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
            
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            
    return total_loss / len(val_loader)

def main():
    parser = argparse.ArgumentParser(description='Pyraformer Training')
    parser.add_argument('--data_path', type=str, default='data/btcusd/test_data.bin', help='Path to binary data file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Num workers for dataloader')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    # Check if data exists, if not use the dummy file created by preprocess script
    if not os.path.exists(args.data_path):
        print(f"Data file {args.data_path} not found. Creating a dummy file for testing...")
        # Create dummy file logic or warn
        # For simplicity, we assume we might run gemini_preprocess.py first or point to valid data.
        # But let's fallback to 'test_data.bin' if default is used and missing.
        if args.data_path == 'data/btcusd/test_data.bin' and not os.path.exists(args.data_path):
             # Try absolute path or local test_data.bin
             if os.path.exists("test_data.bin"):
                 args.data_path = "test_data.bin"
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Setup Data
    train_loader, val_loader = prepare_dataloader(args)
    
    # Setup Model
    model = pyraformer.Model().to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss
    loss_fn = nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch, device)
        val_loss = eval_epoch(model, val_loader, loss_fn, epoch, device)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time() - start_time:.2f}s")
        
    # Save model
    torch.save(model.state_dict(), "pyraformer_checkpoint.pth")
    print("Training complete. Model saved to pyraformer_checkpoint.pth")

if __name__ == '__main__':
    main()