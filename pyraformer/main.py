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
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path to import gemini_preprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_preprocess import GeminiDataset
import pyraformer

def prepare_dataloader(args):
    """
    Load data and creating dataloaders.
    """
    print(f"Loading data from {args.data_path}...")
    # Initialize dataset (Raw, unscaled)
    # We use default input/predict size
    # Note: GeminiDataset no longer auto-scales in __init__
    full_dataset = GeminiDataset(args.data_path, input_size=2048, predict_size=1024)
    
    # Time-series split (Strictly past -> future)
    # Train: 0 -> train_size
    # Val: train_size -> end
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)
    val_size = total_len - train_size
    
    print(f"Total samples: {total_len}. Train: {train_size}, Val: {val_size}")
    
    # Compute stats on TRAINING set only
    # full_dataset.features is [N, 7]
    # We need to map dataset indices to feature indices.
    # Dataset[0] uses features[0 : seq_len]
    # Dataset[train_size-1] uses features[train_size-1 : train_size-1+seq_len]
    # So the training period covers features[0 : train_size + seq_len - 1] basically?
    # Actually simpler: standard practice is to compute mean/std on the features *available* for training.
    # Since we split by sample index, let's look at the range of time covered by train set.
    
    # Determine the end index of the raw features used by the last training sample
    # last_train_sample_idx = train_size - 1
    # It covers features up to last_train_sample_idx + seq_len
    # But wait, if we use the *future* targets of the last training sample to compute stats, 
    # we are technically using data that overlaps with the start of validation inputs?
    # Yes, in sliding window, there is overlap.
    # Strict approach: Compute stats on features[0 : train_size] (roughly). 
    # Or features corresponding to the timestamps strictly before validation starts.
    # Let's use the first train_size rows of features to compute stats.
    # This is safe and sufficient.
    
    train_features = full_dataset.features[:train_size]
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    
    print(f"Computed Normalization Stats (Train only):\nMean: {mean}\nStd: {std}")
    
    # Apply normalization to the ENTIRE dataset using TRAIN stats
    full_dataset.normalize(mean, std)
    
    # Create Subsets using indices
    # We cannot use random_split for time series
    idxs = list(range(total_len))
    train_dataset = torch.utils.data.Subset(full_dataset, idxs[:train_size])
    val_dataset = torch.utils.data.Subset(full_dataset, idxs[train_size:])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, # Shuffle training samples is fine, as long as we don't leak future into stats
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # Don't shuffle val
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader

def train_epoch(model, train_loader, optimizer, loss_fn, epoch, device, scaler):
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
        label_len = input_size // 2
        x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]
        t_dec = covariates[:, input_size:, :]
        
        optimizer.zero_grad()
        
        # Forward
        # forward(self, x_enc, t_enc, x_dec, t_dec, pretrain)
        # pretrain=False usually for standard training? 
        # The model code has `if pretrain: pred = predictor(cat([enc, dec]))`.
        # If pretrain is for pretraining tasks (reconstruction), we might want True.
        # But for forecasting, likely False.
        # Let's try pretrain=False to get `pred = predictor(dec_enc)`
        
        with autocast():
            pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
            
            # pred shape should be [batch, predict_size, enc_in] usually
            
            loss = loss_fn(pred, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
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
            label_len = input_size // 2
            x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]
            t_dec = covariates[:, input_size:, :]
            
            with autocast():
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
    
    # TF32
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Setup Data
    train_loader, val_loader = prepare_dataloader(args)
    
    # Setup Model
    model = pyraformer.Model(input_size=2048, predict_size=1024).to(device)
    
    # Compile model if available (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss
    loss_fn = nn.HuberLoss(delta=1.0)
    
    # Scaler for AMP
    scaler = GradScaler()
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch, device, scaler)
        val_loss = eval_epoch(model, val_loader, loss_fn, epoch, device)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time() - start_time:.2f}s")
        
    # Save model
    torch.save(model.state_dict(), "pyraformer_checkpoint.pth")
    print("Training complete. Model saved to pyraformer_checkpoint.pth")

if __name__ == '__main__':
    main()