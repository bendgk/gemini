import argparse
import sys
import os
import time
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

# Add parent directory to path to import gemini_preprocess and pyraformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live_dataset import LiveGeminiDataset

# Import Pyraformer
pyraformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyraformer')
if os.path.exists(pyraformer_path):
    sys.path.insert(0, pyraformer_path)

try:
    import pyraformer
    import layers 
except ImportError:
    pass

def load_model(checkpoint_path, device, input_size=168, predict_size=168):
    """Load model from checkpoint."""
    # Check if pyraformer module has Model class
    if hasattr(pyraformer, 'Model'):
        model = pyraformer.Model(input_size=input_size, predict_size=predict_size).to(device)
    else:
        # Fallback if pyraformer is a package without implicit export
        from pyraformer.pyraformer import Model
        model = Model(input_size=input_size, predict_size=predict_size).to(device)
        
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Check if it's a full checkpoint dict or just state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Using random weights.")
        
    model.eval()
    return model

def check_for_checkpoint_update(checkpoint_path, last_mtime):
    """Check if checkpoint file has been modified."""
    if not os.path.exists(checkpoint_path):
        return False, last_mtime
        
    current_mtime = os.path.getmtime(checkpoint_path)
    if current_mtime > last_mtime:
        return True, current_mtime
    return False, last_mtime

def visualize_inference(inputs, targets, preds, step, output_dir, symbol):
    """Visualize and save inference results."""
    # Inputs: [batch, input_size, 7]
    # Targets: [batch, predict_size, 7]
    # Preds: [batch, predict_size, 7]
    
    # We'll visualize the first item in the batch
    idx = 0
    
    # Feature 0 is Bid Price
    input_seq = inputs[idx, :, 0]
    target_seq = targets[idx, :, 0]
    pred_seq = preds[idx, :, 0]
    
    plt.figure(figsize=(12, 6))
    
    # Create x axis
    input_len = len(input_seq)
    pred_len = len(target_seq)
    
    x_input = np.arange(input_len)
    x_pred = np.arange(input_len, input_len + pred_len)
    
    plt.plot(x_input, input_seq, label='Input (Bid Price)', color='blue')
    plt.plot(x_pred, target_seq, label='Ground Truth', color='green')
    plt.plot(x_pred, pred_seq, label='Prediction', color='red', linestyle='--')
    
    plt.title(f'Continuous Inference - Step {step}')
    plt.legend()
    plt.grid(True)
    
    # Save to file
    out_file = os.path.join(output_dir, f'live_inference.png')
    plt.savefig(out_file)
    plt.close()
    
    # Also save a timestamped version occasionally or just keep latest
    # For web serving, maintaining a consistent filename is better.

def update_inference_status(symbol, step, loss, output_dir):
    try:
        status_file = os.path.join(output_dir, "inference_status.json")
        status = {
            "timestamp": time.time(),
            "step": step,
            "loss": loss,
            "image": "live_inference.png"
        }
        with open(status_file + ".tmp", 'w') as f:
            json.dump(status, f)
        os.replace(status_file + ".tmp", status_file)
    except Exception as e:
        print(f"Failed to update status: {e}")

def main():
    parser = argparse.ArgumentParser(description='Continuous Pyraformer Inference')
    parser.add_argument('--symbol', type=str, default='btcusd', help='Symbol to infer on')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--steps', type=int, default=0, help='Number of steps to run (0 for infinite)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Starting continuous inference for {args.symbol} on {device}")
    
    # 1. Setup Dataset
    input_size = 8192
    predict_size = 1024
    
    # Use same parameters as training
    dataset = LiveGeminiDataset(args.data_dir, args.symbol, input_size=input_size, predict_size=predict_size)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0) 
    
    # 2. Setup Model & Checkpoint
    checkpoint_dir = os.path.join(args.data_dir, args.symbol, "checkpoints")
    latest_ckpt = os.path.join(checkpoint_dir, "latest.pth")
    
    model = load_model(latest_ckpt, device, input_size, predict_size)
    last_ckpt_mtime = 0
    if os.path.exists(latest_ckpt):
        last_ckpt_mtime = os.path.getmtime(latest_ckpt)
    
    loss_fn = nn.MSELoss()
    
    # Output directory for images
    output_dir = os.path.join(args.data_dir, args.symbol, "inference")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Waiting for data stream...")
    
    step = 0
    
    try:
        for batch_data, batch_covariates in dataloader:
            step += 1
            
            if args.steps > 0 and step > args.steps:
                print(f"Completed {args.steps} steps. Exiting.")
                break
            
            # Check for model update
            updated, new_mtime = check_for_checkpoint_update(latest_ckpt, last_ckpt_mtime)
            if updated:
                print("Checkpoint updated. Reloading model...")
                try:
                    # Reload model weights
                    checkpoint = torch.load(latest_ckpt, map_location=device)
                    # Handle scaler update if present
                    if 'scaler_state_dict' in checkpoint:
                         dataset.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                         print("Scaler updated.")
                         
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    last_ckpt_mtime = new_mtime
                    model.eval()
                except Exception as e:
                    print(f"Failed to reload checkpoint: {e}")

            # Inference
            with torch.no_grad():
                data = batch_data.to(device)
                covariates = batch_covariates.to(device)
                
                x_enc = data[:, :input_size, :]
                t_enc = covariates[:, :input_size, :]
                
                target = data[:, input_size:, :]
                # For inference/validation, we often feed zeros to decoder to see what it hallucinates/predicts
                x_dec = torch.zeros_like(target).to(device)
                t_dec = covariates[:, input_size:, :]
                
                pred = model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
                
                loss = loss_fn(pred, target).item()
                
                print(f"Step {step} | Loss: {loss:.6f}")
                
                # Visualize
                if step % 5 == 0: # Update visual every 5 steps to save IO
                    visualize_inference(
                        x_enc.cpu().numpy(), 
                        target.cpu().numpy(), 
                        pred.cpu().numpy(), 
                        step, 
                        output_dir,
                        args.symbol
                    )
                    update_inference_status(args.symbol, step, loss, output_dir)

    except KeyboardInterrupt:
        print("Inference stopped by user.")

if __name__ == '__main__':
    main()
