import torch
import os
import sys
from types import SimpleNamespace

# Ensure path to pyraformer is correct
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
pyraformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyraformer')
if os.path.exists(pyraformer_path):
    sys.path.insert(0, pyraformer_path)

try:
    # Try importing directly if pyraformer path is in sys.path
    from pyraformer import Model
except ImportError:
    # Fallback
    try:
        from pyraformer.pyraformer import Model
    except ImportError as e:
        print(f"Failed to import pyraformer: {e}")
        sys.exit(1)

def init_checkpoint():
    print("Initializing Pyraformer model (2048 input, 1024 predict)...")
    model = Model(input_size=2048, predict_size=1024)
    
    # Create a dummy scaler state
    scaler_state = {'n': 100, 'mean': [0]*7, 'M2': [1]*7}
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler_state,
        'epoch': 0,
        'step': 0,
        'loss': 0.0
    }
    
    path = "pyraformer_checkpoint_debug.pth"
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

if __name__ == "__main__":
    init_checkpoint()
