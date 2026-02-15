import subprocess
import time
import os
import sys
import signal
from pathlib import Path

def run_verification():
    print("Starting Verification...")
    
    # 1. Start Recorder (Dummy mode if possible, but we use real recorder with existing logic)
    # We'll use a real symbol so subscription works
    symbol = "btcusd"
    data_dir = "data_test"
    
    # Clean up previous test
    import shutil
    if os.path.exists(data_dir):
        try:
            shutil.rmtree(data_dir)
        except PermissionError:
            print("Could not clean up data_test (files locked?), using existing.")
            
    os.makedirs(data_dir, exist_ok=True)
    
    # We must ensure recorder has time to create directory structure
    (Path(data_dir) / symbol).mkdir(parents=True, exist_ok=True)
    
    print(f"Launching Recorder for {symbol} in {data_dir}...")
    recorder_cmd = [sys.executable, "recorder.py", "--symbols", symbol, "--dir", data_dir]
    recorder_proc = subprocess.Popen(recorder_cmd)
    
    time.sleep(10) # Wait longer for connection and initial data
    
    if recorder_proc.poll() is not None:
        print("Recorder failed to start!")
        return

    # Check if file created
    files = list((Path(data_dir) / symbol).glob("*.bin"))
    if not files:
        print("Recorder did not create any bin file yet. Waiting...")
        time.sleep(10)
        files = list((Path(data_dir) / symbol).glob("*.bin"))
        if not files:
            print("Recorder still failed to create files. Aborting.")
            recorder_proc.terminate()
            return
            
    print(f"Recorder created: {[f.name for f in files]}")
    
    # 2. Start Trainer
    print("Launching Continuous Trainer...")
    trainer_cmd = [
        sys.executable, "continuous_train.py", 
        "--symbol", symbol, 
        "--data_dir", data_dir,
        "--batch_size", "2", # Small batch to ensure updates happen quickly
        "--checkpoint_itv", "5",
        "--device", "cpu" # Force CPU for test stability
    ]
    trainer_proc = subprocess.Popen(trainer_cmd)
    
    # Monitor for 30 seconds
    try:
        for i in range(30):
            time.sleep(1)
            if trainer_proc.poll() is not None:
                print("Trainer crashed!")
                break
            if recorder_proc.poll() is not None:
                print("Recorder crashed!")
                break
                
            if i % 5 == 0:
                print(f"Running... {i}s")
                
        # Check for checkpoints
        ckpt_dir = Path(data_dir) / symbol / "checkpoints"
        ckpts = list(ckpt_dir.glob("*.pth"))
        if ckpts:
            print(f"SUCCESS: Checkpoints created: {[f.name for f in ckpts]}")
        else:
            print("FAILURE: No checkpoints created.")
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping processes...")
        recorder_proc.terminate()
        trainer_proc.terminate()
        recorder_proc.wait()
        trainer_proc.wait()
        print("Done.")

if __name__ == "__main__":
    run_verification()
