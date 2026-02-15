import torch
from gemini_preprocess import GeminiDataset
import glob
import os

def main():
    # Find latest bin file
    files = glob.glob("data/btcusd/*.bin")
    if not files:
        print("No binary files found in data/btcusd/")
        return

    latest_file = max(files, key=os.path.getctime)
    print(f"Testing dataset with file: {latest_file}")
    
    try:
        dataset = GeminiDataset(latest_file)
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample 0: {sample}")
            print(f"Sample shape: {sample.shape}")
            print("Successfully read new format!")
        else:
            print("Dataset is empty.")
            
    except Exception as e:
        print(f"Failed to load dataset: {e}")

if __name__ == "__main__":
    main()
