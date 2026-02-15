import struct
import glob
import os
from datetime import datetime

def read_binary_file(filepath):
    struct_fmt = ">Qdddd"
    struct_size = struct.calcsize(struct_fmt)
    
    print(f"Reading file: {filepath}")
    
    with open(filepath, "rb") as f:
        while True:
            data = f.read(struct_size)
            if not data or len(data) < struct_size:
                break
            
            timestamp, bid, bid_qty, ask, ask_qty = struct.unpack(struct_fmt, data)
            
            # Convert timestamp to human readable
            # Timestamp is likely in nanoseconds (19 digits) or milliseconds (13 digits)
            # 1771191589758382800 (19 digits) -> nanoseconds
            if timestamp > 1e16:
                ts_str = datetime.fromtimestamp(timestamp / 1e9).strftime('%Y-%m-%d %H:%M:%S.%f')
            else:
                ts_str = datetime.fromtimestamp(timestamp / 1e3).strftime('%Y-%m-%d %H:%M:%S.%f')
                
            print(f"[{ts_str}] Bid: {bid:.2f} ({bid_qty:.5f}) | Ask: {ask:.2f} ({ask_qty:.5f})")

def main():
    # Find latest bin file
    files = glob.glob("data/btcusd/*.bin")
    if not files:
        print("No binary files found in data/btcusd/")
        return
        
    latest_file = max(files, key=os.path.getctime)
    read_binary_file(latest_file)

if __name__ == "__main__":
    main()
