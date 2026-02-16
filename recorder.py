import asyncio
import websockets
import json
import struct
import time
import os
from datetime import datetime, timezone
import logging
from pathlib import Path
import atexit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GeminiRecorder')

class Recorder:
    def __init__(self, symbols, data_dir="data"):
        """
        Initialize the Recorder.
        
        Args:
            symbols (list): List of symbols to subscribe to (e.g., ["btcusd", "ethusd"]).
            data_dir (str): Directory to save binary data.
        """
        self.symbols = [s.lower() for s in symbols]
        self.data_dir = Path(data_dir)
        self.running = False
        
        # User specified Gemini Fast API endpoint
        self.ws_url = "wss://wsapi.fast.gemini.com"
        
        # Register exit handler
        atexit.register(self.stop)

        
        # Binary format: >Qdddd (Big Endian: Timestamp(u64), Bid(f64), BidQty(f64), Ask(f64), AskQty(f64))
        # Size: 8 + 8 + 8 + 8 + 8 = 40 bytes
        self.struct_fmt = ">Qdddd"
        self.struct_size = struct.calcsize(self.struct_fmt)
        
        # State parsing
        self.file_handles = {} # symbol -> { 'file': file_obj, 'date': 'YYYY-MM-DD' }
        
        # Ensure base data directory exists
        for symbol in self.symbols:
            (self.data_dir / symbol).mkdir(parents=True, exist_ok=True)

    def _get_current_date(self):
        """Get current date string (UTC)."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _get_file_handle(self, symbol):
        """Get or rotate file handle for the symbol."""
        current_date = self._get_current_date()
        
        if symbol not in self.file_handles:
            self.file_handles[symbol] = {'file': None, 'date': None}
            
        handle_info = self.file_handles[symbol]
        
        # Rotate if date changed or file not open
        if handle_info['date'] != current_date or handle_info['file'] is None:
            if handle_info['file']:
                try:
                    handle_info['file'].close()
                except Exception as e:
                    logger.error(f"Error closing file for {symbol}: {e}")
            
            filename = self.data_dir / symbol / f"{current_date}.bin"
            try:
                # Open in append binary mode
                # Using default buffering (typically 4KB-8KB) to balance performance and safety
                f = open(filename, "ab") 
                handle_info['file'] = f
                handle_info['date'] = current_date
                logger.info(f"Opened new file for {symbol}: {filename}")
            except Exception as e:
                logger.critical(f"Failed to open file for {symbol}: {e}")
                return None
                
        return handle_info['file']

    def _write_record(self, symbol, timestamp, bid, bid_qty, ask, ask_qty):
        """Write a record to the binary file."""
        f = self._get_file_handle(symbol)
        if f:
            try:
                data = struct.pack(self.struct_fmt, timestamp, bid, bid_qty, ask, ask_qty)
                f.write(data)
                f.flush() # Ensure data is written to disk immediately for the reader
                
                # Update status file for IPC every 10 records or so to avoid too much I/O
                # Ideally we do this async but for now simple counter is fine
                if not hasattr(self, '_record_counters'):
                    self._record_counters = {}
                
                self._record_counters[symbol] = self._record_counters.get(symbol, 0) + 1
                
                if self._record_counters[symbol] % 10 == 0:
                    self._update_status(symbol, timestamp)
                    
            except Exception as e:
                logger.error(f"Error writing record for {symbol}: {e}")

    def _update_status(self, symbol, timestamp):
        """Update a JSON status file for the trainer to know where we are."""
        status_file = self.data_dir / symbol / "recorder_status.json"
        status = {
            "timestamp": timestamp,
            "current_file": str(self.file_handles[symbol]['file'].name),
            "last_update": time.time()
        }
        
        # Atomic write with retries for Windows file locking
        max_retries = 3
        for i in range(max_retries):
            try:
                # Atomic write not strictly necessary but good practice
                tmp_file = status_file.with_suffix('.tmp')
                with open(tmp_file, 'w') as f:
                    json.dump(status, f)
                os.replace(tmp_file, status_file)
                return # Success
            except PermissionError:
                if i < max_retries - 1:
                    time.sleep(0.01) # Short sleep before retry
                else:
                    logger.warning(f"Failed to update status for {symbol} due to file lock")
            except Exception as e:
                logger.warning(f"Failed to update status for {symbol}: {e}")
                return

    async def _process_message(self, message):
        """Process incoming websocket message."""
        try:
            data = json.loads(message)
            
            # 1. Handle Response messages (subscribe/ping)
            if "status" in data:
                if data["status"] != 200:
                    logger.error(f"Error response from server: {data}")
                else:
                    logger.debug(f"Success response: {data}")
                return

            # 2. Handle Book Ticker Message
            # Docs: { "u": ..., "s": "btcusd", "b": "...", "B": "...", "a": "...", "A": "..." }
            if "s" in data and "b" in data and "a" in data:
                symbol = data["s"].lower()
                
                # Filter for subscribed symbols
                if symbol not in self.symbols:
                    return

                # Parse fields
                # Timestamp: prefer event time "E", fallback to "u" (updateId) isn't time but monotonic.
                # Docs show "E" as timestamp
                if "E" in data:
                    timestamp = int(data["E"])
                else:
                    timestamp = int(time.time() * 1000)

                bid = float(data["b"])
                bid_qty = float(data["B"])
                ask = float(data["a"])
                ask_qty = float(data["A"])
                
                # Validating data integrity
                if bid > 0 and ask > 0:
                     self._write_record(symbol, timestamp, bid, bid_qty, ask, ask_qty)
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _heartbeat(self, ws):
        """Send periodic pings."""
        while self.running:
            try:
                await asyncio.sleep(15) # Send ping every 15 seconds
                ping_msg = {
                    "id": int(time.time()),
                    "method": "ping",
                    "params": {}
                }
                await ws.send(json.dumps(ping_msg))
                logger.debug("Sent ping")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def connect(self):
        """Establish websocket connection and subscribe."""
        while self.running:
            try:
                # Adding ping_interval=None to handle pings manually or let library handle standard pings if server supports it.
                # Since we have a 'ping' method in docs, we should probably use application-layer pings.
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    logger.info(f"Connected to {self.ws_url}")
                    
                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws))
                    
                    # Subscribe
                    # Params format: ["btcusd@bookTicker", ...]
                    params = [f"{s}@bookTicker" for s in self.symbols]
                    sub_msg = {
                        "id": 1,
                        "method": "subscribe",
                        "params": params
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"Sent subscription: {sub_msg}")
                    
                    try:
                        while self.running:
                            msg = await ws.recv()
                            await self._process_message(msg)
                    finally:
                        heartbeat_task.cancel()
                        
            except websockets.ConnectionClosed:
                logger.warning("Connection closed, reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Connection error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                pass

    def start(self):
        """Start the recorder."""
        self.running = True
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.connect())
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()
            
    def stop(self):
        """Stop the recorder and close files."""
        self.running = False
        logger.info("Stopping recorder...")
        
        # Close all file handles
        for symbol, handle_info in self.file_handles.items():
            if handle_info['file']:
                try:
                    handle_info['file'].close()
                    logger.info(f"Closed file for {symbol}")
                except Exception as e:
                    logger.error(f"Error closing file for {symbol}: {e}")
        self.file_handles.clear()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini Data Recorder")
    parser.add_argument("--symbols", nargs="+", default=["btcusd"], help="List of symbols to record")
    parser.add_argument("--dir", default="data", help="Data directory")
    args = parser.parse_args()
    
    recorder = Recorder(args.symbols, data_dir=args.dir)
    recorder.start()
