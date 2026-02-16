import argparse
import sys
import os
import time
import json
import logging
from datetime import datetime
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, IterableDataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_dataset import LiveGeminiDataset
from execution.gemini_client import GeminiClient, MockGeminiClient
from strategies.voting_strategy import VotingStrategy
from risk.risk_manager import RiskManager
from strategies.base_strategy import Signal

# Import Pyraformer
pyraformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pyraformer')
if os.path.exists(pyraformer_path):
    sys.path.insert(0, pyraformer_path)

try:
    import pyraformer
except ImportError:
    pass

class MockDataset(IterableDataset):
    def __init__(self, input_size, predict_size):
        self.input_size = input_size
        self.predict_size = predict_size
        self.seq_len = input_size + predict_size
        # Mock scaler
        from types import SimpleNamespace
        self.scaler = SimpleNamespace(n=0, mean=0, M2=0, load_state_dict=lambda x: None)

    def __iter__(self):
        while True:
            # Generate random data
            # [seq_len, 7]
            x = torch.randn(self.seq_len, 7)
            # [seq_len, 4]
            t = torch.randn(self.seq_len, 4)
            time.sleep(0.1)
            yield x, t # DataLoader adds batch dim
            
            # Wait, dataloader collates. If batch_size=1, dataloader expects item.
            # If iterable dataset yields tensors, dataloader stacks them.
            # My LiveDataset yields (tensor, tensor).
            
    def __len__(self):
        return 100000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TradingEngine')

def check_for_checkpoint_update(checkpoint_path, last_mtime):
    """Check if checkpoint file has been modified."""
    if not os.path.exists(checkpoint_path):
        return False, last_mtime
        
    current_mtime = os.path.getmtime(checkpoint_path)
    if current_mtime > last_mtime:
        return True, current_mtime
    return False, last_mtime

class TradingEngine:
    def __init__(self, symbol, data_dir, model_path, api_key=None, api_secret=None, paper_trading=True, mock_data=False, visualize=False, buy_threshold=0.005, sell_threshold=0.005):
        self.symbol = symbol
        self.data_dir = data_dir
        self.paper_trading = paper_trading
        self.mock_data = mock_data
        self.visualize_mode = visualize
        self.checkpoint_path = model_path
        self.last_ckpt_mtime = 0
        if os.path.exists(model_path):
            self.last_ckpt_mtime = os.path.getmtime(model_path)
        
        self.visualize_mode = visualize
        self.checkpoint_path = model_path
        self.last_ckpt_mtime = 0
        if os.path.exists(model_path):
            self.last_ckpt_mtime = os.path.getmtime(model_path)
            
        # History tracking for aggregation
        self.history = {
            'steps': [],
            'portfolio_values': [],
            'hold_values': [], # Passive Hold Strategy
            'prices': [],
            'signals': [] # list of (step, type, price)
        }
        
        # 1. Setup Client
        if paper_trading:
            logger.info("Initializing Paper Trading Mode")
            self.client = MockGeminiClient()
        else:
            logger.info("Initializing Live Trading Mode")
            # Verify keys
            if not api_key or not api_secret:
                raise ValueError("API Key and Secret required for live trading")
            self.client = GeminiClient(api_key, api_secret, sandbox=True) # Default to Sandbox for safety first
            
        # 2. Setup Risk Manager
        self.risk_manager = RiskManager()
        
        # 3. Setup Strategy
        self.strategy = VotingStrategy(symbol, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
        
        # 4. Setup Data
        self.input_size = 2048
        self.predict_size = 1024
        
        if self.mock_data:
            logger.info("Using Mock Dataset")
            self.dataset = MockDataset(self.input_size, self.predict_size)
        else:
            self.dataset = LiveGeminiDataset(data_dir, symbol, input_size=self.input_size, predict_size=self.predict_size)
            
        self.dataloader = DataLoader(self.dataset, batch_size=1, num_workers=0)

        # 5. Setup Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)

    def _load_model(self, checkpoint_path):
        logger.info(f"Loading model from {checkpoint_path}")
        # Assuming Pyraformer architecture
        if hasattr(pyraformer, 'Model'):
            model = pyraformer.Model(input_size=2048, predict_size=1024).to(self.device)
        else:
            from pyraformer.pyraformer import Model
            model = Model(input_size=2048, predict_size=1024).to(self.device)
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            # Load scaler if available
            if 'scaler_state_dict' in checkpoint:
                if hasattr(self.dataset, 'scaler'):
                    self.dataset.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                    logger.info("Scaler state loaded from checkpoint")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
            
            
        model.eval()
        return model

    def visualize_step(self, step, inputs, targets, prediction, signal, portfolio_value=None, initial_portfolio_value=0.0):
        """Visualize current step using matplotlib."""
        try:
            # Inputs: [batch, input_size, 7] -> take batch 0, feature 0 (Bid)
            input_seq = inputs[0, :, 0].cpu().numpy()
            
            # No GT for future
            target_seq = None
            if targets is not None:
                target_seq = targets[0, :, 0].cpu().numpy()
                
            pred_seq = prediction[:, 0]
            
            input_len = len(input_seq)
            pred_len = len(pred_seq)
            
            # Create a figure with 2 subplots (Micro vs Macro)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
            
            # Match end of history with start of prediction for visual continuity
            # Prepend the last history point to the prediction sequence
            last_hist_val = input_seq[-1]
            connected_pred_seq = np.insert(pred_seq, 0, last_hist_val)
            
            # x values for prediction need to start at the last history index
            x_input = np.arange(input_len)
            x_pred = np.arange(input_len - 1, input_len + pred_len)
            
            ax1.plot(x_input, input_seq, label='History (Bid)', color='blue')
            if target_seq is not None:
                # Also align GT if present
                last_gt_val = input_seq[-1] 
                connected_target = np.insert(target_seq, 0, last_gt_val)
                ax1.plot(x_pred, connected_target, label='Future (GT)', color='green', alpha=0.5)
                
            ax1.plot(x_pred, connected_pred_seq, label='Prediction', color='red', linestyle='--')
            
            title = f"Step {step} | Micro View | "
            if signal:
                title += f"Signal: {signal.type} @ {signal.price} "
                color = 'green' if signal.type == 'BUY' else 'red' if signal.type == 'SELL' else 'gray'
                ax1.axvline(x=input_len, color=color, linestyle='-', label=f'Signal: {signal.type}')
            else:
                title += "No Signal"
                
            ax1.set_title(title)
            ax1.legend()
            ax1.grid(True)
            
            # --- Subplot 2: Macro View (Portfolio & Price) ---
            if len(self.history['steps']) > 0:
                ax2.set_title("Macro View: Portfolio & Price History")
                ax2.set_xlabel("Step")
                
                # Plot Portfolio Value on left y-axis
                color = 'tab:blue'
                ax2.plot(self.history['steps'], self.history['portfolio_values'], color=color, label='Model Portfolio ($)')
                
                # Plot Hold Strategy Value
                if len(self.history['hold_values']) > 0:
                    ax2.plot(self.history['steps'], self.history['hold_values'], color='orange', linestyle='--', label='Buy & Hold ($)')
                
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylabel('Value ($)', color=color)
                ax2.legend(loc='upper left')
                
                # Plot Price on right y-axis
                ax3 = ax2.twinx() 
                color = 'tab:gray'
                ax3.plot(self.history['steps'], self.history['prices'], color=color, alpha=0.5, label='Price', linestyle=':')
                ax3.tick_params(axis='y', labelcolor=color)
                ax3.set_ylabel('Price', color=color)
                
                # Add signal markers
                for s_step, s_type, s_price in self.history['signals']:
                    marker = '^' if s_type == 'BUY' else 'v'
                    color = 'green' if s_type == 'BUY' else 'red'
                    # Plot on Price axis
                    ax3.scatter(s_step, s_price, color=color, marker=marker, s=50, zorder=5)

            # --- Display PnL at the bottom ---
            if portfolio_value is not None and initial_portfolio_value > 0:
                profit = portfolio_value - initial_portfolio_value
                profit_pct = (profit / initial_portfolio_value) * 100
                pnl_color = 'green' if profit >= 0 else 'red'
                pnl_text = f"Current Portfolio: ${portfolio_value:,.2f}  |  PnL: ${profit:,.2f} ({profit_pct:+.2f}%)"
                
                # Add text box at the bottom of the figure
                plt.figtext(0.5, 0.02, pnl_text, ha='center', fontsize=14, 
                            bbox={"facecolor": pnl_color, "alpha": 0.2, "pad": 5},
                            color='black', weight='bold')
                            
                # Adjust layout to make room for text
                plt.subplots_adjust(bottom=0.1)
            else:
                plt.tight_layout()
            
            # Save to file
            out_dir = os.path.join(self.data_dir, self.symbol, "visualization")
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, "live_trading_dashboard.png"))
            plt.close()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")

    def generate_future_covariates(self, last_cov, num_steps):
        """
        Extrapolate future covariates based on the last known covariate set.
        Assumes 1-second time steps.
        last_cov: [4] tensor (sec, min, hour, day) - Normalized
        """
        # Extract scalar values
        c_sec, c_min, c_hour, c_day = last_cov[0].item(), last_cov[1].item(), last_cov[2].item(), last_cov[3].item()
        
        # Denormalize to approximate integers
        # Formula: norm = val/cycle - 0.5  =>  val = (norm + 0.5) * cycle
        current_sec = round((c_sec + 0.5) * 60)
        current_min = round((c_min + 0.5) * 60)
        current_hour = round((c_hour + 0.5) * 24)
        current_day = round((c_day + 0.5) * 7)
        
        future_covs = []
        for _ in range(num_steps):
            current_sec += 1
            if current_sec >= 60:
                current_sec = 0
                current_min += 1
                if current_min >= 60:
                    current_min = 0
                    current_hour += 1
                    if current_hour >= 24:
                        current_hour = 0
                        current_day += 1
                        if current_day >= 7:
                            current_day = 0
                            
            # Normalize back
            future_covs.append([
                current_sec / 60.0 - 0.5,
                current_min / 60.0 - 0.5,
                current_hour / 24.0 - 0.5,
                current_day / 7.0 - 0.5
            ])
            
        return torch.tensor(future_covs, dtype=torch.float32).to(self.device).unsqueeze(0) # [1, steps, 4]

    def run(self, max_steps=0):
        logger.info("Starting Trading Loop...")
        
        # Initial Sync
        try:
            balances = self.client.get_balances()
            if balances:
                self.risk_manager.sync_positions(balances)
                
            # Capture Initial Portfolio Value for PnL calculation
            # We need current price for this. Fetch ticker once.
            ticker = self.client.get_ticker(self.symbol)
            if ticker and balances:
                usd_bal = 0
                btc_bal = 0
                for b in balances:
                    if b['currency'] == 'USD': usd_bal += float(b['available'])
                    if b['currency'] == 'BTC': btc_bal += float(b['available'])
                last_price = float(ticker['last'])
                
                self.initial_usd = usd_bal
                self.initial_btc = btc_bal
                self.initial_portfolio_value = usd_bal + (btc_bal * last_price)
                logger.info(f"Initial Portfolio: ${self.initial_portfolio_value:,.2f} (USD: ${usd_bal:,.2f}, BTC: {btc_bal:.4f})")
            else:
                self.initial_portfolio_value = 0.0
                self.initial_usd = 0.0
                self.initial_btc = 0.0
                logger.warning("Could not calculate initial portfolio value.")
                
        except Exception as e:
            logger.error(f"Failed to sync balances: {e}")
            self.initial_portfolio_value = 0.0
            self.initial_usd = 0.0
            self.initial_btc = 0.0
        
        step = 0
        try:
            for batch_data, batch_covariates in self.dataloader:
                step += 1
                if max_steps > 0 and step > max_steps:
                    logger.info("Max steps reached. Stopping.")
                    break
                
                # Check for model updates every 10 steps to avoid excessive I/O
                if step % 10 == 0:
                    updated, new_mtime = check_for_checkpoint_update(self.checkpoint_path, self.last_ckpt_mtime)
                    if updated:
                        logger.info(f"New checkpoint detected. Reloading model...")
                        try:
                            self.model = self._load_model(self.checkpoint_path)
                            self.last_ckpt_mtime = new_mtime
                            logger.info(f"Model reloaded successfully. Timestamp: {datetime.fromtimestamp(new_mtime)}")
                        except Exception as e:
                            logger.error(f"Failed to reload model: {e}")

                # 1. Inference
                with torch.no_grad():
                    data = batch_data.to(self.device)
                    covariates = batch_covariates.to(self.device)
                    
                    # USE LATEST DATA FOR INFERENCE
                    # Data shape: [batch, seq_len(512), 7]
                    # We want the LAST input_size(256) steps as encoder input
                    x_enc = data[:, -self.input_size:, :] 
                    t_enc = covariates[:, -self.input_size:, :]
                    
                    x_dec = torch.zeros((data.shape[0], self.predict_size, 7)).to(self.device)
                    
                    # Decoder Initialization (Start Token)
                    # Copy last half of encoder to first half of decoder (if sizes allow)
                    label_len = self.input_size // 2
                    x_dec[:, :label_len, :] = x_enc[:, -label_len:, :]
                    
                    # For t_dec (Future Covariates), we MUST extrapolate into the future
                    # Using t_enc.clone() feeds PAST time to FUTURE prediction, confusing the model
                    last_cov = t_enc[0, -1, :] # [4]
                    t_dec = self.generate_future_covariates(last_cov, self.predict_size)
                    
                    # Predict Future
                    pred = self.model(x_enc, t_enc, x_dec, t_dec, pretrain=False)
                    
                    # prediction shape: [1, predict_size, 7]
                    prediction = pred.cpu().numpy()[0]
                    
                    # Denormalize Data (for Strategy & Visualization)
                    # We need real USD values to compare with real Ticker data
                    input_data = x_enc.cpu().numpy()[0]
                    
                    if hasattr(self.dataset, 'scaler'):
                        scaler = self.dataset.scaler
                        if scaler.n > 1:
                            std = np.sqrt(scaler.M2 / (scaler.n - 1))
                            std[std == 0] = 1.0
                            
                            # Denormalize (x * std) + mean
                            input_data = (input_data * std) + scaler.mean
                            prediction = (prediction * std) + scaler.mean
                            
                # 2. Get Current Market Data
                # The "Current" data is the last point in our input window (t-0)
                # Feature 4 is Mid Price
                current_price = input_data[-1, 4].item() # Real USD value now

                # Update Mock Client if in paper trading
                if self.paper_trading and hasattr(self.client, 'set_current_price'):
                    self.client.set_current_price(current_price)

                # Fetch Real-time ticker (Mock or Live)
                ticker = self.client.get_ticker(self.symbol)
                if not ticker:
                    logger.warning("Failed to fetch ticker, skipping step")
                    continue
                    
                current_market_data = {
                    'bid': float(ticker['bid']),
                    'ask': float(ticker['ask']),
                    'last': float(ticker['last'])
                }
                
                # 3. Generate Signal
                signal = self.strategy.generate_signal(current_market_data, prediction)
                
                # 4. Risk Check & Execution
                if signal:
                    logger.info(f"Signal Generated: {signal}")
                    
                    if signal.type != Signal.HOLD:
                        # Update risk manager with current balance
                        balances = self.client.get_balances()
                        # Use updated mock balance
                        current_balance = 0.0
                        for b in balances:
                            if b['currency'] == 'USD': current_balance += float(b['available'])

                        if self.risk_manager.check_trade(signal, signal.price, current_balance):
                            # Execute Order
                            logger.info(f"Executing {signal.type} order...")
                            order = self.client.place_order(
                                signal.symbol, 
                                signal.quantity, 
                                signal.price, 
                                signal.type.lower(), 
                                type="exchange limit"
                            )
                            
                            if order:
                                logger.info(f"Order Placed: {order}")
                                self.risk_manager.update_position(signal.symbol, signal.type, signal.quantity, signal.price)
                            else:
                                logger.error("Order execution failed")
                        else:
                            logger.warning("Trade rejected by Risk Manager")
                
                # --- Performance Monitoring & Visualization ---
                # Update history every step for accuracy
                try:
                    balances = self.client.get_balances()
                    usd_bal = 0
                    btc_bal = 0
                    for b in balances:
                       if b['currency'] == 'USD': usd_bal += float(b['available'])
                       if b['currency'] == 'BTC': btc_bal += float(b['available'])
                    
                    portfolio_value = usd_bal + (btc_bal * current_price)
                    hold_value = self.initial_usd + (self.initial_btc * current_price)
                    
                    self.history['steps'].append(step)
                    self.history['portfolio_values'].append(portfolio_value)
                    self.history['hold_values'].append(hold_value)
                    self.history['prices'].append(current_price)
                    
                    if signal and signal.type in ['BUY', 'SELL']:
                         # If multiple signals per step, this logs the last one, which is fine
                         self.history['signals'].append((step, signal.type, current_price))
                    
                    # Log every 10 steps
                    if step % 10 == 0:
                        # Calculate Profit
                        if self.initial_portfolio_value > 0:
                            profit = portfolio_value - self.initial_portfolio_value
                            profit_pct = (profit / self.initial_portfolio_value) * 100
                            logger.info(f"Step {step} | Portfolio: ${portfolio_value:,.2f} | Profit: ${profit:,.2f} ({profit_pct:+.2f}%) | USD: ${usd_bal:,.2f} BTC: {btc_bal:.4f} | Price: ${current_price:,.2f}")
                        else:
                            logger.info(f"Step {step} | Portfolio: ${portfolio_value:,.2f} | USD: ${usd_bal:,.2f} BTC: {btc_bal:.4f} | Price: ${current_price:,.2f}")
                        
                except Exception as e:
                    logger.warning(f"Failed to update history: {e}")

                # Visualization matches logging freq or every 5 steps
                if self.visualize_mode and step % 5 == 0:
                     # We pass None for targets because we are predicting the future
                     # input_data and prediction are now denormalized (Real USD)
                     # We need to reshape input_data back to [1, input_size, 7] for the visualizer
                     # Or better, update visualize_step to accept [input_size, 7]
                     
                     # Reshape for compatibility with existing visualize_step signature which expects [batch, seq, feat]
                     input_vis = torch.tensor(input_data).unsqueeze(0).to(self.device)
                     
                     self.visualize_step(step, input_vis, None, prediction, signal, portfolio_value, self.initial_portfolio_value)

                if step % 10 == 0:
                    logger.info(f"Step {step} completed. Monitoring...")

        except KeyboardInterrupt:
            logger.info("Trading engine stopped by user")
        except Exception as e:
            logger.critical(f"Trading engine crashed: {e}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Trading Engine")
    parser.add_argument("--symbol", default="btcusd", help="Trading symbol")
    parser.add_argument("--data_dir", default="data", help="Data directory")
    parser.add_argument("--key", default="", help="API Key")
    parser.add_argument("--secret", default="", help="API Secret")
    parser.add_argument("--paper", action="store_true", default=True, help="Run in paper trading mode")
    parser.add_argument("--steps", type=int, default=0, help="Number of steps to run (0 for infinite)")
    parser.add_argument("--mock-data", action="store_true", help="Use mock data generator")
    parser.add_argument("--visualize", action="store_true", help="Enable debug visualization")
    parser.add_argument("--buy-threshold", type=float, default=0.005, help="Buy threshold (default: 0.005)")
    parser.add_argument("--sell-threshold", type=float, default=0.005, help="Sell threshold (default: 0.005)")
    
    args = parser.parse_args()
    
    # Path to latest checkpoint
    #checkpoint_path = os.path.join(args.data_dir, args.symbol, "checkpoints", "latest.pth")
    checkpoint_path = "pyraformer_checkpoint.pth"
    
    if not os.path.exists(checkpoint_path):
        print(f"No model checkpoint found at {checkpoint_path}")
        sys.exit(1)
        
    engine = TradingEngine(
        args.symbol, 
        args.data_dir, 
        checkpoint_path, 
        api_key=args.key, 
        api_secret=args.secret, 
        paper_trading=args.paper,
        mock_data=args.mock_data,
        visualize=args.visualize,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold
    )
    
    engine.run(max_steps=args.steps)
