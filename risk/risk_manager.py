import logging

logger = logging.getLogger('RiskManager')

class RiskManager:
    def __init__(self, max_position_size=1.0, max_drawdown=0.05, max_daily_loss=100.0):
        """
        Initialize the Risk Manager.
        
        Args:
            max_position_size (float): Maximum position size (in base currency, e.g., BTC).
            max_drawdown (float): Maximum allowed drawdown (percentage).
            max_daily_loss (float): Maximum allowed daily loss (in quote currency, e.g., USD).
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        
        self.current_positions = {} # symbol -> quantity
        self.daily_pnl = 0.0
        self.start_balance = 0.0 
        
    def set_start_balance(self, balance):
        self.start_balance = balance

    def sync_positions(self, balances):
        """
        Sync internal position tracking with exchange balances.
        
        Args:
            balances (list): List of balance dicts from Gemini API. 
                             e.g., [{'currency': 'BTC', 'available': '10.0', ...}, ...]
        """
        for bal in balances:
            currency = bal['currency'].upper()
            available = float(bal['available'])
            # Assuming symbol format "BTCUSD" -> currency "BTC"
            # This is a simplification. Ideally we map currency to symbols.
            # For now, we just track currency amounts if we use them for checks.
            # But our current_positions dict uses SYMBOL keys (e.g. 'btcusd').
            # We need a mapping. 
            pass 
            
        # For simplicity in this iteration, we trusting the execution engine to keep us in check
        # But we really should update self.current_positions.
        # Let's assume we are only trading BTCUSD for now.
        for bal in balances:
            if bal['currency'].upper() == 'BTC':
                 self.current_positions['btcusd'] = float(bal['available'])
                 logger.info(f"Risk Manager synced: btcusd position = {self.current_positions['btcusd']}")

    def check_trade(self, signal, current_price, current_balance):
        """
        Check if a trade is allowed based on risk parameters.
        
        Args:
            signal (Signal): The trading signal.
            current_price (float): Current price of the asset.
            current_balance (float): Current account balance.
            
        Returns:
            bool: True if trade is allowed, False otherwise.
        """
        
        # 1. Check Max Daily Loss
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning(f"Risk Check Failed: Max daily loss exceeded ({self.daily_pnl} < -{self.max_daily_loss})")
            return False
            
        # 2. Check position limits
        current_pos = self.current_positions.get(signal.symbol, 0.0)
        
        if signal.type == "BUY":
            # Check if adding to position exceeds max size
            new_pos_size = current_pos + signal.quantity
            if new_pos_size > self.max_position_size:
                logger.warning(f"Risk Check Failed: Max position size exceeded ({new_pos_size} > {self.max_position_size})")
                return False
                
            # Check if we have enough balance (simplified)
            cost = signal.quantity * signal.price
            if cost > current_balance:
                 logger.warning(f"Risk Check Failed: Insufficient balance ({current_balance} < {cost})")
                 return False

        elif signal.type == "SELL":
             # Check if we are selling more than we have (if shorting is not allowed)
             # Assuming no shorting for now
             if current_pos < signal.quantity:
                 logger.warning(f"Risk Check Failed: Insufficient position to sell ({current_pos} < {signal.quantity})")
                 return False
                 
        return True

    def update_position(self, symbol, side, quantity, price):
        """Update position tracking after a trade execution."""
        if side == "BUY":
            self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) + quantity
        elif side == "SELL":
            self.current_positions[symbol] = self.current_positions.get(symbol, 0.0) - quantity
            
    def update_pnl(self, realized_pnl):
        self.daily_pnl += realized_pnl
