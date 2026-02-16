from abc import ABC, abstractmethod
from datetime import datetime

class Signal:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

    def __init__(self, type, symbol, price=None, quantity=None, timestamp=None):
        self.type = type
        self.symbol = symbol
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp or datetime.now()

    def __repr__(self):
        return f"Signal({self.type}, {self.symbol}, price={self.price}, quantity={self.quantity})"

class BaseStrategy(ABC):
    def __init__(self, symbol):
        self.symbol = symbol

    @abstractmethod
    def generate_signal(self, market_data, model_prediction=None):
        """
        Generate a trading signal based on market data and optional model prediction.
        
        Args:
            market_data (dict): Current market data (e.g., {'bid': ..., 'ask': ...})
            model_prediction (any): Prediction output from the model (optional)
            
        Returns:
            Signal: A Signal object (BUY, SELL, or HOLD)
        """
        pass
