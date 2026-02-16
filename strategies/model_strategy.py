import numpy as np
from strategies.base_strategy import BaseStrategy, Signal

class ModelStrategy(BaseStrategy):
    def __init__(self, symbol, buy_threshold=0.005, sell_threshold=0.005, quantity=0.001, cooldown_steps=5):
        """
        Args:
            symbol (str): Trading symbol (e.g., 'btcusd').
            buy_threshold (float): Expected return threshold to trigger BUY.
            sell_threshold (float): Expected return threshold to trigger SELL.
            quantity (float): Fixed quantity to trade.
            cooldown_steps (int): Number of steps to wait after a trade before trading again.
        """
        super().__init__(symbol)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.quantity = quantity
        self.cooldown_steps = cooldown_steps
        self.current_cooldown = 0

    def generate_signal(self, market_data, model_prediction=None):
        # Cooldown Check
        if self.current_cooldown > 0:
            self.current_cooldown -= 1
            return None
            
        if model_prediction is None:
            return None

        # market_data: {'bid': float, 'ask': float}
        # model_prediction: numpy array of shape [predict_len, features]
        # target feature 0 is Bid Price
        
        current_bid = market_data['bid']
        current_ask = market_data['ask']
        
        # Simple Logic: 
        # 1. Take average of predicted bid prices
        # 2. Compare with current ask (for buying) or current bid (for selling)
        
        predicted_bids = model_prediction[:, 0]
        avg_predicted_bid = np.mean(predicted_bids)
        
        # Calculate expected return
        # If we buy now at Ask, will the future Bid be higher?
        expected_return_buy = (avg_predicted_bid - current_ask) / current_ask
        
        # If we sell now at Bid, will the future Bid be lower? (for buying back lower)
        # Or simpler: if predicted price is lower than current bid, sell.
        expected_return_sell = (current_bid - avg_predicted_bid) / current_bid
        
        if expected_return_buy > self.buy_threshold:
            self.current_cooldown = self.cooldown_steps
            return Signal(Signal.BUY, self.symbol, price=current_ask, quantity=self.quantity)
            
        elif expected_return_sell > self.sell_threshold:
            self.current_cooldown = self.cooldown_steps
            return Signal(Signal.SELL, self.symbol, price=current_bid, quantity=self.quantity)
            
        return Signal(Signal.HOLD, self.symbol)
