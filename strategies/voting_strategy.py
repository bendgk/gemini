import numpy as np
from strategies.base_strategy import BaseStrategy, Signal

class VotingStrategy(BaseStrategy):
    def __init__(self, symbol, buy_threshold=0.005, sell_threshold=0.005, quantity=0.001, cooldown_steps=5, confidence_threshold=0.6):
        """
        Args:
            symbol (str): Trading symbol (e.g., 'btcusd').
            buy_threshold (float): Required return to consider a step as "BUYable".
            sell_threshold (float): Required return to consider a step as "SELLable" (short).
            quantity (float): Fixed quantity to trade.
            cooldown_steps (int): Number of steps to wait after a trade before trading again.
            confidence_threshold (float): Fraction of future steps that must agree to trigger a signal (0.0 - 1.0).
        """
        super().__init__(symbol)
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.quantity = quantity
        self.cooldown_steps = cooldown_steps
        self.current_cooldown = 0
        self.confidence_threshold = confidence_threshold

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
        
        predicted_prices = model_prediction[:, 0]
        num_steps = len(predicted_prices)
        
        # Voting Logic
        # Count how many steps in the future offer a profitable exit
        
        # BUY Logic: If we buy at ASK now, how many future steps have predicted BID > ASK * (1 + threshold)?
        # (Using BID for future sale price approximation, although model predicts 'Close' or 'Bid', assuming Feature 0 is price)
        buy_votes = np.sum(predicted_prices > current_ask * (1 + self.buy_threshold))
        
        # SELL Logic: If we sell at BID now, how many future steps have predicted BID < BID * (1 - threshold)?
        # (Betting on price drop to buy back lower)
        sell_votes = np.sum(predicted_prices < current_bid * (1 - self.sell_threshold))
        
        buy_confidence = buy_votes / num_steps
        sell_confidence = sell_votes / num_steps
        
        if buy_confidence >= self.confidence_threshold:
            self.current_cooldown = self.cooldown_steps
            return Signal(Signal.BUY, self.symbol, price=current_ask, quantity=self.quantity)
            
        elif sell_confidence >= self.confidence_threshold:
            self.current_cooldown = self.cooldown_steps
            return Signal(Signal.SELL, self.symbol, price=current_bid, quantity=self.quantity)
            
        return Signal(Signal.HOLD, self.symbol)
