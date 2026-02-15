from sortedcontainers import SortedDict

class OrderBook:
    def __init__(self):
        self.bids = SortedDict()
        self.asks = SortedDict()
        self.spread = None

    def best_bid(self):
        if len(self.bids) > 0:
            return self.bids.peekitem(-1)
        else:
            return None
        
    def best_ask(self):
        if len(self.asks) > 0:
            return self.asks.peekitem(0)
        else:
            return None
        
    def settle_trade(self, makerSide, price, amount):
        if makerSide == "bid":
            if price not in self.bids: raise Exception(f"Trade: {makerSide}, {price}, {amount} not in bids!")

            remaining = self.bids[price] - amount
            if remaining == 0:
                del self.bids[price]
            else:
                self.bids[price] = remaining
        else:
            #makerSide == "ask"
            if price not in self.asks: raise Exception(f"Trade: {makerSide}, {price}, {amount} not in asks!")

            remaining = self.asks[price] - amount
            if remaining == 0:
                del self.asks[price]
            else:
                self.asks[price] = remaining

    def set(self, side, price, amount):
        if side == "bid":
            if amount == 0:
                del self.bids[price]
            else:
                self.bids[price] = amount
        else:
            if amount == 0:
                del self.asks[price]
            else:
                self.asks[price] = amount


        if len(self.bids) == 0 or len(self.asks) == 0:
            self.spread = None

        if len(self.bids) > 0 and len(self.asks) > 0:
            self.spread = self.asks.peekitem(0)[0] - self.bids.peekitem(-1)[0]