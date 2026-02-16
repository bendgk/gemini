import logging
import time
import hmac
import hashlib
import base64
import json
import requests

logger = logging.getLogger('GeminiClient')

class GeminiClient:
    def __init__(self, api_key, api_secret, sandbox=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.sandbox.gemini.com" if sandbox else "https://api.gemini.com"
        self.sandbox = sandbox

    def _get_nonce(self):
        return int(time.time() * 1000)

    def _sign_payload(self, payload):
        b64_payload = base64.b64encode(json.dumps(payload).encode())
        signature = hmac.new(
            self.api_secret.encode(),
            b64_payload,
            hashlib.sha384
        ).hexdigest()
        return b64_payload, signature

    def _request(self, endpoint, payload=None):
        if payload is None:
            payload = {}
        
        payload["request"] = endpoint
        payload["nonce"] = self._get_nonce()
        
        b64_payload, signature = self._sign_payload(payload)
        
        headers = {
            'Content-Type': "text/plain",
            'Content-Length': "0",
            'X-GEMINI-APIKEY': self.api_key,
            'X-GEMINI-PAYLOAD': b64_payload,
            'X-GEMINI-SIGNATURE': signature,
            'Cache-Control': "no-cache"
        }
        
        url = self.base_url + endpoint
        try:
            response = requests.post(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None

    def get_ticker(self, symbol):
        endpoint = f"/v1/pubticker/{symbol}"
        try:
            response = requests.get(self.base_url + endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    def place_order(self, symbol, amount, price, side, type="exchange limit"):
        endpoint = "/v1/order/new"
        payload = {
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "side": side,
            "type": type
        }
        return self._request(endpoint, payload)

    def cancel_order(self, order_id):
        endpoint = "/v1/order/cancel"
        payload = {"order_id": order_id}
        return self._request(endpoint, payload)

    def get_active_orders(self):
        endpoint = "/v1/orders"
        return self._request(endpoint)

    def get_balances(self):
        endpoint = "/v1/balances"
        return self._request(endpoint)

class MockGeminiClient:
    """Mock client for paper trading / testing."""
    def __init__(self, balances=None):
        self.balances = balances if balances else {"USD": 100000.0, "BTC": 10.0}
        self.orders = []
        self.current_price = 90000.0 # Default fallback

    def set_current_price(self, price):
        self.current_price = float(price)

    def get_ticker(self, symbol):
        # Return ticker based on current set price
        return {
            "bid": str(self.current_price),
            "ask": str(self.current_price + 1.0), # Spread
            "last": str(self.current_price)
        }

    def place_order(self, symbol, amount, price, side, type="exchange limit"):
        order_id = str(int(time.time() * 1000))
        
        # Simple Mock Execution Logic
        amount = float(amount)
        price = float(price)
        cost = amount * price
        
        # Check sufficient funds
        if side == "buy":
            if self.balances["USD"] >= cost:
                self.balances["USD"] -= cost
                self.balances["BTC"] += amount
                logger.info(f"MOCK EXECUTION: Bought {amount} BTC for {cost} USD")
            else:
                logger.warning(f"MOCK EXECUTION FAILED: Insufficient USD ({self.balances['USD']} < {cost})")
                return None
        elif side == "sell":
             if self.balances["BTC"] >= amount:
                 self.balances["BTC"] -= amount
                 self.balances["USD"] += cost
                 logger.info(f"MOCK EXECUTION: Sold {amount} BTC for {cost} USD")
             else:
                 logger.warning(f"MOCK EXECUTION FAILED: Insufficient BTC ({self.balances['BTC']} < {amount})")
                 return None
                 
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "side": side,
            "type": type,
            "is_live": False
        }
        self.orders.append(order)
        logger.info(f"MOCK ORDER PLACED: {side} {amount} {symbol} @ {price}")
        return order

    def cancel_order(self, order_id):
        self.orders = [o for o in self.orders if o["order_id"] != order_id]
        logger.info(f"MOCK ORDER CANCELLED: {order_id}")
        return {"order_id": order_id}

    def get_active_orders(self):
        return self.orders

    def get_balances(self):
        # Return in Gemini format
        return [
            {"currency": "BTC", "amount": str(self.balances["BTC"]), "available": str(self.balances["BTC"])},
            {"currency": "USD", "amount": str(self.balances["USD"]), "available": str(self.balances["USD"])}
        ]
