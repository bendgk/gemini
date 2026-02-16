
import logging
from execution.gemini_client import MockGeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def verify_portfolio_logic():
    logger.info("--- Verifying Portfolio Logic ---")
    
    # 1. Initialize Client
    initial_usd = 100000.0
    initial_btc = 10.0
    client = MockGeminiClient(balances={"USD": initial_usd, "BTC": initial_btc})
    
    logger.info(f"Initial Balance: USD={initial_usd}, BTC={initial_btc}")
    
    # 2. Simulate Price
    price = 50000.0
    client.set_current_price(price)
    logger.info(f"Set Price: {price}")
    
    # 3. Execute BUY (Buy 1 BTC)
    quantity = 1.0
    cost = quantity * price
    logger.info(f"\nExecuting BUY: {quantity} BTC @ ${price} (Cost should be ${cost})")
    
    client.place_order("btcusd", quantity, price, "buy")
    
    # Verify New Balance
    expected_usd = initial_usd - cost
    expected_btc = initial_btc + quantity
    
    logger.info(f"New Balance: USD={client.balances['USD']}, BTC={client.balances['BTC']}")
    
    if client.balances['USD'] == expected_usd and client.balances['BTC'] == expected_btc:
        logger.info("✅ BUY Logic Correct")
    else:
        logger.error(f"❌ BUY Logic Failed! Expected USD={expected_usd}, BTC={expected_btc}")

    # 4. Calculate Portfolio Value
    portfolio_value = client.balances['USD'] + (client.balances['BTC'] * price)
    expected_value = expected_usd + (expected_btc * price)
    # Note: Value should remain constant relative to initial if we ignore spread/fees, 
    # because we exchanged USD for BTC at fair value.
    # Initial Value = 100k + 10*50k = 600k
    # New Value = 50k + 11*50k = 600k 
    
    logger.info(f"Portfolio Value: ${portfolio_value}")
    if portfolio_value == 600000.0:
        logger.info("✅ Portfolio Value Logic Correct")
    else:
        logger.error(f"❌ Portfolio Value Logic Failed! Expected 600,000, got {portfolio_value}")
        
    # 5. Simulate Price Change (BTC goes up to 60k)
    new_price = 60000.0
    client.set_current_price(new_price)
    logger.info(f"\nPrice moves to ${new_price}")
    
    # Value should increase by (Holdings * Price Increase)
    # Holdings = 11 BTC. Increase = 10k. Value gain = 110k.
    # New Value should be 710k.
    
    new_portfolio_value = client.balances['USD'] + (client.balances['BTC'] * new_price)
    logger.info(f"New Portfolio Value: ${new_portfolio_value}")
    
    if new_portfolio_value == 710000.0:
        logger.info("✅ Price Update Logic Correct")
    else:
        logger.error(f"❌ Price Update Logic Failed! Expected 710,000, got {new_portfolio_value}")

    # 6. Execute SELL (Sell 0.5 BTC)
    sell_qty = 0.5
    proceeds = sell_qty * new_price # 0.5 * 60k = 30k
    logger.info(f"\nExecuting SELL: {sell_qty} BTC @ ${new_price} (Proceeds should be ${proceeds})")
    
    client.place_order("btcusd", sell_qty, new_price, "sell")
    
    logger.info(f"New Balance: USD={client.balances['USD']}, BTC={client.balances['BTC']}")
    
    expected_usd_2 = expected_usd + proceeds # 50k + 30k = 80k
    expected_btc_2 = expected_btc - sell_qty # 11 - 0.5 = 10.5
    
    if client.balances['USD'] == expected_usd_2 and client.balances['BTC'] == expected_btc_2:
        logger.info("✅ SELL Logic Correct")
    else:
        logger.error(f"❌ SELL Logic Failed! Expected USD={expected_usd_2}, BTC={expected_btc_2}")

if __name__ == "__main__":
    verify_portfolio_logic()
