import websocket
import json
import ssl
from orderbook import OrderBook
import struct
import atexit
import time
import os
import logging

db = open(f"data_{int(time.time())}.bin", "ab+")
message_count = 0
order_book = OrderBook()

last_socket_sequence = None

def on_message(ws: websocket.WebSocketApp, message):  
    global message_count
    global db
    global order_book
    global last_socket_sequence

    message = json.loads(message)

    if "events" in message and "socket_sequence" in message:
        if "timestampms" in message:
            timestamp_ms = int(message["timestampms"])
        else:
            timestamp_ms = 0

        socket_sequence = int(message["socket_sequence"])

        if last_socket_sequence is not None and socket_sequence != last_socket_sequence + 1:
            print(f"Socket sequence mismatch: expected {last_socket_sequence + 1}, got {socket_sequence}")

        last_socket_sequence = socket_sequence

        for event in message["events"]:

            event_type = event["type"]
            price = float(event["price"])

            match event_type:
                case "trade":
                    amount = float(event["amount"])
                    maker_side = event["makerSide"]
                    order_book.settle_trade(maker_side, price, amount)

                    side_byte = 0 if maker_side == "bid" else 1

                    #[timestamp_ms, 0 for trade, 1 for change, reason, side, price, amount]
                    db.write(struct.pack("Qbbbff", timestamp_ms, 0, 0, side_byte, price, amount))
                    

                case "change":
                    side = event["side"]    # "bid" or "ask"
                    reason = event["reason"]    # "trade" (0), "place" (1), "cancel" (2), "initial (3)"
                    remaining = float(event["remaining"])
                    delta = float(event["delta"])

                    side_byte = 0 if side == "bid" else 1

                    if reason == "initial":
                        order_book.set(side, price, delta)
                        db.write(struct.pack("Qbbbff", timestamp_ms, 1, 3, side_byte, price, delta))

                    elif reason != "trade": # trades are already handled above
                        if (reason == "cancel") and (price not in order_book.asks and price not in order_book.bids):
                            #someone tried to cancel order that doesn't exist in our books!
                            return

                        if reason == "place" and (remaining == 0.0 or delta <= 0.0):
                            #someone tried to place a funny looking order!
                            return

                        order_book.set(side, price, remaining)

                        reason_byte = None
                        #cant be 0 or 3 since those are handled above
                        match reason:
                            case "place":
                                reason_byte = 1
                            case "cancel":
                                reason_byte = 2
                            case _:
                                pass

                        if reason_byte is None:
                            raise Exception(f"Unknown reason: {reason}")

                        db.write(struct.pack("Qbbbff", timestamp_ms, 1, reason_byte, side_byte, price, remaining))

                case _:
                    pass
    
        message_count += 1
        if message_count % 100 == 0:
            db.flush()
            os.fsync(db.fileno())

def on_exit():
    db.close()

def on_open(ws: websocket.WebSocketApp):
    print("### opened ###")

def on_close(ws: websocket.WebSocketApp, close_status_code, close_msg):
    print("### closed ###")

def on_error(ws, error: Exception):
    print("### error ###")
    logging.exception(error)

if __name__ == "__main__":
    atexit.register(on_exit)

    ws = websocket.WebSocketApp(
        "wss://api.gemini.com/v1/marketdata/BTCUSD",
        on_message=on_message,
        on_close=on_close,
        on_open=on_open,
        on_error=on_error
    )
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})