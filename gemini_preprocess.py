import os
import numpy as np
import math
import random
from tqdm import trange
from scipy import stats
import struct
from orderbook import OrderBook
import time
import torch


class GeminiDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            self.data = f.read()

        #each transaction is 12 bytes unpacked in the struct format (bbbff)
        # read 12 bytes at a time
        size = struct.calcsize("Qbbbff")
        self.num_transactions = len(self.data) // size
        self.trades = []

        for i in range(self.num_transactions):
            # unpack the data
            timestamp_ms, is_change, reason, side, price, amount = struct.unpack("Qbbbff", self.data[i*size:(i+1)*size])

            print(timestamp_ms, is_change, reason, side, price, amount)
            input()

            side = "bid" if side == 0 else "ask"

            #handle order book changes
            if is_change:
                order_book.set(side, price, amount)

            else:
                #trade
                order_book.settle_trade(side, price, amount)

                bids = order_book.bids.items()[::-1][0:32].copy()
                asks = order_book.asks.items()[0:32].copy()

                bid_prices = [b[0] for b in bids]
                bid_amounts = [b[1] for b in bids]
                ask_prices = [a[0] for a in asks]
                ask_amounts = [a[1] for a in asks]

                self.trades.append((timestamp_ms, price, amount, bid_prices, bid_amounts, ask_prices, ask_amounts))
        
        self.num_trades = len(self.trades)

    def __len__(self):
        return self.num_trades
    
    def __getitem__(self, idx):
        trade = self.trades[idx]
        price, amount, bid_prices, bid_amounts, ask_prices, ask_amounts = trade

        # Convert to numpy arrays
        bid_prices = np.array(bid_prices)
        bid_amounts = np.array(bid_amounts)
        ask_prices = np.array(ask_prices)
        ask_amounts = np.array(ask_amounts)

        # Normalize prices and amounts
        bid_prices = (bid_prices - np.mean(bid_prices)) / np.std(bid_prices)
        bid_amounts = (bid_amounts - np.mean(bid_amounts)) / np.std(bid_amounts)
        ask_prices = (ask_prices - np.mean(ask_prices)) / np.std(ask_prices)
        ask_amounts = (ask_amounts - np.mean(ask_amounts)) / np.std(ask_amounts)

        return price, amount, bid_prices, bid_amounts, ask_prices, ask_amounts

order_book = OrderBook()

def prep_data(data, covariates, data_start, Train=True):
    pass

def gen_covariates(order_book: OrderBook):
    bids = order_book.bids
    asks = order_book.asks

    covariates = np.zeros()
    pass

if __name__ == '__main__':
    time_start = time.time()

    dataset = GeminiDataset("data/data.bin")
    print(len(dataset.trades))

    print(f"Time taken: {time.time() - time_start} seconds")


        