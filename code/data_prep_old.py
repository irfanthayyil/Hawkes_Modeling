#!/usr/bin/env python3
import gzip
from datetime import datetime, timedelta
import csv
import math
import os
import logging
import collections
import sys
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np
from sortedcontainers import SortedDict

# -----------------------
# Logging
# -----------------------
logger = logging.getLogger("nse_stream_final")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_handler)

# -----------------------
# CONFIG
# -----------------------
EPOCH = datetime(1980, 1, 1)
JIFFIES_PER_SEC = 65536.0

# -----------------------
# Helpers
# -----------------------
def jiffies_to_dt(jiffies: int) -> str:
    # Pre-format to string to save memory/time in CSV writing
    dt = EPOCH + timedelta(seconds=(jiffies / JIFFIES_PER_SEC))
    return dt.isoformat()

@dataclass(slots=True)
class OrderEvent:
    timestamp: str # pre-formatted ISO string
    order_number: int
    is_buy: bool
    activity_type: int
    price: float
    volume: int
    is_market_order: bool
    is_ioc: bool
    symbol: str

@dataclass(slots=True)
class TradeEvent:
    timestamp: str
    trade_number: int
    price: float
    volume: int
    buy_order_number: int
    sell_order_number: int
    symbol: str

# -----------------------
# LOB Logic
# -----------------------
class SortedLOB:
    def __init__(self):
        self.bids = SortedDict() # key=price, val=vol
        self.asks = SortedDict()
        # map: order_no -> [price, vol, is_buy, is_in_book]
        # We use list instead of tuple because it is mutable (for volume updates)
        self.order_map: Dict[int, List] = {}

    def get_best_bid(self) -> float:
        return self.bids.peekitem(-1)[0] if self.bids else 0.0

    def get_best_ask(self) -> float:
        return self.asks.peekitem(0)[0] if self.asks else 0.0

    def add_order(self, order_no: int, price: float, volume: int, is_buy: bool):
        if volume <= 0:
            return
            
        # CRITICAL LOGIC: Market Orders (Price=0) are tracked for filling, 
        # but NEVER added to the Book (bids/asks dicts).
        if price <= 0:
            # Track as aggressive/hidden
            self.order_map[order_no] = [price, volume, is_buy, False]
            return

        best_ask = self.get_best_ask()
        best_bid = self.get_best_bid()
        
        is_aggressive = False
        if is_buy:
            # Aggressive if crossing the best ask (taking liquidity)
            if best_ask > 0 and price >= best_ask:
                is_aggressive = True
        else:
            # Aggressive if crossing the best bid (taking liquidity)
            if best_bid > 0 and price <= best_bid:
                is_aggressive = True

        if is_aggressive:
            # Track it, but don't add to visible liquidity yet
            self.order_map[order_no] = [price, volume, is_buy, False] 
        else:
            # Passive: add to book
            self.order_map[order_no] = [price, volume, is_buy, True]
            book = self.bids if is_buy else self.asks
            book[price] = book.get(price, 0) + volume

    def remove_order(self, order_no: int):
        if order_no not in self.order_map:
            return
        price, vol, is_buy, is_in_book = self.order_map.pop(order_no)
        if is_in_book:
            book = self.bids if is_buy else self.asks
            if price in book:
                cur = book[price] - vol
                if cur <= 0:
                    del book[price]
                else:
                    book[price] = cur

    def modify_order(self, order_no: int, new_price: float, new_volume: int, is_buy: bool):
        self.remove_order(order_no)
        self.add_order(order_no, new_price, new_volume, is_buy)

    def reduce_for_trade(self, order_no: int, traded_volume: int):
        if order_no not in self.order_map:
            return
        
        order_info = self.order_map[order_no] 
        price = order_info[0]
        vol = order_info[1]
        is_buy = order_info[2]
        is_in_book = order_info[3]
        
        remaining = vol - traded_volume
        
        if remaining <= 0:
            self.remove_order(order_no)
        else:
            # Update volume in the map object
            order_info[1] = remaining
            
            if is_in_book:
                # Reduce liquidity in book
                book = self.bids if is_buy else self.asks
                if price in book:
                    book[price] = book[price] - traded_volume
                    if book[price] <= 0:
                        del book[price]
            else:
                # LOGIC FIX: Aggressive order partially filled.
                # It swept the level. Does the remainder stay hidden or go to book?
                # Correct Logic: Limit orders become passive (Best Bid/Ask).
                # CRITICAL FIX: Market Orders (Price=0) MUST remain hidden/aggressive.
                if price > 0:
                    order_info[3] = True # Set is_in_book = True
                    book = self.bids if is_buy else self.asks
                    book[price] = book.get(price, 0) + remaining

    def get_snapshot(self):
        # Returns best_bid, best_ask, bid_depth, ask_depth
        bb, ba, bd, ad = math.nan, math.nan, 0, 0
        
        if self.bids:
            bb, bd = self.bids.peekitem(-1)
        if self.asks:
            ba, ad = self.asks.peekitem(0)
            
        return float(bb) if not math.isnan(bb) else math.nan, \
               float(ba) if not math.isnan(ba) else math.nan, \
               bd, ad

# -----------------------
# Parsing Iterators
# -----------------------
def iter_orders(path):
    logger.info(f"Opening Orders: {path}")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 80: continue
            try:
                if line[48:50] != "EQ": continue
                
                # Parse specific fields based on fixed width layout
                # Using sys.intern for symbol to save RAM
                sym = sys.intern(line[38:48].strip())
                
                jiffies = int(line[22:36])
                ts = jiffies_to_dt(jiffies)
                
                yield OrderEvent(
                    timestamp=ts,
                    order_number=int(line[6:22]),
                    is_buy=(line[36:37] == 'B'),
                    activity_type=int(line[37:38]),
                    price=int(line[66:74]) / 100.0,
                    volume=int(line[58:66]),
                    is_market_order=(line[82:83] == 'Y'),
                    is_ioc=(line[84:85] == 'Y'),
                    symbol=sym
                )
            except ValueError: continue

def iter_trades(path):
    logger.info(f"Opening Trades: {path}")
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if len(line) < 100: continue
            try:
                if line[46:48] != "EQ": continue
                
                sym = sys.intern(line[36:46].strip())
                jiffies = int(line[22:36])
                ts = jiffies_to_dt(jiffies)

                yield TradeEvent(
                    timestamp=ts,
                    trade_number=int(line[6:22]),
                    price=int(line[48:56]) / 100.0,
                    volume=int(line[56:64]),
                    buy_order_number=int(line[64:80]),
                    sell_order_number=int(line[82:98]),
                    symbol=sym
                )
            except ValueError: continue

def merged_stream(orders, trades):
    """Merges sorted streams of orders and trades by timestamp."""
    o_iter = iter(orders)
    t_iter = iter(trades)
    
    o = next(o_iter, None)
    t = next(t_iter, None)
    
    while o or t:
        if not o:
            yield "TRADE", t
            t = next(t_iter, None)
        elif not t:
            yield "ORDER", o
            o = next(o_iter, None)
        elif o.timestamp <= t.timestamp:
            yield "ORDER", o
            o = next(o_iter, None)
        else:
            yield "TRADE", t
            t = next(t_iter, None)

# -----------------------
# Classification Logic
# -----------------------
def classify_row(row):
    """
    Returns aggressiveness category (1-6) or empty string.
    Based on Biais et al. (1995) and Griffiths et al. (2000).
    """
    # Check if data exists
    if not row['bid'] or not row['ask']: return ""
    
    is_buy = (row['buy'] == '1')
    is_mkt = (row['mkt'] == '1')
    
    # Market orders are always Category 1 (Most Aggressive)
    if is_mkt: return 1

    try:
        price = float(row['price'])
        size = float(row['vol'])
        bid = float(row['bid'])
        ask = float(row['ask'])
        bid_depth = float(row['bd'])
        ask_depth = float(row['ad'])
    except ValueError:
        return ""

    if is_buy:
        if price > ask: return 1
        if price == ask: return 2 if size > ask_depth else 3
        if price > bid and price < ask: return 4
        if price == bid: return 5
        if price < bid: return 6
    else: # Sell
        if price < bid: return 1
        if price == bid: return 2 if size > bid_depth else 3
        if price > bid and price < ask: return 4
        if price == ask: return 5
        if price > ask: return 6
    return ""

# -----------------------
# Main Pipeline
# -----------------------
def run_pipeline(order_path, trade_path, events_out_path, analysis_out_path):
    
    # Map: order_no -> [total_value, total_qty]
    # Using a simple dict is faster but memory intensive. 
    # For 1 day, this should fit in 8GB-16GB RAM.
    fills_map = collections.defaultdict(lambda: [0.0, 0])
    
    lob_map: Dict[str, SortedLOB] = {}
    ord_sym_map: Dict[int, str] = {} 

    logger.info("Pass 1: Streaming Events, Building LOB, Writing Event Stream...")
    
    f_events = open(events_out_path, 'w', newline='')
    event_writer = csv.writer(f_events)
    # Header for the events stream
    event_writer.writerow(["ts","sym","no","buy","price","vol","mkt","ioc","bid","ask","bd","ad"])

    count = 0
    
    for type, ev in merged_stream(iter_orders(order_path), iter_trades(trade_path)):
        count += 1
        if count % 500000 == 0: 
            logger.info(f"Processed {count} events... Unique Fills: {len(fills_map)}")

        if type == "ORDER":
            # Ensure LOB exists for symbol
            if ev.symbol not in lob_map: 
                lob_map[ev.symbol] = SortedLOB()
            lob = lob_map[ev.symbol]
            
            # Activity 1: Entry
            if ev.activity_type == 1:
                # Snapshot LOB state *before* order impact (Pre-order)
                bb, ba, bd, ad = lob.get_snapshot()
                
                # Write to Event Stream (This is the dataset for analysis)
                # Formatting floats to prevent scientific notation or precision noise
                event_writer.writerow([
                    ev.timestamp, ev.symbol, ev.order_number, int(ev.is_buy),
                    f"{ev.price:.2f}", ev.volume, int(ev.is_market_order), int(ev.is_ioc),
                    f"{bb:.2f}" if not math.isnan(bb) else "", 
                    f"{ba:.2f}" if not math.isnan(ba) else "", 
                    bd, ad
                ])
                
                # Update LOB with new order
                lob.add_order(ev.order_number, ev.price, ev.volume, ev.is_buy)
                ord_sym_map[ev.order_number] = ev.symbol

            # Activity 3: Cancel
            elif ev.activity_type == 3:
                lob.remove_order(ev.order_number)
                # Clean up map to save memory (optional, risky if delayed trades arrive, but standard practice)
                if ev.order_number in ord_sym_map:
                    del ord_sym_map[ev.order_number]
            
            # Activity 4: Modify
            elif ev.activity_type == 4:
                lob.modify_order(ev.order_number, ev.price, ev.volume, ev.is_buy)

        elif type == "TRADE":
            # 1. Update Buy Side
            if ev.buy_order_number:
                f = fills_map[ev.buy_order_number]
                f[0] += (ev.price * ev.volume)
                f[1] += ev.volume
                
                # Reduce LOB volume
                sym = ord_sym_map.get(ev.buy_order_number)
                if sym and sym in lob_map: 
                    lob_map[sym].reduce_for_trade(ev.buy_order_number, ev.volume)

            # 2. Update Sell Side
            if ev.sell_order_number:
                f = fills_map[ev.sell_order_number]
                f[0] += (ev.price * ev.volume)
                f[1] += ev.volume
                
                sym = ord_sym_map.get(ev.sell_order_number)
                if sym and sym in lob_map: 
                    lob_map[sym].reduce_for_trade(ev.sell_order_number, ev.volume)

    f_events.close()
    logger.info(f"Pass 1 Complete. Events Saved to: {events_out_path}")
    
    # Clean up LOB maps to free RAM for Pass 2 (we only need fills_map now)
    del lob_map
    del ord_sym_map
    import gc; gc.collect()

    # -----------------------
    # Pass 2: Analysis
    # -----------------------
    logger.info("Pass 2: Calculating Price Impact & Aggressiveness...")
    
    f_final = open(analysis_out_path, 'w', newline='')
    headers = ["timestamp","symbol","order_number","is_buy","aggressiveness","price_impact",
               "size","price","fill_price","midquote","spread","rel_spread",
               "pre_order_bid","pre_order_ask","pre_order_bid_depth","pre_order_ask_depth"]
    
    final_writer = csv.DictWriter(f_final, fieldnames=headers)
    final_writer.writeheader()

    # Re-read the events stream we just created
    with open(events_out_path, 'r') as f_in:
        reader = csv.DictReader(f_in)
        
        for row in reader:
            # Skip orders where LOB wasn't established (Pre-market or sparse data)
            if not row['bid'] or not row['ask']:
                continue

            order_no = int(row['no'])
            
            # Only include orders that were eventually filled (partially or fully)
            # Griffiths et al. Table 3 analyzes "Executed Orders"
            if order_no in fills_map:
                tot_val, tot_qty = fills_map[order_no]
                
                if tot_qty > 0:
                    avg_fill = tot_val / tot_qty
                    
                    try:
                        bid = float(row['bid'])
                        ask = float(row['ask'])
                    except ValueError:
                        continue

                    mid = (bid + ask) / 2.0
                    
                    if mid <= 0: continue

                    # Price Impact = ln(Fill / Mid)

                    impact = math.log(avg_fill / mid)
                    
                    # Classification
                    cat = classify_row(row)
                    
                    final_writer.writerow({
                        "timestamp": row['ts'],
                        "symbol": row['sym'],
                        "order_number": order_no,
                        "is_buy": row['buy'],
                        "aggressiveness": cat,
                        "price_impact": f"{impact:.6f}",
                        "size": row['vol'],
                        "price": row['price'],
                        "fill_price": f"{avg_fill:.4f}",
                        "midquote": f"{mid:.2f}",
                        "spread": f"{(ask - bid):.2f}",
                        "rel_spread": f"{((ask - bid) / mid):.6f}",
                        "pre_order_bid": row['bid'],
                        "pre_order_ask": row['ask'],
                        "pre_order_bid_depth": row['bd'],
                        "pre_order_ask_depth": row['ad']
                    })

    f_final.close()
    logger.info(f"Data Preparation Complete. Final CSV: {analysis_out_path}")

if __name__ == "__main__":
    # order_path = r"C:\Users\Irfan\MM-Empirical_Work\data\CASH_Orders_20082019.DAT.gz"
    # trade_path = r"C:\Users\Irfan\MM-Empirical_Work\data\CASH_Trades_20082019.DAT.gz"
    # events_out_path = "../events_stream.csv"
    # analysis_out_path = "../analysis_final.csv"
    # run_pipeline(order_path, trade_path, events_out_path, analysis_out_path)
    import sys
    if len(sys.argv) < 5:
        print("Usage: python data_prep_final.py <orders.gz> <trades.gz> <events_stream.csv> <analysis_final.csv>")
    else:
        run_pipeline(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])