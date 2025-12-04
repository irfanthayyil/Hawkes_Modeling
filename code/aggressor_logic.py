import pandas as pd
import numpy as np

def determine_aggressor_side(trades_df, orders_df):
    """
    Determines the aggressor side for each trade based on order entry timestamps.
    
    Args:
        trades_df (pd.DataFrame): DataFrame containing trade data with 'buy_order_number' and 'sell_order_number'.
        orders_df (pd.DataFrame): DataFrame containing order data with 'order_number', 'timestamp', and 'activity_type'.
        
    Returns:
        pd.DataFrame: The trades_df with added columns 'buy_entry_ts', 'sell_entry_ts', and 'aggressor_side'.
                      aggressor_side: +1 if Buyer Initiated, -1 if Seller Initiated, 0 if unknown/same time.
    """
    # Filter for ENTRY orders to get original arrival time
    # We assume 'activity_type' column exists and 'ENTRY' denotes the initial order
    if 'activity_type' in orders_df.columns:
        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY'].copy()
    else:
        # Fallback if activity_type is missing, though it should be there based on NseDataLoader
        orders_entry = orders_df.copy()
    
    # Create lookup for order timestamps
    # taking the first entry if duplicates exist (though order_number should be unique for ENTRY)
    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']
    
    # Map timestamps to trades
    trades_df = trades_df.copy()
    trades_df['buy_entry_ts'] = trades_df['buy_order_number'].map(orders_ts)
    trades_df['sell_entry_ts'] = trades_df['sell_order_number'].map(orders_ts)
    
    def get_aggressor(row):
        buy_ts = row['buy_entry_ts']
        sell_ts = row['sell_entry_ts']
        
        buy_present = pd.notna(buy_ts)
        sell_present = pd.notna(sell_ts)
        
        if buy_present and sell_present:
            if buy_ts > sell_ts:
                return 1 # Buyer initiated (arrived later)
            elif sell_ts > buy_ts:
                return -1 # Seller initiated (arrived later)
            else:
                return 0 # Same time
        elif buy_present and not sell_present:
            # Buy present -> Buy arrived first -> Seller arrived last -> Seller initiated
            return -1
        elif not buy_present and sell_present:
            # Sell present -> Sell arrived first -> Buyer arrived last -> Buyer initiated
            return 1
        else:
            return 0 # Both missing
            
    trades_df['aggressor_side'] = trades_df.apply(get_aggressor, axis=1)
    return trades_df
