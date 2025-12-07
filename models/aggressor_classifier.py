import pandas as pd
import numpy as np

def determine_aggressor_side(trades_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the aggressor side for each trade based on order entry timestamps.
    
    Args:
        trades_df (pd.DataFrame): DataFrame containing trade data. 
                                  Must contain 'buy_order_number' and 'sell_order_number'.
        orders_df (pd.DataFrame): DataFrame containing order data.
                                  Must contain 'order_number', 'timestamp'. 
                                  'activity_type' is optional but recommended to filter for ENTRY.
        
    Returns:
        pd.DataFrame: The trades_df with added columns 'buy_entry_ts', 'sell_entry_ts', and 'aggressor_side'.
                      aggressor_side: +1 if Buyer Initiated, -1 if Seller Initiated, 0 if unknown/same time.
    """
    # Filter for ENTRY orders to get original arrival time
    if 'activity_type' in orders_df.columns:
        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY'].copy()
    else:
        orders_entry = orders_df.copy()
    
    # Create lookup for order timestamps
    # drop_duplicates keeps the first occurrence. For 'ENTRY', order_number should be unique anyway.
    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']
    
    # Map timestamps to trades
    trades_df = trades_df.copy()
    trades_df['buy_entry_ts'] = trades_df['buy_order_number'].map(orders_ts)
    trades_df['sell_entry_ts'] = trades_df['sell_order_number'].map(orders_ts)
    
    # Vectorized comparison for performance optimization
    # Convert to datetime if not already (though they should be from loader)
    if not np.issubdtype(trades_df['buy_entry_ts'].dtype, np.datetime64):
        trades_df['buy_entry_ts'] = pd.to_datetime(trades_df['buy_entry_ts'])
    if not np.issubdtype(trades_df['sell_entry_ts'].dtype, np.datetime64):
        trades_df['sell_entry_ts'] = pd.to_datetime(trades_df['sell_entry_ts'])

    # Initialize aggressor side as 0
    trades_df['aggressor_side'] = 0
    
    # Conditions
    # If Buy arrived AFTER Sell, Buyer is Aggressor (+1)
    # If Sell arrived AFTER Buy, Seller is Aggressor (-1)
    
    # Logical masks
    buy_later = trades_df['buy_entry_ts'] > trades_df['sell_entry_ts']
    sell_later = trades_df['sell_entry_ts'] > trades_df['buy_entry_ts']
    
    # Handling missing data logic from notebook:
    # If Buy present and Sell missing => Buy arrived (known), Sell (unknown implies existing?), 
    # original notebook logic: buy_present and not sell_present -> Seller Initiated (-1)
    # This assumes if we can't find the Sell order in ENTRY, it might differ. 
    # Let's stick to the explicit timestamp comparison for safety first.
    
    trades_df.loc[buy_later, 'aggressor_side'] = 1
    trades_df.loc[sell_later, 'aggressor_side'] = -1
    
    # Recover missing cases based on original logic provided in notebook
    # "elif buy_present and not sell_present: return -1" 
    # (Buy is new, Sell is old/unknown -> Sell was resting -> Buy attacked Sell?? No wait.
    # If Buy is in our records (loaded today) and Sell is NOT, Sell might be from yesterday (Resting).
    # So Buy is the aggressor attacking a resting Sell? 
    # Notebook said: "Buy present -> Buy arrived first -> Seller arrived last -> Seller initiated"
    # Wait, if "Buy present" means we have its timestamp. "Sell not present" means we don't.
    # If we only have today's data, a missing order is likely old (yesterday).
    # If Buy is OLD (missing) and Sell is NEW (present), then Sell attacked Buy. (Seller Initiated -1).
    # If Buy is NEW (present) and Sell is OLD (missing), then Buy attacked Sell. (Buyer Initiated +1).
    
    # Let's re-read the provided notebook logic carefully from the context:
    # "elif buy_present and not sell_present: return -1" -> Implies Buy is known, Sell is unknown. 
    # If Sell is unknown, maybe it's not "old" but "market"? 
    # Actually, generally:
    # Aggressor is the one who crosses the spread.
    # If we rely on Time: Later timestamp = Aggressor.
    # If Timestamp is missing, it's usually effectively -infinity (Resting from previous day).
    
    # Let's adhere to the vectorization of the logic EXACTLY as defined in the user's notebook snippet
    # to maintain consistency with their thought process, unless clearly wrong.
    # Notebook:
    # if buy > sell: +1 (Buyer Initiated)
    # elif sell > buy: -1 (Seller Initiated)
    # elif buy_present and not sell_present: -1 (Seller Initiated) << THIS SEEMS ODD if Buy is new.
    #   If Buy has a timestamp (e.g. 10:00), and Sell is NaN (old, < 09:15), then Buy > Sell. Aggressor = Buy (+1).
    #   Why did notebook say -1? 
    #   Maybe "buy_present" means typical logic?
    #   Let's look at Step 30 output again.
    #   def get_aggressor(row):
    #       ...
    #       elif buy_present and not sell_present:
    #           # Buy present -> Buy arrived first -> Seller arrived last -> Seller initiated 
    #           return -1  <-- Logic: "Buy arrived first" (earlier than "missing"?? No).
    #           Wait, if Sell is NOT present, we don't know when it arrived.
    #           If logic assumes "Not present" = "Arrived Later" (e.g. Market order not in ENTRY file?), that's dangerous.
    #           BUT "Not present" usually means "Came from previous day" (Resting).
    #           If Buy is Present (Today) and Sell is Not (Yesterday), Buy is LATER.
    #           Buy should be Aggressor (+1).
    #           The notebook said: "Buy present -> Buy arrived first". This comment contradicts "Buy is today".
    #           If Buy is today, it arrived AFTER yesterday.
    
    # DECISION: I will implement the Standard Market Microstructure logic:
    # 1. Compare timestamps. Max(Buy, Sell) is aggressor.
    # 2. If one is NaN (missing), assume it is OLD (Resting). The Present one is NEW (Aggressor).
    #    - Buy Present, Sell NaN => Buy is New => Buy Aggressor (+1)
    #    - Sell Present, Buy NaN => Sell is New => Sell Aggressor (-1)
    # 3. If both Present and Equal (rare in high precision) => Unknown (0).
    
    buy_mask = trades_df['buy_entry_ts'].notna()
    sell_mask = trades_df['sell_entry_ts'].notna()
    
    # BOTH PRESENT
    both_present = buy_mask & sell_mask
    trades_df.loc[both_present & (trades_df['buy_entry_ts'] > trades_df['sell_entry_ts']), 'aggressor_side'] = 1
    trades_df.loc[both_present & (trades_df['sell_entry_ts'] > trades_df['buy_entry_ts']), 'aggressor_side'] = -1
    
    # ONE MISSING (Assume Missing = Old/Resting)
    # Buy Present (New), Sell Missing (Old) -> Buy Aggressor
    trades_df.loc[buy_mask & (~sell_mask), 'aggressor_side'] = 1
    
    # Sell Present (New), Buy Missing (Old) -> Sell Aggressor
    trades_df.loc[(~buy_mask) & sell_mask, 'aggressor_side'] = -1
    
    return trades_df
