import pandas as pd
import numpy as np
import os

# Paths
orders_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\orders\bbbbbbINFY_orders_19082019.csv'
trades_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\trades\bbbbbbINFY_trades_19082019.csv'

# Load data - limit rows for speed if needed, but 100k is fine
print("Loading data...")
orders_df = pd.read_csv(orders_path)
trades_df = pd.read_csv(trades_path)

# Ensure timestamps are datetime
orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])

print(f"Orders: {len(orders_df)}, Trades: {len(trades_df)}")

# Original Function
def determine_aggressor_side_original(trades_df, orders_df):
    if 'activity_type' in orders_df.columns:
        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY'].copy()
    else:
        orders_entry = orders_df.copy()
    
    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']
    
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
                return 1 
            elif sell_ts > buy_ts:
                return -1 
            else:
                return 0 
        elif buy_present and not sell_present:
            # Original Logic: Buy present -> Buy arrived first -> Seller arrived last -> Seller initiated (-1)
            return -1 
        elif not buy_present and sell_present:
            # Original Logic: Sell present -> Sell arrived first -> Buyer arrived last -> Buyer initiated (1)
            return 1
        else:
            return 0 
            
    trades_df['aggressor_side'] = trades_df.apply(get_aggressor, axis=1)
    return trades_df

# Proposed Vectorized & Fixed Function
def determine_aggressor_side_fixed(trades_df, orders_df):
    if 'activity_type' in orders_df.columns:
        orders_entry = orders_df[orders_df['activity_type'] == 'ENTRY']
    else:
        orders_entry = orders_df
    
    # Use drop_duplicates to keep first entry
    orders_ts = orders_entry.drop_duplicates('order_number').set_index('order_number')['timestamp']
    
    trades_df = trades_df.copy()
    trades_df['buy_entry_ts'] = trades_df['buy_order_number'].map(orders_ts)
    trades_df['sell_entry_ts'] = trades_df['sell_order_number'].map(orders_ts)
    
    buy_ts = trades_df['buy_entry_ts']
    sell_ts = trades_df['sell_entry_ts']
    
    # Initialize with 0
    aggressor = np.zeros(len(trades_df), dtype=int)
    
    # Masks
    buy_exists = pd.notna(buy_ts)
    sell_exists = pd.notna(sell_ts)
    
    # Case 1: Both exist
    both_exist = buy_exists & sell_exists
    aggressor[both_exist & (buy_ts > sell_ts)] = 1  # Buy is later (Aggressor)
    aggressor[both_exist & (sell_ts > buy_ts)] = -1 # Sell is later (Aggressor)
    
    # Case 2: Only Buy exists (Sell is missing -> Sell is older)
    # Buy is Newer -> Buy is Aggressor
    aggressor[buy_exists & ~sell_exists] = 1
    
    # Case 3: Only Sell exists (Buy is missing -> Buy is older)
    # Sell is Newer -> Sell is Aggressor
    aggressor[~buy_exists & sell_exists] = -1
    
    trades_df['aggressor_side'] = aggressor
    return trades_df

# Check Order IDs relation for missing cases
def check_missing_ids(trades_df, orders_df):
    # Only check 'ENTRY' orders
    entry_ids = set(orders_df[orders_df['activity_type'] == 'ENTRY']['order_number'])
    
    missing_sell = trades_df[~trades_df['sell_order_number'].isin(entry_ids)].copy()
    missing_buy = trades_df[~trades_df['buy_order_number'].isin(entry_ids)].copy()
    
    print(f"Trades with missing Sell Entry: {len(missing_sell)}")
    print(f"Trades with missing Buy Entry: {len(missing_buy)}")
    
    if len(missing_sell) > 0:
        # Check if missing sell ID is generally smaller than buy ID (implying older)
        # We assume Buy IS present in these cases (or at least check the ones where Buy IS present)
        missing_sell_buy_present = missing_sell[missing_sell['buy_order_number'].isin(entry_ids)]
        if len(missing_sell_buy_present) > 0:
            avg_diff = (missing_sell_buy_present['buy_order_number'] - missing_sell_buy_present['sell_order_number']).mean()
            print(f"Avg (Buy ID - Sell ID) when Sell missing: {avg_diff:.2f} (Positive implies Buy > Sell, Sell is older)")
            
    if len(missing_buy) > 0:
        missing_buy_sell_present = missing_buy[missing_buy['sell_order_number'].isin(entry_ids)]
        if len(missing_buy_sell_present) > 0:
            avg_diff = (missing_buy_sell_present['sell_order_number'] - missing_buy_sell_present['buy_order_number']).mean()
            print(f"Avg (Sell ID - Buy ID) when Buy missing: {avg_diff:.2f} (Positive implies Sell > Buy, Buy is older)")

# Run Tests
print("\nRunning Logic Checks...")
df_orig = determine_aggressor_side_original(trades_df.head(10000), orders_df)
df_fixed = determine_aggressor_side_fixed(trades_df.head(10000), orders_df)

print("\nComparison on first 10000 trades:")
print("Original Counts:\n", df_orig['aggressor_side'].value_counts())
print("Fixed Counts:\n", df_fixed['aggressor_side'].value_counts())

diff = df_orig['aggressor_side'] != df_fixed['aggressor_side']
print(f"\nDiffering rows: {sum(diff)}")
if sum(diff) > 0:
    sample_diff = df_fixed[diff].head()
    print("\nSample Diff (Fixed):")
    cols = ['buy_entry_ts', 'sell_entry_ts', 'aggressor_side']
    print(sample_diff[cols])
    print("\nOriginal for same rows:")
    print(df_orig.loc[sample_diff.index, cols])
    
print("\nChecking Missing ID Hypothesis...")
check_missing_ids(trades_df, orders_df)
