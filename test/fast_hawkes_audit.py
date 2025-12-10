
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from models.aggressor_classifier import determine_aggressor_side
from models.hawkes_model import MultivariateHawkes

def run_fast_audit():
    # Load paths
    orders_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\orders\bbbbbbINFY_orders_19082019.csv'
    trades_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\trades\bbbbbbINFY_trades_19082019.csv'
    
    if not os.path.exists(orders_path):
        # Fallback to the one seen in file list if 19th doesnt exist
        orders_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\orders\bbbbbbINFY_orders_13082019.csv'
        trades_path = r'c:\Users\Irfan\Hawkes_Modeling\data\outputs\trades\bbbbbbINFY_trades_13082019.csv'
        
    print(f"Loading data from {orders_path}...")
    # Load limited rows
    orders_df = pd.read_csv(orders_path)
    trades_df = pd.read_csv(trades_path) # Load all trades to filter properly, or top 5000 but might mismatch order IDs.
    
    # Let's take head, but robustly
    trades_df = trades_df.head(5000)
    
    # Ensure timestamps
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    
    print("Determining Aggressor...")
    trades_df = determine_aggressor_side(trades_df, orders_df)
    
    # CRITICAL CHECK FOR AUDIT: How many Unknowns (0)?
    unknowns = len(trades_df[trades_df['aggressor_side'] == 0])
    total = len(trades_df)
    print(f"AUDIT FINDING: {unknowns}/{total} trades have Unknown Aggressor (0).")
    
    # Filter for model
    buyer_trades = trades_df[trades_df['aggressor_side'] == 1]['timestamp']
    seller_trades = trades_df[trades_df['aggressor_side'] == -1]['timestamp']
    
    print(f"Modeling {len(buyer_trades)} Buys and {len(seller_trades)} Sells.")
    
    if len(buyer_trades) < 10 or len(seller_trades) < 10:
        print("Not enough events.")
        return

    t_min = trades_df['timestamp'].min()
    buy_ts = (buyer_trades - t_min).dt.total_seconds().values
    sell_ts = (seller_trades - t_min).dt.total_seconds().values
    buy_ts.sort()
    sell_ts.sort()
    
    print("Fitting Hawkes (max 2000 events for speed)...")
    # Truncate for speed if still too big
    limit = 2000
    events = [buy_ts[:limit], sell_ts[:limit]]
    
    hawkes = MultivariateHawkes(n_nodes=2)
    hawkes.fit(events)
    
    mus, alphas, betas = hawkes.get_parameters()
    ratios = hawkes.get_branching_ratio()
    
    print("\n--- RESULTS ---")
    print("Branching Ratios (n_ij = alpha/beta):")
    print(ratios)
    
    # Total branching ratio (spectral radius approximation)
    print(f"Total Branching (Spectral Radius): {np.max(np.abs(np.linalg.eigvals(ratios))):.4f}")

if __name__ == "__main__":
    run_fast_audit()
