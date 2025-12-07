
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_loader.NseDataLoaderOptimized import NseDataLoaderOptimized
from models.aggressor_classifier import determine_aggressor_side
from models.hawkes_model import MultivariateHawkes

def run_analysis(order_path, trade_path, output_dir='data/outputs'):
    
    loader = NseDataLoaderOptimized()
    
    print("Loading data...")
    # Load orders and trades
    # We might want to use the existing saved CSVs if available to save time
    # Check if we can find them
    
    # For now, let's assume we load from the large files or the specific daily files provided in task
    # To demonstrate, I will use the paths I saw in the file exploration
    
    # Assuming paths are passed or we find them
    if not os.path.exists(order_path) or not os.path.exists(trade_path):
        print("Data files not found.")
        return

    # Load data
    orders_df, trades_df = loader.load_symbol_for_day(order_path, trade_path, 'INFY')
    
    print(f"Loaded {len(orders_df)} orders and {len(trades_df)} trades.")
    
    # Aggressor Determination
    print("Determining aggressor side...")
    trades_df = determine_aggressor_side(trades_df, orders_df)
    
    # Filter valid aggressors
    buyer_trades = trades_df[trades_df['aggressor_side'] == 1]['timestamp']
    seller_trades = trades_df[trades_df['aggressor_side'] == -1]['timestamp']
    
    print(f"Buyer Initiated Events: {len(buyer_trades)}")
    print(f"Seller Initiated Events: {len(seller_trades)}")
    
    # Prepare for Hawkes
    # Convert to float seconds from start
    t_min = trades_df['timestamp'].min()
    
    buy_ts = (buyer_trades - t_min).dt.total_seconds().values
    sell_ts = (seller_trades - t_min).dt.total_seconds().values
    
    # Sort
    buy_ts.sort()
    sell_ts.sort()
    
    events = [buy_ts, sell_ts] # Node 0: Buy, Node 1: Sell
    
    # Fit Model
    print("Fitting Multivariate Hawkes Process (2 nodes: Buy, Sell)...")
    hawkes = MultivariateHawkes(n_nodes=2)
    hawkes.fit(events)
    
    # Extract results
    mus, alphas, betas = hawkes.get_parameters()
    branching_ratio = hawkes.get_branching_ratio()
    
    print("\n--- Model Results ---")
    nodes = ['Buy', 'Sell']
    
    for i in range(2):
        print(f"\nNode {nodes[i]} (Target):")
        print(f"  Baseline Intensity (mu): {mus[i]:.4f}")
        for j in range(2):
            source = nodes[j]
            print(f"  From {source}: Alpha={alphas[i,j]:.4f}, Beta={betas[i,j]:.4f}, Branching={branching_ratio[i,j]:.4f}")
            
    # Forecasting
    # Short term forecast (1 minute)
    # Expected number of events in next delta_t given history H_t
    # E[N(t+dt) - N(t) | H_t] = integral_t^{t+dt} lambda(u) du
    # If we approximate lambda constant for short dt? No, it decays.
    # We can simulate or integrate decay analytically.
    
    T_end = (trades_df['timestamp'].max() - t_min).total_seconds()
    forecast_horizon = 60 # 1 minute
    
    print(f"\nForecasting next {forecast_horizon} seconds...")
    pred_counts = []
    
    for i in range(2):
        # Calculate current intensity at T_end
        # lambda_i(T_end) = mu_i + sum_j sum_k alpha * exp(-beta * (T_end - t_jk))
        
        # Integrated intensity over [T_end, T_end + horizon]
        # Int(mu) = mu * horizon
        # Int(kernels) = sum_j sum_k (alpha/beta) * (exp(-beta*(T_end - t_jk)) - exp(-beta*(T_end + h - t_jk)))
        #              = sum_j sum_k (alpha/beta) * exp(-beta*(T_end - t_jk)) * (1 - exp(-beta*h))
        # This effectively decays the current residual intensity.
        
        # Calculate residual intensity R_j(T_end) first
        # R_j(T) = sum_{t_k < T} exp(-beta * (T - t_k))
        
        expected_events = mus[i] * forecast_horizon
        
        for j in range(2):
            a = alphas[i, j]
            b = betas[i, j]
            ts = events[j]
            
            # Sum exp decay from all past events
            # We can use the fact that R_j(T_end) was effectively tracked.
            # But let's recalculate simply
            decays = np.exp(-b * (T_end - ts))
            R_val = np.sum(decays)
            
            # Integrate future decay
            # Int_{0}^{h} alpha * R_val * exp(-beta * u) du
            # = alpha * R_val * (1 - exp(-beta * h)) / beta
            
            term = (a * R_val / b) * (1 - np.exp(-b * forecast_horizon))
            expected_events += term
            
        pred_counts.append(expected_events)
        print(f"  {nodes[i]}: {expected_events:.2f} expected events")

    # Save Results
    results_txt = os.path.join(output_dir, 'hawkes_results.txt')
    with open(results_txt, 'w') as f:
        f.write("Hawkes Process Results\n")
        f.write("======================\n")
        f.write(f"Mus: {mus}\n")
        f.write(f"Alphas:\n{alphas}\n")
        f.write(f"Betas:\n{betas}\n")
        f.write(f"Branching Ratios:\n{branching_ratio}\n")
        f.write(f"Forecast (60s): Buy={pred_counts[0]:.2f}, Sell={pred_counts[1]:.2f}\n")
    
    print(f"Results saved to {results_txt}")

if __name__ == "__main__":
    # Hardcoded sample paths for the run
    # Adjust as needed or use arguments
    base_data = r"c:\Users\Irfan\Hawkes_Modeling\data"
    # Using 13th August data based on file list
    orders = os.path.join(base_data, "outputs", "orders", "bbbbbbINFY_orders_13082019.csv")
    trades = os.path.join(base_data, "outputs", "trades", "bbbbbbINFY_trades_13082019.csv")
    
    # Note: These paths point to OUTPUTS (CSVs), so loader should handle CSVs too or we adjust loader?
    # The OptimizedLoader handles DAT.gz strings usually. 
    # But wait, `load_symbol_for_day` calls `_stream_fwf_symbol` which expects .gz.
    # If inputs are CSV, we should just read pd.csv.
    
    # Smart Switch
    if orders.endswith('.csv'):
        print("Reading from CSVs directly...")
        orders_df = pd.read_csv(orders)
        trades_df = pd.read_csv(trades)
        
        # Ensure timestamp conversion
        orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        print(f"Loaded {len(orders_df)} orders and {len(trades_df)} trades.")
        
        print("Determining aggressor side...")
        trades_df = determine_aggressor_side(trades_df, orders_df)
        
        buyer_trades = trades_df[trades_df['aggressor_side'] == 1]['timestamp']
        seller_trades = trades_df[trades_df['aggressor_side'] == -1]['timestamp']
        
        t_min = trades_df['timestamp'].min()
        buy_ts = (buyer_trades - t_min).dt.total_seconds().values
        sell_ts = (seller_trades - t_min).dt.total_seconds().values
        buy_ts.sort()
        sell_ts.sort()
        events = [buy_ts, sell_ts]
        
        T_end = (trades_df['timestamp'].max() - t_min).total_seconds()
        
        print("Fitting Multivariate Hawkes Process (2 nodes: Buy, Sell)...")
        hawkes = MultivariateHawkes(n_nodes=2)
        
        # Better initial guess: mu ~ mean_intensity * 0.5
        avg_rate_0 = len(buy_ts) / T_end
        avg_rate_1 = len(sell_ts) / T_end
        print(f"Average Rates: Buy={avg_rate_0:.2f}/s, Sell={avg_rate_1:.2f}/s")
        
        # Patching fit method init guess via subclassing or just trusting the optimizer?
        # Let's just run fit. The class uses fixed 0.1. We should update class or just run.
        # Actually, let's update the class `fit` method in the file `models/hawkes_model.py` as well?
        # Or just instantiate usage here? 
        # I'll modify `models/hawkes_model.py` to take init_guess or be smarter.
        # But for now, let's just run and see if the print below shows success.
        
        hawkes.fit(events)
        
        mus, alphas, betas = hawkes.get_parameters()
        branching_ratio = hawkes.get_branching_ratio()
        
        # Save Results
        results_txt = os.path.join(base_data, "outputs", "hawkes_results.txt")
        with open(results_txt, 'w') as f:
            f.write("Hawkes Process Results\n")
            f.write("======================\n")
            f.write(f"Mus: {mus}\n")
            f.write(f"Alphas:\n{alphas}\n")
            f.write(f"Betas:\n{betas}\n")
            f.write(f"Branching Ratios:\n{branching_ratio}\n")
            
            nodes = ['Buy', 'Sell']
            print("\n--- Model Results ---")
            for i in range(2):
                print(f"\nNode {nodes[i]} (Target):")
                print(f"  Baseline Intensity (mu): {mus[i]:.4f}")
                for j in range(2):
                    source = nodes[j]
                    print(f"  From {source}: Alpha={alphas[i,j]:.4f}, Beta={betas[i,j]:.4f}, Branching={branching_ratio[i,j]:.4f}")
            
            forecast_horizon = 60
            pred_counts = []
            for i in range(2):
                expected_events = mus[i] * forecast_horizon
                for j in range(2):
                    a = alphas[i, j]
                    b = betas[i, j]
                    ts = events[j]
                    
                    # Vectorized decay forecast
                    # R_val = sum exp(-b * (T_end - ts))
                    # We can use the last computed state if we had it, but recomputing is cheap: O(N)
                    # Optimization: Filter ts to only recent events that matter? 
                    # exp(-beta*dt) < 1e-6 => -beta*dt < -14 => dt > 14/beta.
                    # If beta=1, look back 14s. If beta=10, 1.4s.
                    # Full sum is fine for accuracy.
                    
                    deltas = T_end - ts
                    # Filter positive deltas (past events)
                    deltas = deltas[deltas > 0]
                    decays = np.exp(-b * deltas)
                    R_val = np.sum(decays)
                    
                    term = (a * R_val / b) * (1 - np.exp(-b * forecast_horizon))
                    expected_events += term
                pred_counts.append(expected_events)
                print(f"Forecast {nodes[i]} (60s): {expected_events:.2f}")
                f.write(f"Forecast {nodes[i]} (60s): {expected_events:.2f}\n")
        
        print(f"Results saved to {results_txt}")

