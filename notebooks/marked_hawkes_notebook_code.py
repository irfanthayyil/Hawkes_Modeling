"""
Code to add to notebook 03_ofi_analysis.ipynb for Marked Hawkes Analysis

This code processes all dates using the optimized MarkedBivariateHawkesOptimized model
and saves results similar to the existing 'all_day_results' variable.
"""

# Cell 1: Import the optimized marked Hawkes model
from MarkedBivariateHawkes_optimized import MarkedBivariateHawkesOptimized
import pandas as pd
import numpy as np

# Cell 2: Process all dates with marked Hawkes model
all_day_marked_results = {}

for i, target_date in enumerate(dates):
    print(f"Processing {i+1}/{len(dates)}: {target_date}...")
    
    # Prepare data for the day
    df_day = df[df['date'] == target_date].sort_values(['timestamp', 'trade_number'])
    market_open = pd.to_datetime((target_date) + ' 09:15:00')
    df_day['t'] = (df_day['timestamp'] - market_open).dt.total_seconds()
    
    # Extract buy and sell events with volumes
    buy_mask = df_day['aggressor_side'] == +1
    sell_mask = df_day['aggressor_side'] == -1
    
    buy_times = df_day[buy_mask]['t'].to_numpy()
    sell_times = df_day[sell_mask]['t'].to_numpy()
    buy_volumes = df_day[buy_mask]['volume'].to_numpy()
    sell_volumes = df_day[sell_mask]['volume'].to_numpy()
    
    # Skip if insufficient data
    if len(buy_times) < 10 or len(sell_times) < 10:
        print(f"  Skipping {target_date}: insufficient events (buy={len(buy_times)}, sell={len(sell_times)})")
        continue
    
    try:
        # Fit marked Hawkes model
        model = MarkedBivariateHawkesOptimized()
        model.fit(buy_times, sell_times, buy_volumes, sell_volumes, optimize_beta=True, verbose=False)
        
        # Get results
        results = model.get_results()
        all_day_marked_results[target_date] = results
        
        print(f"  Success! Log-likelihood: {results['log_likelihood']:.2f}, "
              f"Spectral radius: {results['spectral_radius']:.4f}, "
              f"Avg mark: {results['avg_mark']:.4f}")
    
    except Exception as e:
        print(f"  Error for {target_date}: {e}")
        continue

print(f"\nCompleted: {len(all_day_marked_results)}/{len(dates)} dates processed successfully")

# Cell 3: Convert to DataFrame and save
df_marked_results = pd.DataFrame(all_day_marked_results).T
print(f"\nMarked Hawkes Results Shape: {df_marked_results.shape}")
print(df_marked_results.head())

# Cell 4: Save to CSV
df_marked_results.to_csv('../data/all_day_marked_results.csv')
print("Saved to: ../data/all_day_marked_results.csv")

# Cell 5: Compare with unmarked model (optional)
# If you already have all_day_results from the unmarked model
try:
    df_unmarked = pd.DataFrame(all_day_results).T
    
    print("\n" + "="*70)
    print("COMPARISON: Marked vs Unmarked Hawkes")
    print("="*70)
    
    # Merge on date index
    comparison = pd.merge(
        df_unmarked[['log_likelihood', 'spectral_radius', 'total_branching']],
        df_marked_results[['log_likelihood', 'spectral_radius', 'total_branching', 'avg_mark']],
        left_index=True,
        right_index=True,
        suffixes=('_unmarked', '_marked')
    )
    
    # Calculate improvements
    comparison['ll_improvement'] = comparison['log_likelihood_marked'] - comparison['log_likelihood_unmarked']
    comparison['sr_change'] = comparison['spectral_radius_marked'] - comparison['spectral_radius_unmarked']
    
    print(f"\nAverage Log-Likelihood Improvement: {comparison['ll_improvement'].mean():.2f}")
    print(f"Average Spectral Radius Change: {comparison['sr_change'].mean():.4f}")
    print(f"Average Volume Mark: {df_marked_results['avg_mark'].mean():.4f}")
    
    print("\nSample Comparison:")
    print(comparison.head())
    
    # Visualization (optional)
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Log-likelihood comparison
    axes[0, 0].scatter(comparison['log_likelihood_unmarked'], comparison['log_likelihood_marked'], alpha=0.6)
    axes[0, 0].plot([comparison['log_likelihood_unmarked'].min(), comparison['log_likelihood_unmarked'].max()],
                     [comparison['log_likelihood_unmarked'].min(), comparison['log_likelihood_unmarked'].max()],
                     'r--', label='y=x')
    axes[0, 0].set_xlabel('Unmarked Log-Likelihood')
    axes[0, 0].set_ylabel('Marked Log-Likelihood')
    axes[0, 0].set_title('Log-Likelihood: Marked vs Unmarked')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Spectral radius comparison
    axes[0, 1].scatter(comparison['spectral_radius_unmarked'], comparison['spectral_radius_marked'], alpha=0.6)
    axes[0, 1].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 1].set_xlabel('Unmarked Spectral Radius')
    axes[0, 1].set_ylabel('Marked Spectral Radius')
    axes[0, 1].set_title('Spectral Radius: Marked vs Unmarked')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Log-likelihood improvement distribution
    axes[1, 0].hist(comparison['ll_improvement'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(comparison['ll_improvement'].mean(), color='r', linestyle='--', 
                        label=f'Mean: {comparison["ll_improvement"].mean():.2f}')
    axes[1, 0].set_xlabel('Log-Likelihood Improvement')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Log-Likelihood Improvement')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average mark distribution
    axes[1, 1].hist(df_marked_results['avg_mark'], bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].axvline(df_marked_results['avg_mark'].mean(), color='r', linestyle='--',
                        label=f'Mean: {df_marked_results["avg_mark"].mean():.4f}')
    axes[1, 1].set_xlabel('Average Volume Mark: log(1 + volume)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Average Volume Marks')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/marked_vs_unmarked_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization saved to: ../figures/marked_vs_unmarked_comparison.png")

except NameError:
    print("\nNote: Unmarked results (all_day_results) not found. Skipping comparison.")
except Exception as e:
    print(f"\nError during comparison: {e}")
