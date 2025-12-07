"""
Comparison script to demonstrate the fixes and improvements in the optimized version.
"""
import numpy as np
import time
from BivariateHawkes import BivariateHawkes
from BivariateHawkes_optimized import BivariateHawkesOptimized

def generate_synthetic_hawkes_data(mu, alpha, beta, T=100, seed=42):
    """
    Generate synthetic bivariate Hawkes data for testing.
    Simple simulation using Ogata's thinning algorithm.
    """
    np.random.seed(seed)
    
    buy_times = []
    sell_times = []
    
    t = 0
    lambda_max = mu[0] + mu[1] + 10  # Upper bound on intensity
    
    while t < T:
        # Generate candidate inter-arrival time
        u = np.random.random()
        dt = -np.log(u) / lambda_max
        t = t + dt
        
        if t > T:
            break
        
        # Compute actual intensity at time t
        lambda_buy = mu[0]
        lambda_sell = mu[1]
        
        # Add contributions from past events
        for tb in buy_times:
            if t - tb < 10/beta:  # Only consider recent events
                lambda_buy += alpha[0, 0] * np.exp(-beta * (t - tb))
                lambda_sell += alpha[1, 0] * np.exp(-beta * (t - tb))
        
        for ts in sell_times:
            if t - ts < 10/beta:
                lambda_buy += alpha[0, 1] * np.exp(-beta * (t - ts))
                lambda_sell += alpha[1, 1] * np.exp(-beta * (t - ts))
        
        total_lambda = lambda_buy + lambda_sell
        
        # Accept/reject
        s = np.random.random()
        if s * lambda_max <= total_lambda:
            # Accept event, decide type
            if np.random.random() < lambda_buy / total_lambda:
                buy_times.append(t)
            else:
                sell_times.append(t)
    
    return np.array(buy_times), np.array(sell_times)

def test_comparison():
    """
    Compare original and optimized implementations.
    """
    print("=" * 70)
    print("BIVARIATE HAWKES MODEL - COMPARISON TEST")
    print("=" * 70)
    
    # True parameters for synthetic data
    true_mu = np.array([0.5, 0.4])
    true_alpha = np.array([[0.3, 0.2], [0.15, 0.25]])
    true_beta = 1.5
    
    print("\n1. Generating synthetic data with known parameters:")
    print(f"   True mu: {true_mu}")
    print(f"   True alpha:\n{true_alpha}")
    print(f"   True beta: {true_beta}")
    print(f"   True branching ratios:\n{true_alpha / true_beta}")
    
    buy_times, sell_times = generate_synthetic_hawkes_data(
        true_mu, true_alpha, true_beta, T=1000
    )
    
    print(f"\n   Generated {len(buy_times)} buy events and {len(sell_times)} sell events")
    
    # Test original implementation
    print("\n2. Testing ORIGINAL implementation...")
    try:
        start = time.time()
        model_orig = BivariateHawkes(beta_init=None)
        model_orig.fit(buy_times, sell_times, optimize_beta=True)
        time_orig = time.time() - start
        
        results_orig = model_orig.get_results()
        print(f"   [OK] Completed in {time_orig:.2f} seconds")
        print(f"   Fitted parameters:")
        for key, val in results_orig.items():
            print(f"     {key}: {val:.4f}")
    except Exception as e:
        print(f"   [FAILED]: {e}")
        results_orig = None
        time_orig = None
    
    # Test optimized implementation
    print("\n3. Testing OPTIMIZED implementation...")
    try:
        start = time.time()
        model_opt = BivariateHawkesOptimized(beta_init=None)
        model_opt.fit(buy_times, sell_times, optimize_beta=True, verbose=False)
        time_opt = time.time() - start
        
        results_opt = model_opt.get_results()
        print(f"   [OK] Completed in {time_opt:.2f} seconds")
        print(f"   Fitted parameters:")
        for key, val in results_opt.items():
            print(f"     {key}: {val:.4f}")
        
        if time_orig:
            speedup = time_orig / time_opt
            print(f"\n   >> SPEEDUP: {speedup:.2f}x faster")
    except Exception as e:
        print(f"   [FAILED]: {e}")
        results_opt = None
    
    # Compare results
    if results_orig and results_opt:
        print("\n4. Comparing log-likelihoods:")
        if 'log_likelihood' in results_opt:
            print(f"   Optimized log-likelihood: {results_opt['log_likelihood']:.2f}")
            print("   (Original version did not track this)")
        
        print("\n5. Checking parameter recovery:")
        print("   Parameter      True      Original   Optimized")
        print("   " + "-" * 55)
        params_to_check = [
            ('mu_buy', true_mu[0]),
            ('mu_sell', true_mu[1]),
            ('alpha_bb', true_alpha[0, 0]),
            ('alpha_bs', true_alpha[0, 1]),
            ('alpha_sb', true_alpha[1, 0]),
            ('alpha_ss', true_alpha[1, 1]),
            ('beta', true_beta)
        ]
        
        for param_name, true_val in params_to_check:
            orig_val = results_orig.get(param_name, np.nan)
            opt_val = results_opt.get(param_name, np.nan)
            print(f"   {param_name:12s} {true_val:8.4f}  {orig_val:8.4f}   {opt_val:8.4f}")
    
    print("\n" + "=" * 70)
    print("KEY IMPROVEMENTS IN OPTIMIZED VERSION:")
    print("=" * 70)
    print("""
1. **Fixed Critical Bug**: Loop now starts at index 0, not 1
   - Original version missed the first event's contribution
   - This caused biased parameter estimates

2. **Corrected Integral Term**: Properly computes the compensator
   - Original used rough approximation
   - New version uses exact formula

3. **Speed Improvements**:
   - Numba JIT compilation for hot loops (if available)
   - Pre-allocated arrays
   - Better numerical stability checks

4. **Enhanced Features**:
   - Tracks log-likelihood value
   - Checks process stability (spectral radius < 1)
   - Better initial parameter guesses
   - Intensity prediction method
   - Results as DataFrame

5. **Improved Robustness**:
   - Better bounds on parameters
   - Stability constraints
   - Graceful handling of edge cases
    """)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_comparison()
