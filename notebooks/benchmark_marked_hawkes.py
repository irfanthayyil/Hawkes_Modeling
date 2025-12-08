"""
Performance Benchmark: Comparing MarkedBivariateHawkes vs MarkedBivariateHawkesOptimized
"""
import numpy as np
import time
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n_events = 1000

# Simulate buy and sell events
buy_times = np.sort(np.random.uniform(0, 1000, n_events // 2))
sell_times = np.sort(np.random.uniform(0, 1000, n_events // 2))
buy_volumes = np.random.lognormal(mean=2, sigma=1, size=n_events // 2)
sell_volumes = np.random.lognormal(mean=2, sigma=1, size=n_events // 2)

print(f"Benchmark with {n_events} events ({n_events//2} buy, {n_events//2} sell)")
print(f"Average volume: {np.mean(np.concatenate([buy_volumes, sell_volumes])):.2f}")
print("-" * 70)

# Test 1: Original MarkedBivariateHawkes
print("\n1. Testing MarkedBivariateHawkes (non-optimized)...")
try:
    from MarkedBivariateHawkes import MarkedBivariateHawkes
    
    start = time.time()
    model1 = MarkedBivariateHawkes()
    model1.fit(buy_times, sell_times, buy_volumes, sell_volumes, optimize_beta=True, verbose=False)
    time1 = time.time() - start
    
    results1 = model1.get_results()
    print(f"   [OK] Completed in {time1:.3f} seconds")
    print(f"   Log-likelihood: {results1['log_likelihood']:.2f}")
    print(f"   Spectral radius: {results1['spectral_radius']:.4f}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    time1 = None
    results1 = None

# Test 2: Optimized MarkedBivariateHawkesOptimized
print("\n2. Testing MarkedBivariateHawkesOptimized (with Numba)...")
try:
    from MarkedBivariateHawkes_optimized import MarkedBivariateHawkesOptimized
    
    start = time.time()
    model2 = MarkedBivariateHawkesOptimized()
    model2.fit(buy_times, sell_times, buy_volumes, sell_volumes, optimize_beta=True, verbose=False)
    time2 = time.time() - start
    
    results2 = model2.get_results()
    print(f"   [OK] Completed in {time2:.3f} seconds")
    print(f"   Log-likelihood: {results2['log_likelihood']:.2f}")
    print(f"   Spectral radius: {results2['spectral_radius']:.4f}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")
    time2 = None
    results2 = None

# Summary
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

if time1 and time2:
    speedup = time1 / time2
    print(f"MarkedBivariateHawkes:          {time1:.3f} seconds")
    print(f"MarkedBivariateHawkesOptimized: {time2:.3f} seconds")
    print(f"Speedup:                        {speedup:.2f}x")
    
    if speedup > 1.5:
        print(f"\n[SUCCESS] Optimization successful! {speedup:.2f}x faster")
    elif speedup > 1.0:
        print(f"\n[INFO] Minor speedup of {speedup:.2f}x (expected with JIT compilation)")
    else:
        print(f"\n[WARNING] No speedup detected. First run may include JIT compilation overhead.")
        print("   Run the benchmark again to see true performance after JIT warmup.")
    
    # Check if results match
    if results1 and results2:
        ll_diff = abs(results1['log_likelihood'] - results2['log_likelihood'])
        sr_diff = abs(results1['spectral_radius'] - results2['spectral_radius'])
        
        print(f"\nResult Consistency Check:")
        print(f"  Log-likelihood difference: {ll_diff:.6f}")
        print(f"  Spectral radius difference: {sr_diff:.6f}")
        
        if ll_diff < 1.0 and sr_diff < 0.01:
            print(f"  [OK] Results are consistent!")
        else:
            print(f"  [WARNING] Results differ slightly (expected due to optimization convergence)")
else:
    print("Could not complete benchmark comparison")

print("\n" + "=" * 70)

# Additional test: Multiple runs to measure JIT warmup
print("\n3. Testing JIT warmup effect (3 consecutive runs)...")
print("   This will show speedup after Numba compilation is cached")

try:
    from MarkedBivariateHawkes_optimized import MarkedBivariateHawkesOptimized
    
    times_warmup = []
    for run in range(3):
        start = time.time()
        model = MarkedBivariateHawkesOptimized()
        model.fit(buy_times, sell_times, buy_volumes, sell_volumes, optimize_beta=True, verbose=False)
        elapsed = time.time() - start
        times_warmup.append(elapsed)
        print(f"   Run {run+1}: {elapsed:.3f} seconds")
    
    print(f"\n   First run (with JIT compilation): {times_warmup[0]:.3f}s")
    print(f"   Average of runs 2-3 (JIT cached):  {np.mean(times_warmup[1:]):.3f}s")
    print(f"   Improvement after warmup:          {times_warmup[0]/np.mean(times_warmup[1:]):.2f}x")
    
except Exception as e:
    print(f"   Error during warmup test: {e}")

print("\n" + "=" * 70)
print("Benchmark complete!")
