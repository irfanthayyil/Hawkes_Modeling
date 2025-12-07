
import numpy as np
from scipy.optimize import minimize

class HawkesProcess:
    """
    Univariate Hawkes Process with Exponential Kernel.
    
    Conditional Intensity function:
        lambda(t) = mu + sum_{t_i < t} alpha * beta * exp(-beta * (t - t_i))
        
    Parameters:
    -----------
    mu : float
        Baseline intensity > 0.
    alpha : float
        Excitation parameter (branching ratio approx) > 0.
    beta : float
        Decay rate > 0.
    """
    
    def __init__(self):
        self.params = None
        self.mu = None
        self.alpha = None
        self.beta = None
        
    def _recursive_intensity(self, timestamps, alpha, beta):
        """
        Calculate intensity values at each event timestamp using recursive formulation.
        R(i) = exp(-beta * (t_i - t_{i-1})) * (R(i-1) + alpha * beta)
        lambda(t_i) = mu + R(i)
        """
        n = len(timestamps)
        r = np.zeros(n)
        
        # Precompute time differences
        dt = np.diff(timestamps)
        # Insert 0 at start for iteration convenience, though loop handles it
        
        # R[0] is 0 because no previous events
        current_r = 0.0
        
        # Iterate (could be optimized with numba/cython but pure python for portability)
        # Using a loop is slow for millions of points, but fine for thousands.
        # Vectorization is tricky for the recursive part.
        
        for i in range(1, n):
            # R(i) depends on R(i-1) decayed
            decay = np.exp(-beta * dt[i-1])
            current_r = decay * (current_r + alpha * beta)
            r[i] = current_r
            
        return r

    def log_likelihood(self, params, timestamps, T):
        """
        Calculate negative log likelihood.
        params: [mu, alpha, beta]
        """
        mu, alpha, beta = params
        
        if mu <= 0 or alpha < 0 or beta <= 0:
            return np.inf
            
        n = len(timestamps)
        
        # 1. Sum of log intensities
        # lambda(t_i) = mu + R(i) (accumulated excitation from past)
        # We need to exclude the self-excitation at the exact moment t_i if we strictly follow lambda(t_i-)
        # Usually for likelihood we take lambda(t_i) just before the jump.
        
        r_vals = self._recursive_intensity(timestamps, alpha, beta)
        lam_vals = mu + r_vals
        
        # Avoid log(0)
        lam_vals[lam_vals <= 1e-9] = 1e-9
        term1 = np.sum(np.log(lam_vals))
        
        # 2. Integral of intensity function from 0 to T
        # Int(mu) = mu * T
        # Int(sum kernel) = sum_{i} Int_{t_i}^{T} (alpha * beta * exp(-beta(t-t_i)))
        #                 = sum_{i} alpha * (1 - exp(-beta(T - t_i)))
        
        term2 = mu * T + np.sum(alpha * (1 - np.exp(-beta * (T - timestamps))))
        
        ll = term1 - term2
        return -ll # Return negative LL for minimization

    def fit(self, timestamps, T=None):
        """
        Fit the model to timestamps (1D array of event times).
        T: end time of observation. If None, max(timestamps).
        """
        timestamps = np.array(timestamps)
        timestamps.sort()
        
        if T is None:
            T = timestamps[-1]
            
        # Initial guess
        # mu: estimation ~ N / T
        # alpha: small < 1
        # beta: moderate
        initial_guess = [len(timestamps)/T * 0.5, 0.5, 1.0]
        
        bounds = [(1e-6, None), (1e-6, None), (1e-6, None)]
        
        res = minimize(
            self.log_likelihood, 
            initial_guess, 
            args=(timestamps, T), 
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.params = res.x
        self.mu, self.alpha, self.beta = res.x
        return self

    def get_branching_ratio(self):
        # Integral of kernel alpha*beta*exp(-beta*t) is alpha
        return self.alpha
