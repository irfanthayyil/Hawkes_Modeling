import numpy as np
from scipy.optimize import minimize
import pandas as pd
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Install with: pip install numba")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

class BivariateHawkesOptimized:
    """
    An optimized Bivariate Hawkes Process solver using Scipy.
    Models two event streams (e.g., Buy=0, Sell=1) with a common decay beta.
    
    Improvements:
    - Fixed logical errors in log-likelihood calculation
    - Corrected integral term computation
    - Added Numba JIT compilation for speed
    - Fixed indexing bug in the recursive loop
    - Improved numerical stability
    """
    def __init__(self, beta_init=None):
        self.beta_init = beta_init
        self.params = None # [mu1, mu2, a11, a12, a21, a22] (and beta if optimizing)
        self.fitted_beta = None
        self.log_likelihood_value = None

    def _compute_integral_term(self, mu, alpha, beta, times, events, T_max):
        """
        Correctly compute the integral term (compensator).
        ∫₀ᵀ λₘ(t)dt = μₘ·T + Σₙ Σⱼ αₘⱼ/β · (1 - exp(-β(T-tₙ)))
        where n loops over all events and j is the type of event n.
        """
        # Background intensity contribution
        integral = np.sum(mu) * T_max
        
        # Excitation contribution - for each past event
        for i in range(len(times)):
            event_type = events[i]
            time_to_end = T_max - (times[i] - times[0])
            
            if time_to_end > 0:
                exp_term = 1 - np.exp(-beta * time_to_end)
                # This event of type event_type contributes to both processes
                integral += (alpha[0, event_type] + alpha[1, event_type]) / beta * exp_term
        
        return integral

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_intensities_numba(times, events, mu, alpha, beta):
        """
        JIT-compiled function to compute log-intensities efficiently.
        Returns sum of log intensities.
        """
        n_events = len(times)
        sum_log_lambda = 0.0
        
        # State variables: r[j] tracks sum of exp(-beta*(t-s)) for all s < t where event s had type j
        r = np.zeros(2)
        
        for i in range(n_events):
            if i == 0:
                # First event: no history, just use background rate
                event_type = events[i]
                intensity = mu[event_type]
            else:
                curr_time = times[i]
                prev_time = times[i-1]
                dt = curr_time - prev_time
                event_type = events[i]
                
                # Decay the accumulated excitation
                decay_factor = np.exp(-beta * dt)
                r = r * decay_factor
                
                # Add contribution from previous event
                prev_type = events[i-1]
                r[prev_type] += 1.0
                
                # Compute intensity for current event type
                intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            # Safety check
            if intensity <= 0:
                return -1e9  # Signal invalid parameters
            
            sum_log_lambda += np.log(intensity)
        
        return sum_log_lambda

    def _compute_intensities_python(self, times, events, mu, alpha, beta):
        """
        Pure Python fallback if Numba is not available.
        """
        n_events = len(times)
        sum_log_lambda = 0.0
        r = np.zeros(2)
        
        for i in range(n_events):
            if i == 0:
                event_type = events[i]
                intensity = mu[event_type]
            else:
                curr_time = times[i]
                prev_time = times[i-1]
                dt = curr_time - prev_time
                event_type = events[i]
                
                decay_factor = np.exp(-beta * dt)
                r = r * decay_factor
                
                prev_type = events[i-1]
                r[prev_type] += 1.0
                
                intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            if intensity <= 0:
                return -1e9
            
            sum_log_lambda += np.log(intensity)
        
        return sum_log_lambda

    def _log_likelihood(self, params, times, events, optimize_beta=False):
        """
        Calculates the negative log-likelihood (for minimization).
        Fixed version with correct integral term and indexing.
        """
        # Unpack parameters
        if optimize_beta:
            mu = params[0:2]
            alpha = params[2:6].reshape(2, 2)
            beta = params[6]
        else:
            mu = params[0:2]
            alpha = params[2:6].reshape(2, 2)
            beta = self.beta_init

        # Safety checks
        if beta <= 0 or np.any(mu < 0) or np.any(alpha < 0):
            return 1e9
        
        # Check stability: spectral radius of alpha/beta should be < 1
        br = alpha / beta
        spectral_radius = np.max(np.abs(np.linalg.eigvals(br)))
        if spectral_radius >= 1.0:
            return 1e9  # Unstable process

        n_events = len(times)
        
        # 1. Compute integral term (compensator)
        T_max = times[-1] - times[0]
        integral_term = self._compute_integral_term(mu, alpha, beta, times, events, T_max)
        
        # 2. Compute sum of log intensities
        if NUMBA_AVAILABLE:
            sum_log_lambda = self._compute_intensities_numba(times, events, mu, alpha, beta)
        else:
            sum_log_lambda = self._compute_intensities_python(times, events, mu, alpha, beta)
        
        if sum_log_lambda == -1e9:
            return 1e9  # Invalid parameters

        # Log Likelihood = Sum(ln(lambda)) - Integral
        log_likelihood = sum_log_lambda - integral_term
        
        return -log_likelihood  # Negative because we minimize

    def fit(self, buy_times, sell_times, optimize_beta=True, verbose=False):
        """
        Fits the model.
        buy_times: sorted array/list of timestamps
        sell_times: sorted array/list of timestamps
        optimize_beta: if True, optimize beta; otherwise use beta_init
        verbose: if True, print optimization progress
        """
        # 1. Merge and sort events
        buy_times = np.asarray(buy_times)
        sell_times = np.asarray(sell_times)
        
        # Create (time, type) tuples. 0=Buy, 1=Sell
        events_b = np.column_stack((buy_times, np.zeros(len(buy_times), dtype=int)))
        events_s = np.column_stack((sell_times, np.ones(len(sell_times), dtype=int)))
        
        data = np.vstack((events_b, events_s))
        data = data[data[:, 0].argsort()]  # Sort by time
        
        times = data[:, 0]
        events = data[:, 1].astype(int)
        
        # 2. Better initial guesses based on data
        T = times[-1] - times[0]
        n_buy = len(buy_times)
        n_sell = len(sell_times)
        
        # Assume 70% of events are from background (conservative)
        mu_init = [n_buy/T * 0.7, n_sell/T * 0.7]
        
        # Start with small alpha values
        alpha_init = [0.05, 0.05, 0.05, 0.05]
        
        if optimize_beta:
            beta_start = 10.0 if self.beta_init is None else self.beta_init
            init_params = mu_init + alpha_init + [beta_start]
            # bounds = [(1e-6, None), (1e-6, None),  # mu bounds
            #          (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0),  # alpha bounds
            #          (1e-3, 100.0)]  # beta bound

            # Allow Alpha up to 20,000 and Beta up to 50,000 (approx 20 microsecond resolution)
            bounds = [
                (1e-10, None), (1e-10, None),          # Mu (Base intensity)
                (1e-10, 50000.0), (1e-10, 50000.0),    # Alpha BB, BS
                (1e-10, 50000.0), (1e-10, 50000.0),    # Alpha SB, SS
                (1.0, 100000.0)                         # Beta (Decay)
            ]
        else:
            if self.beta_init is None:
                raise ValueError("beta_init must be provided when optimize_beta=False")
            init_params = mu_init + alpha_init
            bounds = [(1e-6, None), (1e-6, None),  # mu bounds
                     (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0)]  # alpha bounds

        # 3. Optimization with improved settings
        res = minimize(
            self._log_likelihood,
            x0=init_params,
            args=(times, events, optimize_beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 500, 'disp': verbose}
        )
        
        if not res.success and verbose:
            print(f"Warning: Optimization did not converge. Message: {res.message}")
        
        self.params = res.x
        self.log_likelihood_value = -res.fun  # Store the final log-likelihood
        
        if optimize_beta:
            self.fitted_beta = res.x[6]
        else:
            self.fitted_beta = self.beta_init
            
        return self

    def get_results(self):
        """
        Returns a dictionary with all fitted parameters and derived statistics.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        p = self.params
        mu = p[0:2]
        alpha = p[2:6].reshape(2,2)
        beta = self.fitted_beta
        
        # Branching Ratios (Alpha / Beta)
        br = alpha / beta
        spectral_radius = np.max(np.abs(np.linalg.eigvals(br)))
        
        # Compute endogeneity ratio
        # Expected number of offspring = sum of branching ratios
        total_branching = np.sum(br)
        
        return {
            'mu_buy': mu[0], 
            'mu_sell': mu[1],
            'alpha_bb': alpha[0,0],  # Buy -> Buy
            'alpha_bs': alpha[0,1],  # Sell -> Buy
            'alpha_sb': alpha[1,0],  # Buy -> Sell
            'alpha_ss': alpha[1,1],  # Sell -> Sell
            'beta': beta,
            'br_bb': br[0,0], 
            'br_bs': br[0,1],
            'br_sb': br[1,0], 
            'br_ss': br[1,1],
            'spectral_radius': spectral_radius,
            'total_branching': total_branching,
            'log_likelihood': self.log_likelihood_value,
            'is_stationary': spectral_radius < 1.0
        }
    
    def get_results_df(self):
        """
        Returns results as a pandas DataFrame for easy viewing.
        """
        results = self.get_results()
        return pd.DataFrame([results])
    
    def predict_intensity(self, times, events, prediction_time):
        """
        Predict the intensity at a given time based on past events.
        
        Parameters:
        -----------
        times: array of event times (sorted)
        events: array of event types (0 or 1)
        prediction_time: time at which to predict intensity
        
        Returns:
        --------
        intensity for buy (type 0) and sell (type 1) processes
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        mu = self.params[0:2]
        alpha = self.params[2:6].reshape(2, 2)
        beta = self.fitted_beta
        
        # Filter events before prediction_time
        mask = times < prediction_time
        past_times = times[mask]
        past_events = events[mask]
        
        if len(past_times) == 0:
            # No history, return background rate
            return mu[0], mu[1]
        
        # Compute excitation from past events
        r = np.zeros(2)
        for i in range(len(past_times)):
            dt = prediction_time - past_times[i]
            event_type = past_events[i]
            r[event_type] += np.exp(-beta * dt)
        
        # Compute intensities
        lambda_buy = mu[0] + alpha[0, 0] * r[0] + alpha[0, 1] * r[1]
        lambda_sell = mu[1] + alpha[1, 0] * r[0] + alpha[1, 1] * r[1]
        
        return lambda_buy, lambda_sell
