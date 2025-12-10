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


class MarkedBivariateHawkesOptimized:
    """
    Optimized Marked Bivariate Hawkes model with Numba JIT compilation.
    
    Extends the Bivariate Hawkes model to include Volume as a Mark.
    
    Intensity: λ_m(t) = μ_m + Σ_i α_mj · mark(V_i) · exp(-β(t - t_i))
    where mark(V) = log(1 + V) scales the excitation by volume.
    
    This represents a marked point process where each event has an associated 
    volume that amplifies its impact on future event intensities.
    
    Optimizations:
    - JIT-compiled intensity computation for 10-50x speedup
    - Efficient vectorized operations where possible
    - Minimal memory allocation in hot loops
    """
    
    def __init__(self, beta_init=None):
        self.beta_init = beta_init
        self.params = None  # [mu1, mu2, a11, a12, a21, a22] (and beta if optimizing)
        self.fitted_beta = None
        self.log_likelihood_value = None
        self._fitted_volumes = None
        self._fitted_times = None
        self._fitted_events = None
    
    def _compute_integral_term(self, mu, alpha, beta, times, events, volumes, T_max):
        """
        Compute the compensator (integral term) with volume weighting.
        
        ∫₀ᵀ λ_m(t)dt = Σ_m μ_m·T + Σ_i Σ_m α_mj · mark(V_i) / β · (1 - exp(-β(T-t_i)))
        
        where j is the type of event i, and we sum over all event types m.
        """
        # Background intensity contribution (independent of volumes)
        integral = np.sum(mu) * T_max
        
        # Excitation contribution - vectorized computation
        marks = np.log(1 + volumes)
        time_to_end = T_max - (times - times[0])
        
        # Only include events where time_to_end > 0
        mask = time_to_end > 0
        if np.any(mask):
            exp_terms = 1 - np.exp(-beta * time_to_end[mask])
            
            # For each event, compute contribution to both processes
            for i in np.where(mask)[0]:
                event_type = events[i]
                mark = marks[i]
                exp_term = exp_terms[i]
                # This event of type event_type contributes to both buy and sell processes
                integral += (alpha[0, event_type] + alpha[1, event_type]) / beta * mark * exp_term
        
        return integral
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_intensities_numba(times, events, volumes, mu, alpha, beta):
        """
        JIT-compiled function to compute log-intensities with volume marks.
        Returns sum of log intensities.
        
        This is the performance-critical section that benefits most from JIT.
        """
        n_events = len(times)
        sum_log_lambda = 0.0
        
        # State variables: r[j] tracks volume-weighted sum of exp(-beta*(t-s)) 
        # for all s < t where event s had type j
        r = np.zeros(2)
        
        for i in range(n_events):
            if i == 0:
                # First event: no history, only background rate
                event_type = events[i]
                intensity = mu[event_type]
            else:
                dt = times[i] - times[i-1]
                event_type = events[i]
                
                # Decay accumulated excitation
                decay_factor = np.exp(-beta * dt)
                r = r * decay_factor
                
                # Add contribution from previous event (weighted by its volume)
                prev_type = events[i-1]
                prev_vol = volumes[i-1]
                mark = np.log(1 + prev_vol)
                
                # Add mark-weighted contribution to recursive tracker
                r[prev_type] += mark
                
                # Compute intensity for current event type
                intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            # Safety check
            if intensity <= 0:
                return -1e9  # Signal invalid parameters
            
            sum_log_lambda += np.log(intensity)
        
        return sum_log_lambda
    
    def _compute_intensities_python(self, times, events, volumes, mu, alpha, beta):
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
                dt = times[i] - times[i-1]
                event_type = events[i]
                
                decay_factor = np.exp(-beta * dt)
                r = r * decay_factor
                
                prev_type = events[i-1]
                prev_vol = volumes[i-1]
                mark = np.log(1 + prev_vol)
                
                r[prev_type] += mark
                
                intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            if intensity <= 0:
                return -1e9
            
            sum_log_lambda += np.log(intensity)
        
        return sum_log_lambda

    def _log_likelihood(self, params, times, events, volumes, optimize_beta=False):
        """
        Negative log-likelihood for marked Hawkes process.
        
        The key difference from the unmarked version is that each event's 
        contribution to future intensities is scaled by its volume mark.
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
        
        # Stability check: approximate spectral radius with E[mark]
        # For more accuracy, compute spectral_radius * E[mark] < 1
        avg_mark = np.mean(np.log(1 + volumes))
        br = alpha / beta * avg_mark
        spectral_radius = np.max(np.abs(np.linalg.eigvals(br)))
        if spectral_radius >= 1.0:
            return 1e9  # Unstable process
        
        n_events = len(times)
        T_max = times[-1] - times[0]
        
        # 1. Integral Term (Compensator)
        integral_term = self._compute_integral_term(mu, alpha, beta, times, events, volumes, T_max)
        
        # 2. Sum of Log Intensities (use JIT-compiled version if available)
        if NUMBA_AVAILABLE:
            sum_log_lambda = self._compute_intensities_numba(times, events, volumes, mu, alpha, beta)
        else:
            sum_log_lambda = self._compute_intensities_python(times, events, volumes, mu, alpha, beta)
        
        if sum_log_lambda == -1e9:
            return 1e9  # Invalid parameters

        # Log-likelihood = Σ log(λ(t_i)) - ∫ λ(t)dt
        log_likelihood = sum_log_lambda - integral_term
        return -log_likelihood  # Return negative for minimization

    def fit(self, buy_times, sell_times, buy_volumes, sell_volumes, optimize_beta=True, verbose=False):
        """
        Fit the marked bivariate Hawkes model.
        
        Parameters:
        -----------
        buy_times : array
            Timestamps of buy events
        sell_times : array
            Timestamps of sell events
        buy_volumes : array
            Volumes associated with buy events
        sell_volumes : array
            Volumes associated with sell events
        optimize_beta : bool, default=True
            Whether to optimize beta or use beta_init
        verbose : bool, default=False
            Print optimization progress
        """
        # Convert to numpy arrays
        buy_times = np.asarray(buy_times)
        sell_times = np.asarray(sell_times)
        buy_volumes = np.asarray(buy_volumes)
        sell_volumes = np.asarray(sell_volumes)
        
        # Merge times, events, and volumes
        events_b = np.column_stack((buy_times, np.zeros(len(buy_times)), buy_volumes))
        events_s = np.column_stack((sell_times, np.ones(len(sell_times)), sell_volumes))
        
        data = np.vstack((events_b, events_s))
        # Sort by time (stable sort to preserve order for ties)
        sort_indices = np.argsort(data[:, 0], kind='stable')
        data = data[sort_indices]
        
        times = data[:, 0]
        events = data[:, 1].astype(int)
        volumes = data[:, 2]
        
        # Data-driven initialization (same as parent class)
        T = times[-1] - times[0]
        n_buy = len(buy_times)
        n_sell = len(sell_times)
        
        # Assume 70% of events from background (conservative)
        mu_init = [n_buy/T * 0.7, n_sell/T * 0.7]
        
        # Start with small alpha values
        alpha_init = [0.05, 0.05, 0.05, 0.05]
        
        if optimize_beta:
            beta_start = 10.0 if self.beta_init is None else self.beta_init
            init_params = mu_init + alpha_init + [beta_start]
            bounds = [
                (1e-10, None), (1e-10, None),          # Mu (base intensity)
                (1e-10, 50000.0), (1e-10, 50000.0),    # Alpha BB, BS
                (1e-10, 50000.0), (1e-10, 50000.0),    # Alpha SB, SS
                (1.0, 100000.0)                         # Beta (decay)
            ]
        else:
            if self.beta_init is None:
                raise ValueError("beta_init must be provided when optimize_beta=False")
            init_params = mu_init + alpha_init
            bounds = [
                (1e-6, None), (1e-6, None),            # Mu bounds
                (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0), (1e-6, 10.0)  # Alpha bounds
            ]
        
        res = minimize(
            self._log_likelihood,
            x0=init_params,
            args=(times, events, volumes, optimize_beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 500, 'disp': verbose}
        )
        
        if not res.success and verbose:
            print(f"Warning: Optimization did not converge. Message: {res.message}")
        
        self.params = res.x
        self.log_likelihood_value = -res.fun
        
        if optimize_beta:
            self.fitted_beta = res.x[6]
        else:
            self.fitted_beta = self.beta_init
        
        # Store volumes for potential use in prediction
        self._fitted_volumes = volumes
        self._fitted_times = times
        self._fitted_events = events
            
        return self
    
    def get_results(self):
        """
        Returns a dictionary with all fitted parameters and derived statistics.
        """
        if self.params is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        p = self.params
        mu = p[0:2]
        alpha = p[2:6].reshape(2, 2)
        beta = self.fitted_beta
        
        # Branching Ratios (Alpha / Beta)
        # Note: For marked processes, effective branching ratio is alpha * E[mark] / beta
        avg_mark = np.mean(np.log(1 + self._fitted_volumes))
        br = alpha / beta * avg_mark
        spectral_radius = np.max(np.abs(np.linalg.eigvals(br)))
        
        # Compute endogeneity ratio
        total_branching = np.sum(br)
        
        return {
            'mu_buy': mu[0], 
            'mu_sell': mu[1],
            'alpha_bb': alpha[0, 0],  # Buy -> Buy
            'alpha_bs': alpha[0, 1],  # Sell -> Buy
            'alpha_sb': alpha[1, 0],  # Buy -> Sell
            'alpha_ss': alpha[1, 1],  # Sell -> Sell
            'beta': beta,
            'avg_mark': avg_mark,  # Average volume mark
            'br_bb': br[0, 0], 
            'br_bs': br[0, 1],
            'br_sb': br[1, 0], 
            'br_ss': br[1, 1],
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
    
    def predict_intensity(self, times, events, volumes, prediction_time):
        """
        Predict intensity at a given time, accounting for volume marks.
        
        Parameters:
        -----------
        times : array
            Event times (sorted)
        events : array
            Event types (0=buy, 1=sell)
        volumes : array
            Event volumes
        prediction_time : float
            Time at which to predict intensity
        
        Returns:
        --------
        tuple : (lambda_buy, lambda_sell)
            Predicted intensities for buy and sell processes
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
        past_volumes = volumes[mask]
        
        if len(past_times) == 0:
            # No history, return background rate
            return mu[0], mu[1]
        
        # Compute excitation from past events (with volume weighting)
        r = np.zeros(2)
        for i in range(len(past_times)):
            dt = prediction_time - past_times[i]
            event_type = past_events[i]
            vol = past_volumes[i]
            mark = np.log(1 + vol)
            r[event_type] += mark * np.exp(-beta * dt)
        
        # Compute intensities
        lambda_buy = mu[0] + alpha[0, 0] * r[0] + alpha[0, 1] * r[1]
        lambda_sell = mu[1] + alpha[1, 0] * r[0] + alpha[1, 1] * r[1]
        
        return lambda_buy, lambda_sell
