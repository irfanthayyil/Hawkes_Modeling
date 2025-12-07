import numpy as np
from scipy.optimize import minimize
import pandas as pd

class BivariateHawkes:
    """
    A custom Bivariate Hawkes Process solver using Scipy.
    Models two event streams (e.g., Buy=0, Sell=1) with a common decay beta.
    """
    def __init__(self, beta_init=None):
        self.beta_init = beta_init
        self.params = None # [mu1, mu2, a11, a12, a21, a22] (and beta if optimizing)
        self.fitted_beta = None

    def _log_likelihood(self, params, times, events, optimize_beta=False):
        """
        Calculates the negative log-likelihood (for minimization).
        Recursive O(N) implementation.
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

        # Safety checks for solver
        if beta <= 0 or np.any(mu < 0) or np.any(alpha < 0):
            return 1e9

        n_events = len(times)
        
        # 1. Integral Term (Compensator)
        # Integral of lambda(t) from 0 to T
        # \int (mu + sum alpha*exp) = mu*T + sum(alpha/beta * (1-exp(-beta(T-tk))))
        # Approximation for T large: mu*T + sum(alpha/beta * count)
        T_max = times[-1] - times[0]
        integral_term = np.sum(mu) * T_max
        
        counts = np.array([np.sum(events == 0), np.sum(events == 1)])
        # Sum of branching ratio * total events causing excitation
        term_1_branching = (alpha[0,0] + alpha[1,0]) / beta * counts[0]
        term_2_branching = (alpha[0,1] + alpha[1,1]) / beta * counts[1]
        
        integral_term += term_1_branching + term_2_branching

        # 2. Sum of Logs Term (Recursive)
        sum_log_lambda = 0
        
        # State variables for recursion:
        # r[0] tracks excitation from past BUYS (dim 0)
        # r[1] tracks excitation from past SELLS (dim 1)
        # Note: These are scaled. Real intensity contribution is alpha * r
        # To optimize, we track the *decay factor* sum.
        r = np.zeros(2) 
        
        prev_time = times[0]
        
        for i in range(1, n_events):
            curr_time = times[i]
            dt = curr_time - prev_time
            event_type = events[i] # 0 or 1
            
            # Decay the past excitation
            decay_factor = np.exp(-beta * dt)
            r = r * decay_factor
            
            # Add the spike from the *previous* event to the history tracker
            prev_type = events[i-1]
            r[prev_type] += 1 
            
            # Calculate intensity at current moment (JUST BEFORE the jump)
            # lambda_curr = mu + alpha_m0 * r0 + alpha_m1 * r1
            intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            if intensity <= 0:
                return 1e9 # Penalty for negative intensity
                
            sum_log_lambda += np.log(intensity)
            
            prev_time = curr_time

        # Log Likelihood = Sum(ln(lambda)) - Integral
        return -(sum_log_lambda - integral_term) # Negative because we minimize

    def fit(self, buy_times, sell_times, optimize_beta=True):
        """
        Fits the model.
        buy_times: sorted list of timestamps
        sell_times: sorted list of timestamps
        """
        # 1. Merge and sort events
        # Create (time, type) tuples. 0=Buy, 1=Sell
        events_b = np.column_stack((buy_times, np.zeros(len(buy_times), dtype=int)))
        events_s = np.column_stack((sell_times, np.ones(len(sell_times), dtype=int)))
        
        data = np.vstack((events_b, events_s))
        data = data[data[:, 0].argsort()] # Sort by time
        
        times = data[:, 0]
        events = data[:, 1].astype(int)
        
        # 2. Initial Guesses
        # mu: roughly count / total_time
        T = times[-1] - times[0]
        mu_init = [len(buy_times)/T * 0.5, len(sell_times)/T * 0.5] # Start assuming 50% endogeneity
        alpha_init = [0.1, 0.1, 0.1, 0.1] # Small positive values
        
        if optimize_beta:
            beta_start = 1.0 if self.beta_init is None else self.beta_init
            init_params = mu_init + alpha_init + [beta_start]
            bounds = [(1e-5, None)]*7 # All positive
        else:
            init_params = mu_init + alpha_init
            bounds = [(1e-5, None)]*6

        # 3. Optimization
        res = minimize(
            self._log_likelihood,
            x0=init_params,
            args=(times, events, optimize_beta),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-4, 'disp': False}
        )
        
        self.params = res.x
        if optimize_beta:
            self.fitted_beta = res.x[6]
        else:
            self.fitted_beta = self.beta_init
            
        return self

    def get_results(self):
        # Unpack
        p = self.params
        mu = p[0:2]
        alpha = p[2:6].reshape(2,2)
        beta = self.fitted_beta
        
        # Branching Ratios (Alpha / Beta)
        br = alpha / beta
        spectral_radius = np.max(np.abs(np.linalg.eigvals(br)))
        
        return {
            'mu_buy': mu[0], 'mu_sell': mu[1],
            'alpha_bb': alpha[0,0], 'alpha_bs': alpha[0,1],
            'alpha_sb': alpha[1,0], 'alpha_ss': alpha[1,1],
            'beta': beta,
            'br_bb': br[0,0], 'br_bs': br[0,1],
            'br_sb': br[1,0], 'br_ss': br[1,1],
            'spectral_radius': spectral_radius
        }