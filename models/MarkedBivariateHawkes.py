import numpy as np
from scipy.optimize import minimize
from BivariateHawkes_optimized import BivariateHawkesOptimized

class MarkedBivariateHawkes(BivariateHawkesOptimized):
    """
    Extends the Bivariate Hawkes model to include Volume as a Mark.
    
    Intensity: λ_m(t) = μ_m + Σ_i α_mj · mark(V_i) · exp(-β(t - t_i))
    where mark(V) = log(1 + V) scales the excitation by volume.
    
    This represents a marked point process where each event has an associated 
    volume that amplifies its impact on future event intensities.
    """
    
    def _compute_integral_term(self, mu, alpha, beta, times, events, volumes, T_max):
        """
        Compute the compensator (integral term) with volume weighting.
        
        ∫₀ᵀ λ_m(t)dt = Σ_m μ_m·T + Σ_i Σ_m α_mj · mark(V_i) / β · (1 - exp(-β(T-t_i)))
        
        where j is the type of event i, and we sum over all event types m.
        """
        # Background intensity contribution (independent of volumes)
        integral = np.sum(mu) * T_max
        
        # Excitation contribution - for each past event
        for i in range(len(times)):
            event_type = events[i]
            vol = volumes[i]
            
            # Volume mark function: log(1 + vol)
            # Alternatives: vol**0.5, vol, or other concave functions
            mark = np.log(1 + vol) 
            
            time_to_end = T_max - (times[i] - times[0])
            if time_to_end > 0:
                exp_term = 1 - np.exp(-beta * time_to_end)
                # This event contributes to both buy and sell processes
                integral += (alpha[0, event_type] + alpha[1, event_type]) / beta * mark * exp_term
        
        return integral

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
        
        # Stability check: approximate spectral radius with E[mark] ≈ 1
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
        
        # 2. Sum of Log Intensities (Recursive computation with volume marks)
        sum_log_lambda = 0.0
        r = np.zeros(2)  # Recursive tracker: r[j] = Σ exp(-β(t-s)) over past events of type j
        
        # Note: Numba JIT would require passing volumes as array, which is doable
        # but requires recompilation. For flexibility, we use pure Python here.
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
                
                # CRITICAL FIX: Add mark-weighted contribution to recursive tracker
                # The previous event of type prev_type adds "mark" to the sum
                r[prev_type] += mark
                
                # Compute intensity for current event type
                # λ_m(t) = μ_m + Σ_j α_mj · r[j]
                # where r[j] already includes volume weighting
                intensity = mu[event_type] + alpha[event_type, 0] * r[0] + alpha[event_type, 1] * r[1]
            
            if intensity <= 0: 
                return 1e9  # Invalid intensity
            sum_log_lambda += np.log(intensity)

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
        # Sort by time
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