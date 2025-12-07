import numpy as np
import scipy.optimize as opt
from scipy.optimize import approx_fprime

class MultivariateHawkes:
    """
    Multivariate Hawkes Process with Exponential Kernel using MLE.
    
    Lambda_i(t) = mu_i + sum_j sum_{t_k < t} alpha_{ij} * exp(-beta_{ij} * (t - t_k))
    
    References:
    - Ogata, Y. (1981). On Lewis' simulation method for point processes.
    - Ozaki, T. (1979). Maximum likelihood estimation of Hawkes' self-exciting point processes.
    """
    
    def __init__(self, n_nodes=2, decay_beta=None):
        """
        Args:
            n_nodes (int): Number of dimensions (e.g., 2 for Buy/Sell).
            decay_beta (float or array): Fixed decay rate(s). If None, it will be estimated.
                                         Can be scalar (shared) or matrix (n,n).
        """
        self.n_nodes = n_nodes
        self.beta_fixed = decay_beta
        self.params = None
        self.se = None # Standard errors
        
    def _log_likelihood_dim(self, mu, alpha, beta, timestamps, T_max):
        """
        Computes Log-Likelihood for a single dimension i using Vectorized operations.
        
        LL_i = sum_{k=1}^{N_i} log(lambda_i(t_k)) - int_0^T lambda_i(u) du
        """
        # --- Integral term (Compensator) ---
        integral_term = mu * T_max
        for j in range(self.n_nodes):
            t_j = timestamps[j] # Events of node j
            if len(t_j) > 0:
                # Sum of integrated kernels for all events in j
                # Integral of alpha*exp(-beta*(t-tk)) from tk to T is (alpha/beta) * (1 - exp(-beta*(T-tk)))
                term_j = (alpha[j] / beta[j]) * np.sum(1 - np.exp(-beta[j] * (T_max - t_j)))
                integral_term += term_j
                
        # --- Log-sum term ---
        t_i = timestamps[self.current_dim_idx] # Target events
        
        if len(t_i) == 0:
            return -integral_term
            
        # Initialize intensity at each event t_{ik} with baseline mu
        lam_values = np.full(len(t_i), mu)
        
        # Add excitation from each dimension j
        for j in range(self.n_nodes):
            t_source = timestamps[j]
            a_ji = alpha[j]
            b_ji = beta[j]
            
            if len(t_source) == 0:
                continue
                
            # Vectorized Kernel Summation using Cumulative Sum property
            # sum_{s < t} exp(-beta * (t - s)) = exp(-beta * t) * sum_{s < t} exp(beta * s)
            
            # 1. Precompute exp(beta * s) for all source events
            exp_beta_s = np.exp(b_ji * t_source)
            
            # 2. Cumulative sum of exp(beta * s)
            # We pad with 0 at the beginning to handle case where no events are before t
            cumsum_exp_beta_s = np.concatenate(([0.0], np.cumsum(exp_beta_s)))
            
            # 3. Find indices of source events strictly before target events
            # t_source is sorted. searchsorted(side='left') returns index where t_i would be inserted
            # indices[k] = m means t_source[0...m-1] are < t_i[k]
            # So we want cumsum_exp_beta_s[m] (which is sum of first m elements)
            
            indices = np.searchsorted(t_source, t_i, side='left')
            
            # 4. Compute sum_{s < t} exp(beta * s)
            sum_exp_beta_s = cumsum_exp_beta_s[indices]
            
            # 5. Multiply by exp(-beta * t) and alpha
            term_val = a_ji * np.exp(-b_ji * t_i) * sum_exp_beta_s
            
            lam_values += term_val
                
        # Safe log
        # Ensure positive intensity
        lam_values[lam_values <= 1e-9] = 1e-9
            
        log_term = np.sum(np.log(lam_values))
        
        return log_term - integral_term


    def fit(self, events):
        """
        Fit parameters using scipy.optimize.minimize.
        
        Args:
            events: List of arrays [t_1, t_2, ...] associated with each node.
        """
        # Global start/end
        T_start = min(e[0] for e in events if len(e)>0)
        T_end = max(e[-1] for e in events if len(e)>0)
        # Shift to 0
        events = [np.array(e) - T_start for e in events]
        T_max = T_end - T_start
        
        self.estimated_params = []
        
        # Fit dimension by dimension (Separable Likelihood)
        for i in range(self.n_nodes):
            self.current_dim_idx = i
            
            # Initial guess: mu=mean_rate*0.5, alpha=0.1, beta=1.0
            # Params vector structure for dim i: [mu, alpha_0, ..., alpha_N, beta_0, ..., beta_N]
            
            rate = len(events[i]) / T_max if T_max > 0 else 0.1
            mu_init = rate * 0.5
            if mu_init < 1e-5: mu_init = 1e-5
            
            x0 = [mu_init] + [0.1]*self.n_nodes + [1.0]*self.n_nodes
            
            # Bounds: all positive
            bounds = [(1e-5, None)] * len(x0)
            
            def objective(params):
                mu = params[0]
                alpha = params[1 : 1+self.n_nodes]
                beta = params[1+self.n_nodes :]
                
                # Stability constraint (spectral radius) is global, but usually alpha/beta < 1 per pair helps.
                # For MLE, we just max likelihood.
                
                ll = self._log_likelihood_dim(mu, alpha, beta, events, T_max)
                return -ll # Minimize NLL
            
            print(f"Fitting dimension {i}...")
            res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
            
            if not res.success:
                print(f"Warning: Optimization failed for dim {i}: {res.message}")
            
            self.estimated_params.append(res.x)
            
        return self.estimated_params

    def get_parameters(self):
        """
        Returns structured parameters: Mu, Alpha_matrix, Beta_matrix
        """
        Mus = []
        Alphas = []
        Betas = []
        
        for p in self.estimated_params:
            Mus.append(p[0])
            Alphas.append(p[1 : 1+self.n_nodes])
            Betas.append(p[1+self.n_nodes :])
            
        return np.array(Mus), np.array(Alphas), np.array(Betas)

    def get_branching_ratio(self):
        """
        Branching ratio matrix G_ij = alpha_ij / beta_ij
        """
        mus, alphas, betas = self.get_parameters()
        # Be careful with shapes. alphas is (N_target, N_source) based on our loop
        # Loop i (target): params has alpha[j] (source j)
        # So alphas[i, j] is effect OF j ON i.
        
        return alphas / betas
