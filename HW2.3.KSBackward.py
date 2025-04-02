import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit, prange

# ============================
# 1. Model Parameters
# ============================
beta = 0.99        # Discount factor
sigma = 1.0        # Risk aversion (log utility)
delta = 0.025      # Depreciation rate
alpha = 0.36       # Capital share
ell = 0.3271       # Effective labor supply
ug = 0.04          # Unemployment rate in good times
ub = 0.10          # Unemployment rate in bad times
zg = 1.01          # TFP in good times
zb = 0.99          # TFP in bad times

# Transition probabilities for (z, e) pairs
# States: 0=BU, 1=BE, 2=GU, 3=GE
prob_gg = 0.875   # Prob(z'=g | z=g)
prob_bb = 0.875   # Prob(z'=b | z=b)
prob_gb = 1 - prob_gg
prob_bg = 1 - prob_bb

Pi = np.zeros((4, 4))
Pi[0] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[1] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[2] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]
Pi[3] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]

# Asset grid parameters
a_min, a_max, a_size = 1.0, 20.0, 200
grid_a = np.linspace(a_min, a_max, a_size)

num_agents = 2000
T = 20  # Transition horizon

# Forecasting rule coefficients (pre-calibrated)
# For good state:
a0_g, a1_g = 0.0288, 0.9858  
# For bad state:
a0_b, a1_b = 0.0082, 0.9944

# ============================
# 2. Core Functions
# ============================
@jit(nopython=True)
def utility(c):
    # Log utility function; returns a very low number if consumption is nonpositive
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def wage(r, z, K):
    # Wage function derived from production
    return (1 - alpha) * z * (K**alpha) * ell

@jit(nopython=True)
def interest(z, K):
    # Interest rate function
    return alpha * z * (K**(alpha - 1)) - delta

@jit(nopython=True, parallel=True)
def solve_backward_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g, T, tol=1e-6, max_iter=500):
    """
    Solves the household problem via backward induction.
    We assume that period T is in equilibrium (i.e. terminal condition given by the stationary solution).
    Then, for t = T-1,...,0 we solve the Bellman equation.
    
    The shock specification is:
      - t = 0: Pre-shock equilibrium (bad state: use zb, bad rule).
      - t = 1: MIT shock period (good state: use zg, good rule).
      - t >= 2: Aftershock, the economy is in the bad state (use zb, bad rule).
      
    The forecast for aggregate capital is computed using:
       log(K_forecast) = a0 + a1 * log(40)
    with 40 being the fixed initial guess.
    """
    a_size = len(grid_a)
    # Allocate arrays for the value function and policy function for each period and state.
    # Dimensions: time x state x asset grid
    V = np.empty((T+1, 4, a_size))
    PF = np.empty((T+1, 4, a_size))
    
    # ------------------------------------------------------------------------
    # Terminal condition: assume period T is in equilibrium.
    # We compute the equilibrium value function using a fixed-point iteration.
    # Here, we use a simple backward value function iteration (for max_iter iterations).
    V_T = np.zeros((4, a_size))
    V_T[0, :] = 50   # BU
    V_T[1, :] = 100  # BE
    V_T[2, :] = 150  # GU
    V_T[3, :] = 200  # GE
    PF_T = np.empty((4, a_size))
    K_guess = 40.0
    logK_guess = np.log(K_guess)
    
    for it in range(max_iter):
        V_new = V_T.copy()
        for s in prange(4):
            # For terminal equilibrium, use state-specific rules:
            if s < 2:
                z = zb
                logK = a0_b + a1_b * logK_guess
            else:
                z = zg
                logK = a0_g + a1_g * logK_guess
            K_val = np.exp(logK)
            r = interest(z, K_val)
            w_val = wage(r, z, K_val)
            e = 0 if s % 2 == 0 else 1  # employment indicator: 0 for unemployed, 1 for employed
            y = w_val if e == 1 else 0.0
            
            for i_a in prange(a_size):
                current_a = grid_a[i_a]
                max_val = -1e12
                best_ap = grid_a[0]
                for i_ap in range(a_size):
                    ap = grid_a[i_ap]
                    c = y + (1 + r) * current_a - ap
                    if c <= 0:
                        continue
                    EV = 0.0
                    # Terminal period: no continuation value, so we set continuation = 0
                    val = utility(c)  # + beta * EV (EV = 0)
                    if val > max_val:
                        max_val = val
                        best_ap = ap
                V_new[s, i_a] = max_val
                PF_T[s, i_a] = best_ap
        if np.max(np.abs(V_T - V_new)) < tol:
            break
        V_T = V_new
    # Set terminal condition
    V[T, :, :] = V_T
    PF[T, :, :] = PF_T
    # ------------------------------------------------------------------------
    
    # Backward induction for periods t = T-1,...,0
    for t in range(T-1, -1, -1):
        for s in prange(4):
            # Determine which forecasting rule and TFP to use based on period and state.
            # Shock specification:
            #  - At t==1, use good state parameters for states 2 and 3.
            #  - At t==0 and t>=2, use bad state parameters (and z = zb).
            if t == 1 and (s == 2 or s == 3):
                z = zg
                a0 = a0_g
                a1 = a1_g
            else:
                z = zb
                a0 = a0_b
                a1 = a1_b
            logK = a0 + a1 * logK_guess
            K_val = np.exp(logK)
            r = interest(z, K_val)
            w_val = wage(r, z, K_val)
            # Determine employment status based on state index (even: unemployed, odd: employed)
            e = 0 if s % 2 == 0 else 1
            y = w_val if e == 1 else 0.0
            
            for i_a in prange(a_size):
                current_a = grid_a[i_a]
                max_val = -1e12
                best_ap = grid_a[0]
                # Discrete grid search over next period asset choice
                for i_ap in range(a_size):
                    ap = grid_a[i_ap]
                    c = y + (1 + r) * current_a - ap
                    if c <= 0:
                        continue
                    EV = 0.0
                    # Expected continuation value from period t+1 (using terminal backward functions)
                    for sp in range(4):
                        EV += Pi[s, sp] * V[t+1, sp, i_ap]
                    val = utility(c) + beta * EV
                    if val > max_val:
                        max_val = val
                        best_ap = ap
                V[t, s, i_a] = max_val
                PF[t, s, i_a] = best_ap
    return V, PF

@jit(nopython=True)
def simulate_wealth_path_backward(PF, grid_a, T):
    """
    Simulate the aggregate capital path forward using the policy functions
    obtained from backward induction.
    We assume the initial wealth is 5 for all agents.
    
    The simulation uses:
      - At t = 0, the initial state is the steady state (bad state).
      - At t = 1, use the good state policy function (state index 2).
      - For t >= 2, use the bad state policy function (state index 0).
    """
    wealth = np.ones(num_agents) * 5.0
    K_path = np.empty(T)
    K_path[0] = np.mean(wealth)
    
    # Note: we use the policy functions from period 0 (for t=0),
    # period 1 (for the shock) and then period 2 (for t>=2).
    for t in range(1, T):
        if t == 1:
            # For period 1, use the policy function computed at t=1 for good state (state index 2)
            PF_used = PF[1, 2, :]  # period t=1, state=GU
        else:
            # For t>=2, use the policy function computed at that period for bad state (state index 0)
            PF_used = PF[t, 0, :]
        new_wealth = np.empty_like(wealth)
        for i in range(num_agents):
            idx = np.argmin(np.abs(grid_a - wealth[i]))
            new_wealth[i] = PF_used[idx]
        wealth = new_wealth
        K_path[t] = np.mean(wealth)
    return K_path

# ============================
# 3. Main Execution
# ============================
def run_backward_ks_mit_shock():
    print("Solving the household problem via backward induction...")
    V, PF = solve_backward_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g, T)
    
    print("Simulating wealth path forward using backward-induced policy functions...")
    K_path = simulate_wealth_path_backward(PF, grid_a, T)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_path, marker='o', linestyle='--', label="Aggregate Capital")
    plt.axhline(y=K_path[0], color='r', linestyle='--', label="Steady State")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.title("MIT Shock Transition via Backward Induction (T=20 Equilibrium)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
    
 

if __name__ == "__main__":
    run_backward_ks_mit_shock()
