import numpy as np  
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit

# ============================
# 1. Model Parameters
# ============================
beta = 0.99       # Discount factor
sigma = 1.0       # Risk aversion (log utility)
delta = 0.025     # Depreciation rate
alpha = 0.36      # Capital share
ell = 0.3271      # Effective labor supply
ug = 0.04         # Unemployment rate in good times
ub = 0.10         # Unemployment rate in bad times
zg = 1.01         # Total Factor Productivity (TFP) in good times
zb = 0.99         # TFP in bad times

# Transition probabilities for (z, e) pairs
# State indices: 0=BU, 1=BE, 2=GU, 3=GE
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
T_sim = 20              # Transition horizon for the MIT shock simulation
max_iter_forecast = 10  # Number of iterations for the forecasting rule update

# Initial forecasting rule coefficients
# For good state
a0_g, a1_g = 0.0288, 0.9858  
# For bad state
a0_b, a1_b = 0.0082, 0.9944

# ============================
# 2. Core Functions
# ============================
@jit(nopython=True)
def utility(c):
    # Log utility function; returns a large negative number if consumption is non-positive
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def wage(r, z, K):
    # Wage function based on the production function
    return (1 - alpha) * z * (K**alpha) * ell

@jit(nopython=True)
def interest(z, K):
    # Interest rate function derived from the production function
    return alpha * z * (K**(alpha - 1)) - delta

# -----------------------------------------------
# Forward Iteration on Policy Function (instead of backward iteration)
# -----------------------------------------------
@jit(nopython=True)
def solve_policy_function_forward(grid_a, beta, a0_b, a1_b, a0_g, a1_g, max_iter=500, tol=1e-6):
    """
    Solves the household's problem via forward iteration on the policy function.
    Instead of using the backward Bellman equation, this function simulates
    forward choices over the asset grid and updates the policy function (PF)
    until convergence.
    
    For each state, the forecasting rule is used to compute aggregate capital,
    which in turn determines wages and interest rates. Then, for each asset level,
    the optimal next-period asset is chosen by comparing the current period utility
    plus a continuation value approximated by the current policy function.
    """
    a_size = len(grid_a)
    # Initialize policy function PF for each state as an identity mapping
    PF = np.empty((4, a_size))
    for s in range(4):
        for i in range(a_size):
            PF[s, i] = grid_a[i]
    
    # Fixed initial guess for aggregate capital
    K_guess = 40.0
    logK_guess = np.log(K_guess)
    
    for it in range(max_iter):
        PF_new = np.empty_like(PF)
        max_diff = 0.0
        for s in range(4):
            # Select TFP and forecasting rule parameters based on state
            if s < 2:
                z = zb
                logK = a0_b + a1_b * logK_guess
            else:
                z = zg
                logK = a0_g + a1_g * logK_guess
            K = np.exp(logK)
            r = interest(z, K)
            w_val = wage(r, z, K)
            # Determine employment status: even index = unemployed (0), odd index = employed (1)
            e = 0 if (s % 2) == 0 else 1
            y = w_val if e == 1 else 0.0
            
            for i_a in range(a_size):
                current_a = grid_a[i_a]
                best_val = -1e12
                best_ap = grid_a[0]
                # Forward iteration: test all possible choices for next-period assets
                for i_ap in range(a_size):
                    ap = grid_a[i_ap]
                    c = y + (1 + r) * current_a - ap
                    if c <= 0:
                        continue
                    # Approximate continuation utility using the current policy function
                    cont_util = 0.0
                    # Average over possible future states weighted by transition probabilities
                    for sp in range(4):
                        # Use utility of consumption after following the current policy as proxy
                        cont_util += Pi[s, sp] * utility(y + (1 + r) * ap - PF[s, i_ap])
                    total_val = utility(c) + beta * cont_util
                    if total_val > best_val:
                        best_val = total_val
                        best_ap = ap
                PF_new[s, i_a] = best_ap
                diff_val = np.abs(best_ap - PF[s, i_a])
                if diff_val > max_diff:
                    max_diff = diff_val
        # Update the policy function with the new iteration results
        for s in range(4):
            for i in range(a_size):
                PF[s, i] = PF_new[s, i]
        if max_diff < tol:
            break
    return PF

@jit(nopython=True)
def simulate_mit_shock(PF, grid_a, T):
    """
    Simulates the economy's transition under a MIT shock using the policy functions.
    - t = 0: Steady state (initial condition)
    - t = 1: A one-period good state shock is used to trigger adjustment (uses GU policy function)
    - t >= 2: The economy is entirely in the bad state (uses BU policy function)
    """
    wealth = np.ones(num_agents) * 5.0
    K_path = np.zeros(T)
    K_path[0] = np.mean(wealth)
    
    for t in range(1, T):
        if t == 1:
            # In period 1, use good state policy function for adjustment
            PF_used = PF[2]  # GU state
        else:
            # From period 2 onward, the economy is in the bad state
            PF_used = PF[0]  # BU state
        new_wealth = np.zeros_like(wealth)
        for i in range(num_agents):
            # Find the closest asset grid point to current wealth and update wealth according to policy function
            diff = np.abs(grid_a - wealth[i])
            idx = 0
            min_diff = diff[0]
            for j in range(len(diff)):
                if diff[j] < min_diff:
                    min_diff = diff[j]
                    idx = j
            new_wealth[i] = PF_used[idx]
        wealth = new_wealth
        K_path[t] = np.mean(wealth)
    return K_path

# ============================
# 3. Forecasting Rule Iteration
# ============================
def iterate_forecasting_rule():
    """
    Iteratively updates the forecasting rule.
    In each iteration, the household problem is solved using forward iteration on the policy function,
    the MIT shock transition is simulated, and then the bad state forecasting rule is updated using an OLS regression
    on the log aggregate capital path. (Note: The good state parameters remain unchanged.)
    """
    global a0_g, a1_g, a0_b, a1_b
    for it in range(max_iter_forecast):
        print(f"Iteration {it+1} / {max_iter_forecast}")
        # Solve the household's problem with current forecasting rules using forward iteration
        PF = solve_policy_function_forward(grid_a, beta, a0_b, a1_b, a0_g, a1_g)
        # Simulate the MIT shock transition path using the obtained policy function
        K_path = simulate_mit_shock(PF, grid_a, T_sim)
        print("  Simulated K path:", np.round(K_path, 3))
        # Update the bad state forecasting rule using OLS regression on the simulated capital path
        logK = np.log(K_path[:-1])
        logKp = np.log(K_path[1:])
        X = np.column_stack((np.ones(logK.shape[0]), logK))
        beta_hat = np.linalg.lstsq(X, logKp, rcond=None)[0]
        a0_b, a1_b = beta_hat[0], beta_hat[1]
        print(f"  Updated bad state rule: logK' = {a0_b:.4f} + {a1_b:.4f} logK")
        print("-"*40)
    return a0_g, a1_g, a0_b, a1_b, PF, K_path

# ============================
# 4. Main Execution
# ============================
if __name__ == "__main__":
    print("Starting forecasting rule iteration with forward iteration on policy function...")
    a0_g, a1_g, a0_b, a1_b, PF, K_path = iterate_forecasting_rule()

    # Plot the final MIT shock transition path
    plt.figure(figsize=(10, 6))
    plt.plot(K_path, marker='o', linestyle='--', label="Aggregate Capital")
    plt.axhline(y=K_path[0], color='r', linestyle='--', label="Steady State")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.title("MIT Shock Transition (t=1: Good state; t>=2: Bad state)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
