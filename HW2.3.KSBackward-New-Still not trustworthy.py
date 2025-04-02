import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit, prange

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
    # Utility function (log utility)
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def wage(r, z, K):
    # Wage function based on production function
    return (1 - alpha) * z * (K**alpha) * ell

@jit(nopython=True)
def interest(z, K):
    # Interest rate function derived from production function
    return alpha * z * (K**(alpha - 1)) - delta

@jit(nopython=True, parallel=True)
def solve_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g, max_iter=500, tol=1e-6):
    """
    Solves the household's problem given forecasting rule coefficients.
    For each state, the function computes a forecast of aggregate capital
    using the rule:
      log(K_forecast) = a0 + a1 * log(K_guess)
    where K_guess is initially fixed at 40.
    """
    a_size = len(grid_a)
    V = np.zeros((4, a_size))
    # Initial guess for the value functions for each state
    V[0, :] = 50   # BU
    V[1, :] = 100  # BE
    V[2, :] = 150  # GU
    V[3, :] = 200  # GE
    PF = np.zeros((4, a_size))
    
    # Fixed initial guess for aggregate capital
    K_guess = 40.0
    logK_guess = np.log(K_guess)
    
    for it in range(max_iter):
        V_new = np.copy(V)
        for s in prange(4):
            # For states 0 and 1, use bad state parameters; for 2 and 3, use good state parameters
            if s < 2:
                z = zb
                # Use bad state forecasting rule for capital
                logK = a0_b + a1_b * logK_guess
            else:
                z = zg
                # Use good state forecasting rule for capital
                logK = a0_g + a1_g * logK_guess
            K = np.exp(logK)
            r = interest(z, K)
            w_val = wage(r, z, K)
            # Determine employment status: even index = unemployed (0), odd index = employed (1)
            e = 0 if s % 2 == 0 else 1
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
                    # Expected value from transition across states
                    for sp in range(4):
                        EV += Pi[s, sp] * V[sp, i_ap]
                    val = utility(c) + beta * EV
                    if val > max_val:
                        max_val = val
                        best_ap = ap
                V_new[s, i_a] = max_val
                PF[s, i_a] = best_ap
        if np.max(np.abs(V - V_new)) < tol:
            break
        V = V_new
    return V, PF

@jit(nopython=True)
def simulate_mit_shock(PF, grid_a, T):
    """
    Simulates the economy's transition under a MIT shock.
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
            z = zg
            PF_used = PF[2]  # GU state
        else:
            # From period 2 onward, economy is in the bad state
            z = zb
            PF_used = PF[0]  # BU state
        new_wealth = np.zeros_like(wealth)
        for i in range(num_agents):
            # Find the closest asset grid point to current wealth
            idx = np.argmin(np.abs(grid_a - wealth[i]))
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
    At each iteration, the household problem is solved with the current rule,
    the MIT shock transition is simulated, and then the bad state forecasting
    rule is updated using an OLS regression on the log aggregate capital path.
    Note: The good state parameters are not updated since only one period of
    good state data is available.
    """
    global a0_g, a1_g, a0_b, a1_b
    for it in range(max_iter_forecast):
        print(f"Iteration {it+1} / {max_iter_forecast}")
        # Solve the household's problem with current forecasting rules
        V, PF = solve_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g)
        # Simulate the MIT shock transition path
        K_path = simulate_mit_shock(PF, grid_a, T_sim)
        print("  Simulated K path:", np.round(K_path, 3))
        # Update the bad state forecasting rule using OLS regression
        # Using all periods from t=0 to t=T_sim-1 (after shock, the economy is in bad state)
        logK = np.log(K_path[:-1])
        logKp = np.log(K_path[1:])
        X = np.column_stack((np.ones(logK.shape[0]), logK))
        beta_hat = np.linalg.lstsq(X, logKp, rcond=None)[0]
        a0_b, a1_b = beta_hat[0], beta_hat[1]
        print(f"  Updated bad state rule: logK' = {a0_b:.4f} + {a1_b:.4f} logK")
        # The good state rule remains unchanged
        print("-"*40)
    return a0_g, a1_g, a0_b, a1_b, PF, K_path

# ============================
# 4. Main Execution
# ============================
if __name__ == "__main__":
    print("Starting forecasting rule iteration and solving household problem...")
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

    # Plot the value functions for each state
    plt.figure(figsize=(10, 6))
    for s, label in zip(range(4), ['BU', 'BE', 'GU', 'GE']):
        plt.plot(grid_a, V[s], label=label)
    plt.title("Value Functions")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the policy functions for each state
    plt.figure(figsize=(10, 6))
    for s, label in zip(range(4), ['BU', 'BE', 'GU', 'GE']):
        plt.plot(grid_a, PF[s], label=label)
    plt.plot(grid_a, grid_a, 'k--', alpha=0.5)
    plt.title("Policy Functions")
    plt.legend()
    plt.grid(True)
    plt.show()

