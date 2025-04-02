# MIT Shock in Full Krusell–Smith Model with Forecast Rule
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit, prange

# ============================
# 1. Model Parameters
# ============================
beta = 0.99
sigma = 1.0
delta = 0.025
alpha = 0.36
ell = 0.3271
ug = 0.04
ub = 0.10
zg = 1.01
zb = 0.99
prob_gg = 0.875
prob_bb = 0.875
prob_gb = 1 - prob_gg
prob_bg = 1 - prob_bb

Pi = np.zeros((4, 4))
Pi[0] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[1] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[2] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]
Pi[3] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]

# Asset grid
a_min, a_max, a_size = 1.0, 20.0, 200
grid_a = np.linspace(a_min, a_max, a_size)

num_agents = 2000
T = 20  # transition horizon

# ============================
# 2. Core Functions
# ============================
@jit(nopython=True)
def utility(c):
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def wage(r, z, K):
    return (1 - alpha) * z * (K**alpha) * ell

@jit(nopython=True)
def interest(z, K):
    return alpha * z * (K**(alpha - 1)) - delta

@jit(nopython=True, parallel=True)
def solve_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g, max_iter=500, tol=1e-6):
    a_size = len(grid_a)
    V = np.zeros((4, a_size))
    V[0, :] = 50   # BU
    V[1, :] = 100  # BE
    V[2, :] = 150  # GU
    V[3, :] = 200  # GE
    PF = np.zeros((4, a_size))

    for _ in range(max_iter):
        V_new = np.copy(V)
        for s in prange(4):
            z = zb if s < 2 else zg
            e = 0 if s % 2 == 0 else 1
            logK = a0_b + a1_b * np.log(40.0) if z == zb else a0_g + a1_g * np.log(40.0)
            K = np.exp(logK)
            r = interest(z, K)
            w = wage(r, z, K)
            y = w if e == 1 else 0.0

            for i_a in prange(a_size):
                current_a = grid_a[i_a]
                max_val = -np.inf
                best_ap = 0.0
                for i_ap in range(a_size):
                    ap = grid_a[i_ap]
                    c = y + (1 + r) * current_a - ap
                    if c <= 0:
                        continue
                    EV = 0.0
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
def simulate_wealth_path(PF, grid_a, initial_state=0, T=20):
    wealth = np.ones(num_agents) * 5.0
    state = np.full(num_agents, initial_state, dtype=np.int64)
    K_path = np.zeros(T)
    K_path[0] = np.mean(wealth)

    for t in range(1, T):
        z = zg if t == 1 else zb
        PF_used = PF[2] if t == 1 else PF[0]
        new_wealth = np.zeros_like(wealth)
        for i in range(num_agents):
            idx = np.argmin(np.abs(grid_a - wealth[i]))
            a_prime = PF_used[idx]
            new_wealth[i] = min(max(a_prime, a_min), a_max)
        wealth = new_wealth
        K_path[t] = np.mean(wealth)

    return K_path

# ============================
# 3. Run KS MIT Shock Simulation
# ============================
def run_ks_mit_shock():
    # Use pre-calibrated forecasting rule coefficients (from previous convergence)
    a0_g, a1_g = 0.0288, 0.9858
    a0_b, a1_b = 0.0082, 0.9944 

    print("Solving value and policy functions...")
    V, PF = solve_value_function(grid_a, beta, a0_b, a1_b, a0_g, a1_g)

    print("Simulating MIT shock transition...")
    K_path = simulate_wealth_path(PF, grid_a, initial_state=0, T=T)

    plt.figure(figsize=(10, 6))
    plt.plot(K_path, marker='o', linestyle='--', label="Aggregate Capital")
    plt.axhline(y=K_path[0], color='r', linestyle='--', label="Steady State")
    plt.xlabel("Time")
    plt.ylabel("Capital")
    plt.title("MIT Shock Transition: z₀=zb, z₁=zg, z₂⁺=zb")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_ks_mit_shock()