import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit, prange

# ============================
# 1. Model Parameters
# ============================
beta = 0.99       # discount factor
sigma = 1.0       # risk aversion (log utility)
delta = 0.025     # depreciation
alpha = 0.36      # capital share
ell = 0.3271      # effective labor supply
ug = 0.04         # unemployment rate in good times
ub = 0.10         # unemployment rate in bad times
zg = 1.01         # TFP in good times
zb = 0.99         # TFP in bad times
prob_gg = 0.875   # Prob(z'=g | z=g)
prob_bb = 0.875   # Prob(z'=b | z=b)
prob_gb = 1 - prob_gg
prob_bg = 1 - prob_bb

# Full 4x4 transition matrix for (z, e) pairs
# State index: 0=BU, 1=BE, 2=GU, 3=GE
Pi = np.zeros((4, 4))
Pi[0] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]  # BU
Pi[1] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]  # BE
Pi[2] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]  # GU
Pi[3] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]  # GE

# Asset grid
a_min, a_max, a_size = 1.0, 20.0, 200
grid_a = np.linspace(a_min, a_max, a_size)

# Simulation parameters
num_agents = 2000
T_sim = 2000
T_burn = 500

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
    return alpha * z * (K**(alpha-1)) - delta

@jit(nopython=True, parallel=True)
def solve_value_function(grid_a, beta, zg, zb, ug, ub, max_iter=500, tol=1e-6):
    a_size = len(grid_a)
    V = np.zeros((4, a_size))  # 4 states: BU, BE, GU, GE
    V[0, :] = 50   # BU
    V[1, :] = 100  # BE
    V[2, :] = 150  # GU
    V[3, :] = 200  # GE
    PF = np.zeros((4, a_size))
    K_vals = [40.0, 40.0, 40.0, 40.0]

    for _ in range(max_iter):
        V_new = np.copy(V)
        for s in prange(4):
            z = zb if s < 2 else zg
            e = 0 if s % 2 == 0 else 1
            K = K_vals[s]
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
                    if c <= 0: continue
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

# ============================
# 3. Simulation + Forecasting Rule
# ============================
def simulate_economy(PF, max_iter=10):
    a0_g, a1_g = 0.1, 0.95
    a0_b, a1_b = 0.1, 0.95

    for iter in range(max_iter):
        K_hist = []
        z_hist = []
        a = np.ones(num_agents) * 5.0
        s = 3  # start in GE

        for t in range(T_sim + T_burn):
            s_next = np.random.choice(4, p=Pi[s])
            z = zb if s_next < 2 else zg
            r = interest(z, np.mean(a))
            w = wage(r, z, np.mean(a))

            a_next = np.zeros_like(a)
            for i in prange(num_agents):
                idx = np.argmin(np.abs(grid_a - a[i]))
                a_next[i] = PF[s_next, idx]

            if t >= T_burn:
                K_hist.append(np.mean(a))
                z_hist.append(s_next)

            a = np.clip(a_next, a_min, a_max)
            s = s_next

        logK = np.log(K_hist[:-1])
        logKp = np.log(K_hist[1:])
        z_hist = np.array(z_hist[:-1])

        mask_g = (z_hist >= 2)
        res_g = sm.OLS(logKp[mask_g], sm.add_constant(logK[mask_g])).fit()
        res_b = sm.OLS(logKp[~mask_g], sm.add_constant(logK[~mask_g])).fit()

        a0_g, a1_g = res_g.params
        a0_b, a1_b = res_b.params

        print(f"Iter {iter+1}:")
        print(f"  Good state: logK' = {a0_g:.4f} + {a1_g:.4f} logK (R²={res_g.rsquared:.4f})")
        print(f"  Bad state:  logK' = {a0_b:.4f} + {a1_b:.4f} logK (R²={res_b.rsquared:.4f})\n")

    return a0_g, a1_g, a0_b, a1_b, PF

# ============================
# 4. Main Execution
# ============================
if __name__ == "__main__":
    print("Solving household problem...")
    V, PF = solve_value_function(grid_a, beta, zg, zb, ug, ub)

    print("\nCalibrating forecasting rule...")
    a0_g, a1_g, a0_b, a1_b, PF = simulate_economy(PF)

    plt.figure(figsize=(10, 6))
    for s, label in zip(range(4), ['BU', 'BE', 'GU', 'GE']):
        plt.plot(grid_a, V[s], label=label)
    plt.title("Value Functions")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    for s, label in zip(range(4), ['BU', 'BE', 'GU', 'GE']):
        plt.plot(grid_a, PF[s], label=label)
    plt.plot(grid_a, grid_a, 'k--', alpha=0.5)
    plt.title("Policy Functions")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============================
    # 5. Wealth Distribution by Type
    # ============================
    print("\nSimulating wealth distribution by state...")
    a = np.ones(num_agents) * 5.0
    s = 3  # start in GE
    data_by_state = {0: [], 1: [], 2: [], 3: []}  # BU, BE, GU, GE

    for t in range(3000):
        s_next = np.random.choice(4, p=Pi[s])
        z = zb if s_next < 2 else zg
        r = interest(z, np.mean(a))
        w = wage(r, z, np.mean(a))

        a_next = np.zeros_like(a)
        for i in range(num_agents):
            idx = np.argmin(np.abs(grid_a - a[i]))
            a_next[i] = PF[s_next, idx]
            if t >= 2900:
                data_by_state[s_next].append(a[i])

        a = np.clip(a_next, a_min, a_max)
        s = s_next

    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for s in range(4):
        plt.hist(data_by_state[s], bins=50, alpha=0.6, density=True, label=['BU', 'BE', 'GU', 'GE'][s], color=colors[s])
    plt.xlabel("Wealth")
    plt.ylabel("Density")
    plt.title("Wealth Distribution by Type (BU, BE, GU, GE)")
    plt.legend()
    plt.grid(True)
    plt.show()