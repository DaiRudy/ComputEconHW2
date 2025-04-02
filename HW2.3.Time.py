import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time

# ========== Shared Parameters ==========
beta = 0.99
sigma = 1.0
delta = 0.025
alpha = 0.36
ug = 0.04
a_min, a_max, a_size = 1.0, 20.0, 200
assets = np.linspace(a_min, a_max, a_size)
income = np.array([0.0, 1.0])
p_uu = 0.6
p_eu = (ug - p_uu * ug) / (1 - ug)
Pi_simple = np.array([[p_uu, 1 - p_uu], [p_eu, 1 - p_eu]])
N = 5000
T_transition = 21

# ========== Common Functions ==========
@jit(nopython=True)
def utility(c):
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def manual_clip(x, lower, upper):
    return max(lower, min(x, upper))

@jit(nopython=True)
def solve_value_function_simple(assets, Pi, beta, r, w, income, max_iter=2000, tol=1e-6):
    a_size = len(assets)
    V = np.zeros((2, a_size))
    policy = np.zeros((2, a_size))

    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        for i_e in range(2):
            for i_a in range(a_size):
                current_a = assets[i_a]
                max_value = -np.inf
                best_a_prime = 0.0
                for i_a_prime in range(a_size):
                    a_prime = assets[i_a_prime]
                    c = w * income[i_e] + (1 + r) * current_a - a_prime
                    if c <= 1e-10: continue
                    EV = 0.0
                    for e_next in range(2):
                        EV += Pi[i_e, e_next] * V[e_next, i_a_prime]
                    val = utility(c) + beta * EV
                    if val > max_value:
                        max_value = val
                        best_a_prime = a_prime
                V_new[i_e, i_a] = max_value
                policy[i_e, i_a] = best_a_prime
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new.copy()
    return V, policy

@jit(nopython=True)
def simulate_wealth_distribution(policy, assets, Pi, T=1000, N=5000):
    wealth = np.ones(N) * 5.0
    employment = np.zeros(N, dtype=np.int64)
    for i in range(N):
        employment[i] = 0 if np.random.random() < ug else 1
    for _ in range(T):
        new_wealth = np.zeros_like(wealth)
        new_employment = np.zeros_like(employment)
        for i in range(N):
            e_cur = employment[i]
            a_cur = wealth[i]
            i_a = np.argmin(np.abs(assets - a_cur))
            a_prime = policy[e_cur, i_a]
            new_wealth[i] = manual_clip(a_prime, a_min, a_max)
            new_employment[i] = 0 if np.random.random() < Pi[e_cur, 0] else 1
        wealth = new_wealth
        employment = new_employment
    return wealth, employment

# ========== Model 1: Original MIT Shock ==========
def simulate_MIT_shock_original(z_steady=2.69, z_shock=2.74):
    def capital_equilibrium(z):
        def inner(K_guess):
            r = alpha * z * K_guess**(alpha - 1) - delta
            w = (1 - alpha) * z * K_guess**alpha
            V, policy = solve_value_function_simple(assets, Pi_simple, beta, r, w, income)
            wealth, _ = simulate_wealth_distribution(policy, assets, Pi_simple)
            return np.mean(wealth) - K_guess, V, policy, r, w

        K_min, K_max = 15.0, 25.0
        for _ in range(50):
            K_guess = 0.5 * (K_min + K_max)
            diff, V, policy, r, w = inner(K_guess)
            if abs(diff) < 1e-2:
                return K_guess, V, policy, r, w
            if diff > 0:
                K_min = K_guess
            else:
                K_max = K_guess
        return K_guess, V, policy, r, w

    K_steady, _, policy_steady, _, _ = capital_equilibrium(z_steady)
    _, _, policy_shock, _, _ = capital_equilibrium(z_shock)

    wealth, employment = simulate_wealth_distribution(policy_steady, assets, Pi_simple, T=2000, N=N)
    K_path = [K_steady]

    for t in range(1, T_transition):
        current_policy = policy_shock if t == 1 else policy_steady
        new_wealth = np.zeros_like(wealth)
        new_employment = np.zeros_like(employment)
        for i in range(N):
            i_a = np.argmin(np.abs(assets - wealth[i]))
            new_wealth[i] = manual_clip(current_policy[employment[i], i_a], a_min, a_max)
            new_employment[i] = 0 if np.random.random() < Pi_simple[employment[i], 0] else 1
        wealth = new_wealth
        employment = new_employment
        K_path.append(np.mean(wealth))
    return K_path

# ========== Model 2: Krusell–Smith with Forecast Rule ==========
@jit(nopython=True)
def solve_value_function_ks(grid_a, beta, a0_b, a1_b, a0_g, a1_g, max_iter=500, tol=1e-6):
    a_size = len(grid_a)
    V = np.zeros((4, a_size))
    V[0, :] = 50
    V[1, :] = 100
    V[2, :] = 150
    V[3, :] = 200
    PF = np.zeros((4, a_size))
    Pi = np.array([[0.7875, 0.0875, 0.1125, 0.0125]]*4)  # Simplified Pi

    for _ in range(max_iter):
        V_new = np.copy(V)
        for s in prange(4):
            z = 0.99 if s < 2 else 1.01
            e = 0 if s % 2 == 0 else 1
            logK = a0_b + a1_b * np.log(40.0) if z == 0.99 else a0_g + a1_g * np.log(40.0)
            K = np.exp(logK)
            r = alpha * z * K**(alpha - 1) - delta
            w = (1 - alpha) * z * K**alpha * 0.3271
            y = w if e == 1 else 0.0
            for i_a in prange(a_size):
                max_val = -np.inf
                best_ap = 0.0
                for i_ap in range(a_size):
                    c = y + (1 + r) * grid_a[i_a] - grid_a[i_ap]
                    if c <= 0: continue
                    EV = 0.0
                    for sp in range(4):
                        EV += Pi[s, sp] * V[sp, i_ap]
                    val = utility(c) + beta * EV
                    if val > max_val:
                        max_val = val
                        best_ap = grid_a[i_ap]
                V_new[s, i_a] = max_val
                PF[s, i_a] = best_ap
        if np.max(np.abs(V_new - V)) < tol:
            break
        V = V_new
    return V, PF

@jit(nopython=True)
def simulate_ks_mit_shock(PF, grid_a, T=20):
    wealth = np.ones(N) * 5.0
    K_path = np.zeros(T)
    K_path[0] = np.mean(wealth)

    for t in range(1, T):
        PF_used = PF[2] if t == 1 else PF[0]
        for i in range(N):
            idx = np.argmin(np.abs(grid_a - wealth[i]))
            wealth[i] = manual_clip(PF_used[idx], a_min, a_max)
        K_path[t] = np.mean(wealth)
    return K_path

# ========== Main Comparison ==========
if __name__ == "__main__":
    print("Running original MIT shock model...")
    t1_start = time.time()
    K_path_orig = simulate_MIT_shock_original()
    t1_end = time.time()
    time_orig = t1_end - t1_start
    print(f"Original model time: {time_orig:.4f} seconds")

    print("Running Krusell–Smith model...")
    t2_start = time.time()
    V_ks, PF_ks = solve_value_function_ks(assets, beta, 0.0082, 0.9944, 0.0288, 0.9858)
    K_path_ks = simulate_ks_mit_shock(PF_ks, assets)
    t2_end = time.time()
    time_ks = t2_end - t2_start
    print(f"KS model time: {time_ks:.4f} seconds")

    print(f"\nTime ratio (original / KS): {time_orig / time_ks:.2f}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_path_orig, label="Original MIT Shock")
    plt.plot(K_path_ks, label="Krusell–Smith Forecast Rule", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Aggregate Capital")
    plt.title("MIT Shock Capital Transition Paths")
    plt.legend()
    plt.grid(True)
    plt.show()
