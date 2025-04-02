import numpy as np
import matplotlib.pyplot as plt
from numba import jit

################################################
# 1. Model Parameters
################################################
beta = 0.99
sigma = 1.0
delta = 0.025
alpha = 0.36 #Based on KS 1998
z_fixed = 2.69 # e^0.99
ug = 0.1

# Asset grid
a_min = 1.0
a_max = 20.0
a_size = 200
assets = np.linspace(a_min, a_max, a_size)

# Income (0 = unemployed, 1 = employed)
income = np.array([0, 1.0])

# Transition matrix for employment
p_uu = 0.6
p_eu = (ug - p_uu * ug) / (1 - ug)
Pi = np.array([
    [p_uu, 1 - p_uu],
    [p_eu, 1 - p_eu]
])

@jit(nopython=True)
def utility(c):
    return np.log(c) if c > 1e-10 else -np.inf

@jit(nopython=True)
def solve_value_function(assets, Pi, beta, r, w, max_iter=2000, tol=1e-6):
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
                    if c <= 1e-10:
                        continue
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
def manual_clip(x, lower, upper):
    return max(lower, min(x, upper))

@jit(nopython=True)
def simulate_wealth_distribution(policy, assets, Pi, T=2000, N=5000):
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
            u2 = np.random.random()
            new_employment[i] = 0 if u2 < Pi[e_cur, 0] else 1
        wealth = new_wealth
        employment = new_employment
    return wealth, employment

def find_equilibrium_K(K_min=15.0, K_max=25.0, tol=1e-2, max_iter=30):
    for i in range(max_iter):
        K_guess = 0.5 * (K_min + K_max)
        r = alpha * z_fixed * K_guess**(alpha - 1) - delta
        w = (1 - alpha) * z_fixed * K_guess**alpha
        print(f"[Iter {i+1}] K_guess = {K_guess:.4f}, r = {r:.4f}, w = {w:.4f}")

        V, policy = solve_value_function(assets, Pi, beta, r, w)
        final_wealth, final_employment = simulate_wealth_distribution(policy, assets, Pi, T=1000)
        K_implied = np.mean(final_wealth)
        print(f"   → K_implied = {K_implied:.4f}")

        if abs(K_implied - K_guess) < tol:
            print(f"✅ Found SREE: K = {K_implied:.4f}")
            return K_guess, r, w, V, policy, final_wealth, final_employment

        if K_implied > K_guess:
            K_min = K_guess
        else:
            K_max = K_guess

    print("❌ Did not converge.")
    return K_guess, r, w, V, policy, final_wealth, final_employment

# ==== Run equilibrium search ====
K_star, r_star, w_star, V, policy, final_wealth, final_employment = find_equilibrium_K()

# ==== Plot Value Functions ====
plt.figure(figsize=(8, 6))
plt.plot(assets, V[0], label="Unemployed")
plt.plot(assets, V[1], label="Employed")
plt.xlabel("Wealth")
plt.ylabel("Value Function")
plt.title("Value Function by Employment Status")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ==== Plot Policy Functions ====
plt.figure(figsize=(8, 6))
plt.plot(assets, policy[0], label="Unemployed")
plt.plot(assets, policy[1], label="Employed")
plt.plot(assets, assets, 'k--', alpha=0.5, label="45-degree line")
plt.xlabel("Current Wealth")
plt.ylabel("Next Period Wealth")
plt.title("Policy Function by Employment Status")
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# ==== Plot Wealth Distribution by Employment ====
plt.figure(figsize=(8, 6))
plt.hist(final_wealth[final_employment == 0], bins=50, alpha=0.6, density=True, label="Unemployed")
plt.hist(final_wealth[final_employment == 1], bins=50, alpha=0.6, density=True, label="Employed")
plt.xlabel("Wealth")
plt.ylabel("Density")
plt.title("Steady State Wealth Distribution by Employment Status")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
