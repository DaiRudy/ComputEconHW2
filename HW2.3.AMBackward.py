# Updated MIT shock code with backward iteration for MIT shock
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# =====================
# 1. Model Parameters
# =====================
beta = 0.99
sigma = 1.0
delta = 0.025
alpha = 0.36
z_steady = 2.69
ug = 0.04

# Asset grid
a_min, a_max, a_size = 1.0, 20.0, 200
assets = np.linspace(a_min, a_max, a_size)

# Income by employment state: 0 = unemployed, 1 = employed
income = np.array([0.0, 1.0])

# Employment transition matrix
p_uu = 0.6
p_eu = (ug - p_uu * ug) / (1 - ug)
Pi = np.array([
    [p_uu, 1 - p_uu],
    [p_eu, 1 - p_eu]
])

# =====================
# 2. Core Functions
# =====================
@jit(nopython=True)
def utility(c):
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def solve_value_function(assets, Pi, beta, r, w, income, V_init=None, max_iter=2000, tol=1e-6):
    a_size = len(assets)
    V = np.zeros((2, a_size)) if V_init is None else V_init.copy()
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
def simulate_one_step(policy, assets, Pi, wealth, employment):
    N = len(wealth)
    new_wealth = np.zeros_like(wealth)
    new_employment = np.zeros_like(employment)
    for i in range(N):
        e_cur = employment[i]
        a_cur = wealth[i]
        i_a = np.argmin(np.abs(assets - a_cur))
        a_prime = policy[e_cur, i_a]
        new_wealth[i] = manual_clip(a_prime, a_min, a_max)
        new_employment[i] = 0 if np.random.random() < Pi[e_cur, 0] else 1
    return new_wealth, new_employment

@jit(nopython=True)
def simulate_wealth_distribution(policy, assets, Pi, T=1000, N=5000):
    wealth = np.ones(N) * 5.0
    employment = np.zeros(N, dtype=np.int64)
    for i in range(N):
        employment[i] = 0 if np.random.random() < ug else 1

    for _ in range(T):
        wealth, employment = simulate_one_step(policy, assets, Pi, wealth, employment)
    return wealth, employment

# =====================
# 3. MIT Shock Simulation (Backward Iteration)
# =====================
def simulate_MIT_shock(assets, Pi, beta, income, T_transition=21, N=5000):
    def capital_equilibrium(z):
        def inner(K_guess):
            r = alpha * z * K_guess**(alpha - 1) - delta
            w = (1 - alpha) * z * K_guess**alpha
            V, policy = solve_value_function(assets, Pi, beta, r, w, income)
            wealth, _ = simulate_wealth_distribution(policy, assets, Pi, T=1000, N=N)
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

    print("Solving for steady state...")
    K_steady, V_steady, policy_steady, r_steady, w_steady = capital_equilibrium(z_steady)

    print("Solving for shock period policy...")
    z_shock = 2.74

    # Create T-t policies via backward induction
    z_path = [z_shock] + [z_steady] * (T_transition - 1)
    policies = [None] * T_transition
    V_next = V_steady.copy()
    for t in reversed(range(T_transition)):
        z_t = z_path[t]
        K_t = K_steady  # assume K does not move during backward iteration
        r_t = alpha * z_t * K_t**(alpha - 1) - delta
        w_t = (1 - alpha) * z_t * K_t**alpha
        V, policy = solve_value_function(assets, Pi, beta, r_t, w_t, income, V_init=V_next)
        policies[t] = policy
        V_next = V

    print("Simulating transition path forward...")
    wealth, employment = simulate_wealth_distribution(policy_steady, assets, Pi, T=2000, N=N)
    K_path = [np.mean(wealth)]

    for t in range(T_transition):
        policy = policies[t]
        wealth, employment = simulate_one_step(policy, assets, Pi, wealth, employment)
        K_path.append(np.mean(wealth))

    return K_path, K_steady

# =====================
# 4. Run and Plot
# =====================
if __name__ == "__main__":
    K_path, K_steady = simulate_MIT_shock(assets, Pi, beta, income, T_transition=21)
    plt.figure(figsize=(10, 6))
    plt.plot(K_path, marker='o', linestyle='--', label="Capital Path")
    plt.axhline(y=K_steady, color='r', linestyle='--', label="Steady State K")
    plt.xlabel("Time")
    plt.ylabel("Aggregate Capital")
    plt.title("Capital Transition Path after MIT Shock (Backward Iteration)")
    plt.grid(True)
    plt.legend()
    plt.show()
