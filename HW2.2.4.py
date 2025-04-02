import numpy as np
import time
from numba import jit, prange

# === 1. Model parameters ===
beta = 0.99
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

# Transition matrix for 4 states: BU, BE, GU, GE
Pi = np.zeros((4, 4))
Pi[0] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[1] = [prob_bb*(1-ub), prob_bb*ub, prob_bg*(1-ug), prob_bg*ug]
Pi[2] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]
Pi[3] = [prob_gb*(1-ub), prob_gb*ub, prob_gg*(1-ug), prob_gg*ug]

# Asset grid
a_min, a_max, a_size = 1.0, 20.0, 100  # smaller for speed
grid_a = np.linspace(a_min, a_max, a_size)

# === 2. Utilities and Prices ===
@jit(nopython=True)
def utility(c):
    return np.log(c) if c > 1e-10 else -1e12

@jit(nopython=True)
def wage(r, z, K):
    return (1 - alpha) * z * (K**alpha) * ell

@jit(nopython=True)
def interest(z, K):
    return alpha * z * (K**(alpha-1)) - delta

# === 3. Parallel version ===
@jit(nopython=True, parallel=True)
def solve_value_function_parallel(grid_a, beta, zg, zb, ug, ub, max_iter=100, tol=1e-6):
    a_size = len(grid_a)
    V = np.zeros((4, a_size))
    V[0, :] = 50
    V[1, :] = 100
    V[2, :] = 150
    V[3, :] = 200
    PF = np.zeros((4, a_size))
    K_vals = [40.0] * 4

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

# === 4. Serial version ===
@jit(nopython=True)
def solve_value_function_serial(grid_a, beta, zg, zb, ug, ub, max_iter=100, tol=1e-6):
    a_size = len(grid_a)
    V = np.zeros((4, a_size))
    V[0, :] = 50
    V[1, :] = 100
    V[2, :] = 150
    V[3, :] = 200
    PF = np.zeros((4, a_size))
    K_vals = [40.0] * 4

    for _ in range(max_iter):
        V_new = np.copy(V)
        for s in range(4):
            z = zb if s < 2 else zg
            e = 0 if s % 2 == 0 else 1
            K = K_vals[s]
            r = interest(z, K)
            w = wage(r, z, K)
            y = w if e == 1 else 0.0

            for i_a in range(a_size):
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

# === 5. Run with warm-up first ===
print("Warming up JIT...")
solve_value_function_parallel(grid_a, beta, zg, zb, ug, ub)
solve_value_function_serial(grid_a, beta, zg, zb, ug, ub)

# === 6. Time the parallel version ===
start_parallel = time.time()
solve_value_function_parallel(grid_a, beta, zg, zb, ug, ub)
end_parallel = time.time()

# === 7. Time the serial version ===
start_serial = time.time()
solve_value_function_serial(grid_a, beta, zg, zb, ug, ub)
end_serial = time.time()

# === 8. Output results ===
time_parallel = end_parallel - start_parallel
time_serial = end_serial - start_serial
speed_ratio = time_serial / time_parallel

print(f"\nParallel time: {time_parallel:.4f} seconds")
print(f"Serial time:   {time_serial:.4f} seconds")
print(f"Speedup ratio: {speed_ratio:.2f}x faster using parallelization 🎉")
