"""
netdyn.py
==========================
Network Dynamics Simulation Toolkit
-----------------------------------

Example
-------
>>> from netdyn import sis_dynamics
>>> import networkx as nx
>>> A = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.2))
>>> X = sis_dynamics(A, beta=0.4, gamma=0.2, T_steps=200)
>>> print(X.shape)
"""

import numpy as np

__all__ = [
    "sis_dynamics",
    "lv_dynamics",
    "mp_dynamics",
    "mm_dynamics",
    "kuramoto_dynamics",
    "wc_dynamics",
]


# ==============================================================
# SIS Model
# ==============================================================
def sis_dynamics(adj_matrix, beta=0.5, gamma=0.2, T_steps=1000, dt=0.1,
                 init_state=None, seed=None):
    """
    Equation:
        dI_i/dt = β (1 - I_i) Σ_j A_ij I_j - γ I_i
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    I = np.random.rand(N) if init_state is None else init_state.copy()
    I = np.clip(I, 0, 1)
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        infection = beta * (1 - I) * (adj_matrix @ I)
        recovery = -gamma * I
        I += dt * (infection + recovery)
        I = np.clip(I, 0, 1)
        states[t] = I

    return states


# ==============================================================
# Lotka–Volterra Model
# ==============================================================
def lv_dynamics(adj_matrix, alpha=1.0, theta=1.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    Generalized Lotka–Volterra (GLV) model.

    Equation:
        dx_i/dt = x_i(α - θ x_i) - x_i Σ_j A_ij x_j
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    x = np.random.rand(N) if init_state is None else init_state.copy()
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        growth = alpha * x - theta * x**2
        interaction = -(x * (adj_matrix @ x))
        x += dt * (growth + interaction)
        x = np.clip(x, 0, None)
        states[t] = x

    return states


# ==============================================================
# Mutualistic Population Model
# ==============================================================
def mp_dynamics(adj_matrix, alpha=1.0, theta=1.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    Mutualistic population dynamics.

    Equation:
        dx_i/dt = x_i(α - θ x_i) + x_i Σ_j A_ij * x_j^2 / (1 + x_j^2)
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    x = np.random.rand(N) if init_state is None else init_state.copy()
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        mutual = adj_matrix @ (x**2 / (1 + x**2))
        dx = dt * (x * (alpha - theta * x + mutual))
        x += dx
        x = np.clip(x, 0, None)
        states[t] = x

    return states


# ==============================================================
# Michaelis–Menten Model
# ==============================================================
def mm_dynamics(adj_matrix, h=2.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    Michaelis–Menten–type regulatory dynamics.

    Equation:
        dx_i/dt = -x_i + Σ_j A_ij * (x_j^h / (1 + x_j^h))
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    x = np.random.rand(N) if init_state is None else init_state.copy()
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        interaction = adj_matrix @ (x**h / (1 + x**h))
        x += dt * (-x + interaction)
        x = np.clip(x, 0, None)
        states[t] = x

    return states


# ==============================================================
# Kuramoto Model
# ==============================================================
def kuramoto_dynamics(adj_matrix, omega=None, T_steps=1000, dt=0.05,
                      init_state=None, seed=None):
    """
    Kuramoto phase oscillator network.

    Equation:
        dθ_i/dt = ω_i + Σ_j A_ij sin(θ_j - θ_i)
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    if omega is None:
        omega = np.random.normal(0, 1, N)
    theta = 2 * np.pi * np.random.rand(N) if init_state is None else init_state.copy()
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        coupling = np.sum(adj_matrix * np.sin(theta[np.newaxis, :] - theta[:, np.newaxis]), axis=1)
        theta += dt * (omega + coupling)
        states[t] = np.mod(theta, 2 * np.pi)

    return states


# ==============================================================
# Wilson–Cowan Model
# ==============================================================
def wc_dynamics(adj_matrix, tau=1.0, mu=0.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    Wilson–Cowan neural firing model.

    Equation:
        dx_i/dt = -x_i + Σ_j A_ij * [1 / (1 + exp(-τ (x_j - μ)))]
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    x = np.random.rand(N) if init_state is None else init_state.copy()
    states = np.zeros((T_steps, N))

    for t in range(T_steps):
        sigmoid = 1 / (1 + np.exp(-tau * (x - mu)))
        interaction = adj_matrix @ sigmoid
        x += dt * (-x + interaction)
        x = np.clip(x, 0, None)
        states[t] = x

    return states


# ==============================================================
# Quick Test (only runs when executed directly)
# ==============================================================
if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    A = nx.to_numpy_array(nx.erdos_renyi_graph(20, 0.2, seed=42))

    print("Running SIS model for demo...")
    # X = sis_dynamics(A, beta=0.4, gamma=0.2, T_steps=300, dt=0.1)
    X = lv_dynamics(A, alpha=1.0, theta=1.0, T_steps=500, dt=0.1)
    print("Simulation complete. Shape:", X.shape)

    plt.plot(X[:, :])
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.title("SIS model example")
    plt.show()
