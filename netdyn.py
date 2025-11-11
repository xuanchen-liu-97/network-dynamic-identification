"""
netdyn.py
==========================
Network Dynamics Simulation Toolkit
-----------------------------------
(版本 2: 使用 Scipy ODE 求解器)

Example
-------
>>> from netdyn import sis_dynamics
>>> import networkx as nx
>>> A = nx.to_numpy_array(nx.erdos_renyi_graph(10, 0.2))
>>> X = sis_dynamics(A, beta=0.4, gamma=0.2, T_steps=200)
>>> print(X.shape)
"""

import numpy as np
from scipy.integrate import solve_ivp

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
    使用 solve_ivp 运行 SIS 模拟。

    Equation:
        dI_i/dt = β (1 - I_i) Σ_j A_ij I_j - γ I_i
        
    注意: beta 和 gamma 可以是标量，也可以是 N 维数组。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    
    # 设置初始状态
    if init_state is None:
        I = np.random.rand(N)
    else:
        I = init_state.copy()
    I = np.clip(I, 0, 1) # 裁剪初始状态

    # 定义 ODE 方程
    def _sis_ode(t, I):
        infection = beta * (1 - I) * (adj_matrix @ I)
        recovery = -gamma * I
        return infection + recovery

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)

    # 运行求解器
    sol = solve_ivp(
        _sis_ode,
        t_span,
        I,
        t_eval=t_eval,
        method='LSODA' # 自动切换的优秀求解器
    )

    if not sol.success:
        print(f"警告: SIS 模拟求解器未能成功: {sol.message}")

    # 格式化输出: 转置为 (T_steps, N) 并裁剪
    states = sol.y.T
    return np.clip(states, 0, 1)


# ==============================================================
# Lotka–Volterra Model
# ==============================================================
def lv_dynamics(adj_matrix, alpha=1.0, theta=1.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    使用 solve_ivp 运行广义 Lotka-Volterra (GLV) 模拟。

    Equation:
        dx_i/dt = x_i(α_i - θ_i x_i) - x_i Σ_j A_ij x_j
        
    注意: alpha 和 theta 可以是标量，也可以是 N 维数组。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]

    if init_state is None:
        x = np.random.rand(N)
    else:
        x = init_state.copy()
    x = np.clip(x, 0, None) # 确保初始种群非负

    # 定义 ODE 方程
    def _lv_ode(t, x):
        growth = alpha * x - theta * x**2
        interaction = -(x * (adj_matrix @ x))
        return growth + interaction

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)
    
    # 运行求解器
    sol = solve_ivp(
        _lv_ode,
        t_span,
        x,
        t_eval=t_eval,
        method='LSODA'
    )

    if not sol.success:
        print(f"警告: LV 模拟求解器未能成功: {sol.message}")
        
    # 格式化输出: 转置为 (T_steps, N) 并裁剪
    states = sol.y.T
    return np.clip(states, 0, None)


# ==============================================================
# Mutualistic Population Model
# ==============================================================
def mp_dynamics(adj_matrix, alpha=1.0, theta=1.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    使用 solve_ivp 运行互惠种群 (MP) 模拟。

    Equation:
        dx_i/dt = x_i(α_i - θ_i x_i) + x_i Σ_j A_ij * x_j^2 / (1 + x_j^2)
        
    注意: alpha 和 theta 可以是标量，也可以是 N 维数组。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    
    if init_state is None:
        x = np.random.rand(N)
    else:
        x = init_state.copy()
    x = np.clip(x, 0, None) # 确保初始种群非负

    # 定义 ODE 方程
    def _mp_ode(t, x):
        x_j_term = x**2 / (1 + x**2)
        mutual = adj_matrix @ x_j_term
        dx = x * (alpha - theta * x + mutual)
        return dx

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)
    
    # 运行求解器
    sol = solve_ivp(
        _mp_ode,
        t_span,
        x,
        t_eval=t_eval,
        method='LSODA'
    )

    if not sol.success:
        print(f"警告: MP 模拟求解器未能成功: {sol.message}")
        
    # 格式化输出: 转置为 (T_steps, N) 并裁剪
    states = sol.y.T
    return np.clip(states, 0, None)


# ==============================================================
# Michaelis–Menten Model
# ==============================================================
def mm_dynamics(adj_matrix, h=2.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    使用 solve_ivp 运行 Michaelis–Menten (MM) 模拟。

    Equation:
        dx_i/dt = -x_i + Σ_j A_ij * (x_j^h / (1 + x_j^h))
        
    注意: h 是一个标量。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    
    if init_state is None:
        x = np.random.rand(N)
    else:
        x = init_state.copy()
    x = np.clip(x, 0, None) # 确保初始表达非负

    # 定义 ODE 方程
    def _mm_ode(t, x):
        interaction = adj_matrix @ (x**h / (1 + x**h))
        return -x + interaction

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)

    # 运行求解器
    sol = solve_ivp(
        _mm_ode,
        t_span,
        x,
        t_eval=t_eval,
        method='LSODA'
    )

    if not sol.success:
        print(f"警告: MM 模拟求解器未能成功: {sol.message}")

    # 格式化输出: 转置为 (T_steps, N) 并裁剪
    states = sol.y.T
    return np.clip(states, 0, None)


# ==============================================================
# Kuramoto Model
# ==============================================================
def kuramoto_dynamics(adj_matrix, omega=None, T_steps=1000, dt=0.05,
                      init_state=None, seed=None):
    """
    使用 solve_ivp 运行 Kuramoto 模拟。

    Equation:
        dθ_i/dt = ω_i + Σ_j A_ij sin(θ_j - θ_i)
        
    注意: omega 可以是标量，也可以是 N 维数组。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    
    if omega is None:
        omega = np.random.normal(0, 1, N)
        
    if init_state is None:
        theta = 2 * np.pi * np.random.rand(N)
    else:
        theta = init_state.copy()

    # 定义 ODE 方程
    def _kur_ode(t, theta):
        sin_diffs = np.sin(theta[np.newaxis, :] - theta[:, np.newaxis])
        coupling = np.sum(adj_matrix * sin_diffs, axis=1)
        return omega + coupling

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)
    
    # 运行求解器
    sol = solve_ivp(
        _kur_ode,
        t_span,
        theta,
        t_eval=t_eval,
        method='LSODA'
    )
    
    if not sol.success:
        print(f"警告: Kuramoto 模拟求解器未能成功: {sol.message}")

    # 格式化输出: 转置为 (T_steps, N) 并取模
    states = sol.y.T
    return np.mod(states, 2 * np.pi)


# ==============================================================
# Wilson–Cowan Model
# ==============================================================
def wc_dynamics(adj_matrix, tau=1.0, mu=0.0, T_steps=1000, dt=0.1,
                init_state=None, seed=None):
    """
    使用 solve_ivp 运行 Wilson-Cowan (WC) 模拟。

    Equation:
        dx_i/dt = -x_i + Σ_j A_ij * [1 / (1 + exp(-τ (x_j - μ)))]
        
    注意: tau 和 mu 可以是标量，也可以是 N 维数组。
    """
    if seed is not None:
        np.random.seed(seed)

    N = adj_matrix.shape[0]
    
    if init_state is None:
        x = np.random.rand(N)
    else:
        x = init_state.copy()
    x = np.clip(x, 0, None) # 确保初始活动非负

    # 定义 ODE 方程
    def _wc_ode(t, x):
        sigmoid = 1 / (1 + np.exp(-tau * (x - mu)))
        interaction = adj_matrix @ sigmoid
        return -x + interaction

    # 设置模拟时间
    t_end = T_steps * dt
    t_span = (0.0, t_end)
    t_eval = np.linspace(0.0, t_end, T_steps)

    # 运行求解器
    sol = solve_ivp(
        _wc_ode,
        t_span,
        x,
        t_eval=t_eval,
        method='LSODA'
    )
    
    if not sol.success:
        print(f"警告: WC 模拟求解器未能成功: {sol.message}")
        
    # 格式化输出: 转置为 (T_steps, N) 并裁剪
    states = sol.y.T
    return np.clip(states, 0, None)


# ==============================================================
# Quick Test (only runs when executed directly)
# ==============================================================
if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    A = nx.to_numpy_array(nx.erdos_renyi_graph(20, 0.2, seed=42))

    print("Running LV model for demo...")
    # 演示每个节点有不同的参数
    N_nodes = A.shape[0]
    alpha_vals = np.linspace(0.5, 1.5, N_nodes)
    theta_vals = np.linspace(0.8, 1.2, N_nodes)
    
    X = lv_dynamics(
        A, 
        alpha=alpha_vals, 
        theta=theta_vals, 
        T_steps=500, 
        dt=0.1
    )
    print("Simulation complete. Shape:", X.shape)

    plt.plot(X[:, :])
    plt.xlabel("Time step")
    plt.ylabel("State")
    plt.title("Lotka-Volterra (LV) model example")
    plt.show()