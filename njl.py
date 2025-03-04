import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

# 常数定义
m0 = 5.0  # 裸夸克质量 (MeV)
G = 5.074e-6  # 耦合常数 (MeV^-2)
p_max = 631.0  # 动量积分上限 (MeV)
# T = 10.0  # 温度 (MeV)
# mu = 400.0  # 化学势 (MeV)
N_c = 3
N_f = 2

# 费米分布函数
def n_f(E, mu, T):
    return 1 / (1 + np.exp((E - mu) / T))

def m_f(E, mu, T):
    return 1 / (1 + np.exp((E + mu) / T))

# 计算 sigma1 (包含 M 和 mu)
def sigma1_integral(M, mu, T):
    def integrand(p):
        E = np.sqrt(p**2 + M**2)
        return p**2 / E * (1 - n_f(E, mu, T) - m_f(E, mu, T))
    
    result, error = integrate.quad(integrand, 0, p_max)
    return -M*N_c*N_f*result / np.pi**2

# Gap 方程的迭代求解
def solve_gap_equation(mu,T,M_init=300, tolerance=1e-6, max_iter=1000):
    M = M_init
    for i in range(max_iter):
        sigma1_value = sigma1_integral(M, mu, T)  # 计算 sigma1
        M_new = m0 - 2 * G * sigma1_value  # 计算新的 M
        if np.abs(M_new - M) < tolerance:  # 判断是否收敛
            print(f"Converged after {i+1} iterations.")
            return M_new
        M = M_new  # 更新 M
    print("Max iterations reached.")
    return M

# # 运行求解
# M_solution = solve_gap_equation()
# print(f"Effective quark mass M: {M_solution:.5f} MeV")

#绘制T为定值时的M-mu 图像
def plot(T):

    mu_values = np.linspace(1,500,500) # mu
    M_values  = [solve_gap_equation(mu=mu,T=T) for mu in mu_values ]

    # plt.figure(figsize=(8, 6))
    plt.plot(mu_values, M_values, label=f'T = {T} MeV')
    plt.xlabel(r'Chemical potential $\mu$ (MeV)')
    plt.ylabel(r'Effective quark mass $M$ (MeV)')
    plt.title(r'$M$ vs $\mu$ at T = {0} MeV'.format(T))
    plt.grid(True)
    plt.legend()
    plt.savefig('T-mu.png',dpi=300)
    plt.show()


# plot 3d figure of  M - (T,mu)
def plot3d():

    #create the figure
    # fig = plt.figure(figsize = ())
    # ax = plt.axes(projection='3d')    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    mu_values = np.linspace(1,20,20)
    T_values = np.linspace(1,20,20)
    
    mu_grid , T_grid = np.meshgrid(mu_values,T_values)
    M_grid = np.zeros_like(T_grid)

    # calc the M
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            M_grid[i,j] = solve_gap_equation(mu=mu_grid[i,j],T=T_grid[i,j])

    # 绘制 3D 图像
    fig = plt.figure(figsize=(12,8))
    # ax = plt.axes(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface( mu_grid,T_grid, M_grid, cmap='plasma')

    # 轴标签
    ax.set_xlabel(r'Temperature $T$ (MeV)')
    ax.set_ylabel(r'Chemical Potential $\mu$ (MeV)')
    ax.set_zlabel(r'Effective quark mass $M$ (MeV)')
    ax.set_title(r'$M(T, \mu)$ in NJL Model')

    fig.colorbar(surf,shrink=0.5,aspect=8)

    plt.show()



