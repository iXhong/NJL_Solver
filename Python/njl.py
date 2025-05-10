import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# 常数定义
m0 = 5.5  # 裸夸克质量 (MeV)
G = 13/12 * 5.074e-6  # 耦合常数 (MeV^-2)
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
def solve_gap_equation(mu,T,M_init, tolerance=1e-6, max_iter=1000):
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
# M_solution = solve_gap_equation(mu=10,T=10,M_init=300)
# print(f"Effective quark mass M: {M_solution} MeV")

#绘制T为定值时的M-mu 图像
def plot(T):

    mu_values = np.linspace(1,500,100) # generate a list of mu
    M_init = 300 #MeV
    M_values  = [solve_gap_equation(mu=mu,T=T,M_init=M_init) for mu in mu_values ]

    # plt.figure(figsize=(8, 6))
    plt.plot(mu_values, M_values, label=f'T = {T} MeV')
    plt.xlabel(r'Chemical potential $\mu$ (MeV)')
    plt.ylabel(r'Effective quark mass $M$ (MeV)')
    plt.title(r'$M$ vs $\mu$ at T = {0} MeV'.format(T))
    plt.grid(True)
    plt.legend()
    # plt.savefig('T-mu.png',dpi=300) #save to a png
    plt.show()


# plot 3d figure of  M - (T,mu)
def plot3d():

    #create the figure
    # fig = plt.figure(figsize = ())
    # ax = plt.axes(projection='3d')    
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    mu_values = np.linspace(1,20,20)
    T_values = np.linspace(1,20,20)
    M_init = 300 #MeV
    
    mu_grid , T_grid = np.meshgrid(mu_values,T_values)
    M_grid = np.zeros_like(T_grid)

    # calc the M
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            M_grid[i,j] = solve_gap_equation(mu=mu_grid[i,j],T=T_grid[i,j],M_init=M_init)

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

#define the gap equation
def gap_eq(M,mu,T):
    return M - m0 + 2*G*sigma1_integral(M,mu,T)

#determine where is the M when the multi-solution occur
def f_values():
    mu = 355 # MeV
    T = 1 #MeV
    M_values = np.linspace(0,400,100)
    f_values = [gap_eq(M,mu,T) for M in M_values]

    plt.plot(M_values,f_values,label='Gap Equation')
    plt.axhline(0, color='k', linestyle='--')
    plt.xlabel("M (MeV)")
    plt.ylabel("Gap Function f(M)")
    plt.legend()
    plt.show()

# test the multi solutions existence
def test_multi_solve():
    mu = 350
    T = 1
    M_init0 = 300 #MeV
    M_init1 = 60  #MeV

    # M_value0 = solve_gap_equation(mu,T,M_init=M_init0)
    # M_value1 = solve_gap_equation(mu,T,M_init=M_init1)
    M_value0 = fsolve(gap_eq,M_init0,args=(mu,T))
    M_value1 = fsolve(gap_eq,M_init1,args=(mu,T))

    print(f"Solution of init M = {M_init0} is {M_value0}")
    print(f"Solution of init M = {M_init1} is {M_value1}")





#find the multi solutions in (T,mu)
def multi_solve():

    T_values = np.linspace(1,40,20)
    mu_values = np.linspace(320,360,20)
    M_init0 = 300   #MeV
    M_init1 = 60    #MeV

    mu_grid, T_grid = np.meshgrid(mu_values,T_values)
    solution_grid0 = np.zeros_like(T_grid)
    solution_grid1 = np.zeros_like(T_grid)
    result_grid = np.zeros_like(T_grid)

    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            solution_grid0[i,j] = fsolve(gap_eq,M_init0,args=(mu_grid[i,j],T_grid[i,j]))[0]
            solution_grid1[i,j] = fsolve(gap_eq,M_init1,args=(mu_grid[i,j],T_grid[i,j]))[0]
            if np.isclose(solution_grid0[i,j],solution_grid1[i,j],rtol=1e-6) == True:
                result_grid[i,j] = 0
            else:
                result_grid[i,j] = 1

    plt.figure(figsize=(8, 6))
    # 使用 imshow 绘制多解区
    plt.imshow(result_grid, origin='lower', 
            extent=[mu_values[0], mu_values[-1], T_values[0], T_values[-1]], 
            aspect='auto')

    # 添加颜色条，标注 1（多解区）和 0（单解区）
    plt.colorbar(label='Multi-solution region (1 = Multi-solution, 0 = Single-solution)')

    # 轴标签
    plt.xlabel('Chemical Potential $\mu$')
    plt.ylabel('Temperature $T$')
    plt.title('Multi-solution Regions in $T$-$\mu$ Plane')

    # 显示图像
    plt.show()


def multi_solve():

    nx = 20
    ny = 20
    T_values = np.linspace(1,40,ny)
    mu_values = np.linspace(320,360,nx)
    M_init0 = 300   #MeV
    M_init1 = 60    #MeV

    solution_grid = np.zeros((20,20))

    for i in range(nx):
        for j in range(ny):
            solution0 = fsolve(gap_eq,M_init0,args=(mu_values[i],T_values[j]))[0]
            solution1 = fsolve(gap_eq,M_init1,args=(mu_values[i],T_values[j]))[0]
            if np.abs(solution0 - solution1) < 1e-1:
                solution_grid[i,j] = 1
            else:
                solution_grid[i,j] = 0

    
    # 绘制结果
    plt.figure(figsize=(8, 6))

    # 使用imshow绘制多解区
    plt.imshow(solution_grid.T, origin='lower', extent=[mu_values[0], mu_values[-1], T_values[0], T_values[-1]], aspect='auto')
    plt.colorbar(label='Multi-solution region (1 = Multi-solution, 0 = Single-solution)')

    # 标注
    plt.xlabel('Chemical Potential $\mu$')
    plt.ylabel('Temperature $T$')
    plt.title('Multi-solution Regions in $T$-$\mu$ Plane')

    # 显示图像
    plt.show()


def hello():
    return 0