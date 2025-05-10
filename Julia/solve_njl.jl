using QuadGK
using Plots
using NPZ

# 常数定义
m0 = 5.5  # 裸夸克质量 (MeV)
G = 13/12 * 5.074e-6  # 耦合常数 (MeV^-2)
p_max = 631.0  # 动量积分上限 (MeV)
N_c = 3
N_f = 2

# 费米分布函数
n_f(E, mu, T) = 1 / (1 + exp((E - mu) / T))
m_f(E, mu, T) = 1 / (1 + exp((E + mu) / T))

# 计算 sigma1 (包含 M 和 mu)
function sigma1_integral(M, mu, T)
    integrand(p) = let E = sqrt(p^2 + M^2)
        p^2 / E * (1 - n_f(E, mu, T) - m_f(E, mu, T))
    end
    result, _ = quadgk(integrand, 0, p_max)
    return -M * N_c * N_f * result / π^2
end

# Gap 方程的迭代求解
function solve_gap_equation(mu, T, M_init; tolerance=1e-6, max_iter=1000)
    M = M_init
    for i in 1:max_iter
        sigma1_value = sigma1_integral(M, mu, T)  # 计算 sigma1
        M_new = m0 - 2 * G * sigma1_value  # 计算新的 M
        if abs(M_new - M) < tolerance  # 判断是否收敛
            println("Converged after $i iterations.")
            return M_new
        end
        M = M_new  # 更新 M
    end
    println("Max iterations reached.")
    return M
end

function result_plot(T)
    # solve M with constant T
    # T = 10 #MeV
    mu_values = LinRange(1,500,500)
    M_results = solve_gap_equation.(mu_values,T,300.0)

    npzwrite("T_10GeV_M_data.npz",Dict(
        "mu_vals" => mu_values,
        "M_vals" => M_results,
    ))

    println("Data saved in T_10GeV_M_data.npz")


    # plot(mu_values,M_results,xlabel="mu MeV",ylabel="M(MeV)",title="Gap Equation Solution")

end

function result_plot3d()
    T_values = LinRange(1,250,250)
    mu_values = LinRange(1,500,500)

    T_grid = repeat(T_values,1,length(mu_values))
    mu_grid = repeat(mu_values',length(T_values),1)

    M_grid = [solve_gap_equation(mu,T,300.0) for (T,mu) in zip(T_grid,mu_grid)]
    
    npzwrite("./M_data.npz",Dict(
        "T_grid" => T_grid,
        "mu_grid" => mu_grid,
        "M_grid" => M_grid,
    ))

    println("Data saved in M_data.npz")
    
    # surface(mu_grid,T_grid,M_grid,xlabel="mu MeV",ylabel="M(MeV)",zlabel="M (MeV)",title="Gap Equation Solution",collor=:rainbow)

    # display(current())]
end

result_plot3d()

# result_plot(10)
