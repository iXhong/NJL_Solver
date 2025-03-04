import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

# Constants
Nc = 3  # Number of colors
Nf = 2  # Number of flavors
T = 10  # Temperature in MeV
mu = 10
G = 5.074e-6  # Coupling constant
m0 = 5.5  # Current quark mass in MeV
Lambda = 631  # Cutoff in MeV

# Fermi-Dirac distribution functions
def n_f(p, mu, M):
    E = np.sqrt(M**2 + p**2)
    return 1 / (1 + np.exp((E - mu) / T))

def m_f(p, mu, M):
    E = np.sqrt(M**2 + p**2)
    return 1 / (1 + np.exp((E + mu) / T))

# Integrands for sigma1 and sigma2
def integrand_sigma1(p,mu, M):
    E = np.sqrt(M**2 + p**2)
    return p**2 / E * (1 - n_f(p, mu, M) - m_f(p, mu, M))

# def integrand_sigma2(p, mu, M):
#     return p**2 * (n(p, mu, M) - m(p, mu, M))

# Self-consistent equations for sigma1 and sigma2
def sigma1(mu, M):
    integral, _ = quad(integrand_sigma1(mu,M), 0, Lambda)
    return -M * Nc * Nf / (np.pi**2) * integral

# def sigma2(mu, M):
#     integral, _ = quad(integrand_sigma2, 0, Lambda, args=(mu, M))
#     return Nc * Nf / (np.pi**2) * integral

# Effective quark mass equation
def effective_mass_equation(M, mu):
    sigma1_val = sigma1(mu, M)
    # sigma2_val = sigma2(mu, M)
    # mu_r = mu - (G / Nc) * sigma2_val
    return m0 - 2 * G * sigma1_val


def solve():
    M_init = 300
    M_solution = fsolve(effective_mass_equation,M_init,args=(mu))

    print(M_solution)


solve()



# # Solve for M for a range of mu values
# mu_values = np.linspace(0, 500, 100)  # Range of mu values
# M_values = []

# for mu in mu_values:
#     # Initial guess for M
#     M_guess = 300  # Initial guess for M
#     # Solve the self-consistent equation
#     M_solution = fsolve(effective_mass_equation, M_guess, args=(mu))
#     M_values.append(M_solution[0])

# # Plot the results
# import matplotlib.pyplot as plt

# plt.plot(mu_values, M_values, label='Effective Quark Mass $M$')
# plt.xlabel('Chemical Potential $\mu_r$ (MeV)')
# plt.ylabel('Effective Quark Mass $M$ (MeV)')
# plt.title('Effective Quark Mass vs Chemical Potential')
# plt.legend()
# plt.grid()
# plt.show()