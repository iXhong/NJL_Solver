import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve

#define the paramters

G = 1
mu = 1
m = 5.5
Lambda = 631
T = 10
Nc = 3
Nf = 2


#define functions
def n_f(p, mu, M):
    E = np.sqrt(M**2 + p**2)
    return 1 / (1 + np.exp((E - mu) / T))


def m_f(p, mu, M):
    E = np.sqrt(M**2 + p**2)
    return 1 / (1 + np.exp((E + mu) / T))


def intergrand(p,M,mu):

    E = np.sqrt(M**2+p**2)

    return p**2 * (1 - n_f(p,mu,M) - m_f(p,mu,M))/E

def sigma1(M,mu):

    intergral,_ =  quad(intergrand,0,Lambda,args=(M,mu))

    return -M*Nc*Nf / (np.pi**2)* intergral

def effective_mass(M,mu):
    
    return m - 2*G*sigma1(M,mu)


# Function solve
M_init = 30
max_iter = 1000
tolerance = 1e-4

M = M_init

# for i in range(max_iter):

#     M_new = effective_mass(M,mu)
#     if np.abs(M_new - M) < tolerance:
#         print(f"solution is {M_new}")
#         break
#     M = M_new

# else:
#     print("sorry")

result = fsolve(effective_mass,M_init,args=(mu))

print(f"result is {result}")