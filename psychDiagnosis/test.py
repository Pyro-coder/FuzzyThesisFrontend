from scipy.stats import norm
from scipy.optimize import root
from scipy.stats import chi2
from scipy.integrate import quad
import numpy as np



def fr(y, a):
    func = lambda r: norm.cdf(y + r) - norm.cdf(y - r) - a
    r = root(func, 0).x[0]
    return r


def integrand(y, xk, a, n):
    term1 = 1 - chi2.cdf((n - 1) * (fr(y, a) ** 2) / (xk ** 2), df=n)
    term2 = np.exp(-(n / 2) * (y ** 2))
    return term1 * term2

def fk(xk, a, n):
    result, _ = quad(integrand, 0, np.inf, args=(xk, a, n))
    return 2 / (np.sqrt(2 * np.pi) * np.sqrt(n)) * result


def kk(a, γ, n):
    func = lambda xk: fk(xk, a, n) - γ
    xk = root(func, 2).x[0]
    return xk

# print(kk(0.05, 0.05, 30))