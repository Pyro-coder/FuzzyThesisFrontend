import numpy as np
import scipy
import math
from scipy.stats import nct
import scipy.optimize as opt
import scipy.stats as stats

# here are the functions

# calculate the approxomation of the ratio of gamma(n+1)/gamma(n+3/2) for large m
def rgamma(n):
    x = n + 1
    fval = np.sqrt(x-.25) * (1 + 1 / (64 * x * x) + 1 / (128 * x * x * x))
    return 1/fval



# calculate the cdf of a non-central t statistic
# t is the cdf variable; n is # degrees of freedom; δ is the noncentrality parameter
def nct_cdf(t, n, delta, tol):
    r = lambda t_val, n_val: t_val*t_val/(t_val*t_val+n_val)
    if (t >= 0):
        out = stats.norm.cdf(-delta)
        for m in range(0, 1000):
            if (m <= 50):
                sumincr = 0.5 * stats.poisson.pmf(m, delta * delta / 2) * (stats.beta.cdf(r(t, n), m + 0.5, n / 2) + ((delta * scipy.special.gamma(m + 1)) / (np.sqrt(2) * scipy.special.gamma(m + 1.5))) * stats.beta.cdf(r(t, n), m + 1, n / 2))
            else:
                sumincr = 0.5 * stats.poisson.pmf(m, delta * delta / 2) * (stats.beta.cdf(r(t, n), m + 0.5, n / 2) + (delta / np.sqrt(2) * rgamma(m)) * stats.beta.cdf(r(t, n), m + 1, n / 2))
            out = out + sumincr

            if (sumincr / out < tol):
                break
    else:
        out = stats.norm.cdf(delta)
        for m in range(0, 1000):
            if (m <= 50):
                sumincr = 0.5 * stats.poisson.pmf(m, delta * delta / 2) * (stats.beta.cdf(r(t, n), m + 0.5, n / 2) + ((delta * scipy.special.gamma(m + 1)) / (np.sqrt(2) * scipy.special.gamma(m + 1.5))) * stats.beta.cdf(r(t, n), m + 1, n / 2))
            else:
                sumincr = 0.5 * stats.poisson.pmf(m, delta * delta / 2) * (stats.beta.cdf(r(t, n), m + 0.5, n / 2) + (delta / np.sqrt(2) * rgamma(m)) * stats.beta.cdf(r(t, n), m + 1, n / 2))

            out = out + sumincr
            if (abs(sumincr)/out < tol):
                out = 1 - out
                break
    return out



# print(nct_cdf(2, 100, 2, 0.00001))


def inv_nct_cdf(P, n, delta, tol):
    #Inverse of non-central t-statistic cdf; finds the threshold yielding a cdf value of P
    #n is the # of degrees of freedom

    # set initial t guess to lowercase delta (delta) and step up and down to find an interval where f(t) has opposite signs for the endpoints
    f = lambda t: nct_cdf(t, n, delta, tol) - P

    x0 = delta

    # Determine the direction to move in
    if f(x0) <= 0:
        x1 = 2 * delta
        while (f(x0) <= 0 and f(x1) <= 0):
            x0 = x1
            x1 = x1 + delta
    else:
        x0 = 0
        x1 = delta
        while (f(x0) > 0 and f(x1) >= 0):
            x1 = x0
            x0 = x0 - delta
    

    return opt.brentq(f, x0, x1)


# print(inv_nct_cdf(.95, 50, 2, 0.000001))
    


def ktol(upsilon, a, n, tol):
    #Find one-sided tolerance interval k (multiple of sample std dev away from sample mean), given γ, α and n
    zp = stats.norm.ppf(1-upsilon, 0, 1)
    delta = math.sqrt(n) * zp
    result = inv_nct_cdf(1 - a, n - 1, delta, tol) / math.sqrt(n)
    return result

# print (ktol(.05, .05, 175, 0.000001))