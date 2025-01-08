from matplotlib import pyplot as plt

import EKM
import trapezoid as tpz
import t2_centroid as t2c
import numpy as np
import alpha_cut_power_mean as acpm
import scipy.optimize as opt
import scipy.integrate as integrate
import mystic.symbolic as ms
from mystic.solvers import fmin
from mystic.penalty import quadratic_inequality
from mystic.constraints import as_constraint


# Define the gcd function
def gcd(x, w, r):
    if r <= 0:
        return EKM.t2wpm(x, w, r)
    else:
        xx = np.zeros_like(x)
        for i in range(len(x)):
            xx[i, 0] = 1 - x[i, 1]
            xx[i, 1] = 1 - x[i, 0]
        out = EKM.t2wpm(xx, w, -r)
        return np.array([1 - out[1], 1 - out[0]])


# Define the zcd function for CD variant
def zcd(x, y, w1, w2, r1, r2):
    if r1 == 0:
        out = ((1 - w1) * x ** (r2 * w2) + y ** (r2 * (1 - w2)) * w1) ** (1 / r2)
    else:
        out = ((1 - w1) * ((w2 * x ** r1 + (1 - w2) * y ** r1) ** (r2 / r1)) + w1 * x ** r2) ** (1 / r2)
    return out


# Define the zdc function for DC variant
def zdc(x, y, w1, w2, r1, r2):
    if r2 == 0:
        out = ((w1 * (x ** r1) + (1 - w1) * (y ** r1)) ** ((1 - w2) / r1)) * (x ** w2)
    else:
        out = ((1 - w2) * ((w1 * (x ** r1) + (1 - w1) * (y ** r1)) ** (r2 / r1)) + w2 * (x ** r2)) ** (1 / r2)
    return out


# Define the andness function
def andness(r):
    def integrand_r0(x, y):
        return np.sqrt(x * y)

    def integrand_r(x, y, r):
        return ((x ** r + y ** r) / 2) ** (1 / r)

    if r == 0:
        integral, _ = integrate.dblquad(integrand_r0, 0, 1, lambda y: 0, lambda y: 1, epsabs=1e-10, epsrel=1e-10)
        out = 2 - 3 * integral
    else:
        try:
            integral, _ = integrate.dblquad(integrand_r, 0, 1, lambda y: 0, lambda y: 1, args=(r,), epsabs=1e-10,
                                            epsrel=1e-10)
            out = 2 - 3 * integral
        except Exception:
            if r > 1:
                out = 0
            else:
                out = 1
    return out


# Define the omega function
def omega(r):
    return 1 - andness(r)


# Define the r_exponent function
def r_exponent(alpha):
    if alpha == 1:
        return -np.inf
    elif alpha == 0:
        return np.inf
    elif 0 < alpha < 1:
        f = lambda r: andness(r) - alpha
        sol = opt.root(f, 0)
        return sol.x[0]
    else:
        raise ValueError("Alpha should be between 0 and 1")


# Define the dc_delta_plus function for DC variant
def dc_delta_plus(w1, w2, r1, r2):
    def integrand(x):
        return zdc(x, 1, w1, w2, r1, r2)

    integral_value, _ = integrate.quad(integrand, 0, 1, epsabs=1e-10, epsrel=1e-10)
    delta_plus = 100 * abs(2 * integral_value - 1)

    return delta_plus


# Define the dc_delta_minus function for DC variant
def dc_delta_minus(w1, w2, r1, r2):
    if r2 != 0:
        out = 100 * (((1 - w2) * (w1 ** (r2 / r1)) + w2) ** (1 / r2) - 1)
    else:
        out = 0
    return out


# Define the cd_delta_plus function for CD variant
def cd_delta_plus(w1, w2, r1, r2):
    """
    Calculate δ+ for CD variant
    Note that r1 < 1, r2 >= 1 in this variant, and a = w1, b = w2 in Dujmovic's paper
    δ+ is the specified (Reward) truth value in % of a CD variant partial absorption operator when the absorption controlled variable y = 0
    δ+ and δ- are used to determine the weights w1 and w2 of partial absorption operator
    """
    integrand = lambda x: zcd(x, 1, w1, w2, r1, r2)

    integral_result, _ = integrate.quad(integrand, 0, 1)

    delta_plus = 100 * abs(2 * integral_result - 1)

    return delta_plus


# Define the cd_delta_minus function for CD variant
def cd_delta_minus(w1, w2, r1, r2):
    """
    Calculate δ- for CD variant
    Note that r1 < 1, r2 >= 1 in this variant, and a = w1, b = w2 in Dujmovic's paper
    δ- is the (Penalty) reduction in truth value (%) of CD variant partial absorption operator when the absorption controlled variable y = 1
    δ+ and δ- are used to determine the weights w1 and w2 of partial absorption operator
    """
    if r2 != 0:
        if r1 > 0:
            out = 100 * ((1 - w1) * (w2 ** (r2 / r1)) + w1) ** (1 / r2) - 100
        else:
            out = 100 * (w1 ** (1 / r2)) - 100
    else:
        out = 0

    return out


# Define the objective function for DC variant
def objective_dc(params, r1, r2, reward, penalty):
    w1, w2 = params
    delta_plus = dc_delta_plus(w1, w2, r1, r2)
    delta_minus = dc_delta_minus(w1, w2, r1, r2)
    return (delta_plus - reward) ** 2 + (delta_minus - penalty) ** 2


# Define the objective function for CD variant
def objective_cd(params, r1, r2, reward, penalty):
    w1, w2 = params
    delta_plus = cd_delta_plus(w1, w2, r1, r2)
    delta_minus = cd_delta_minus(w1, w2, r1, r2)
    return (delta_plus - reward) ** 2 + (delta_minus - penalty) ** 2


# Define the constraints
def constraints(params):
    w1, w2 = params
    return [w1, 1 - w1, w2, 1 - w2]


# Set up penalty functions for the constraints
@quadratic_inequality(lambda w1_w2: w1_w2[0] >= 0)
@quadratic_inequality(lambda w1_w2: w1_w2[0] <= 1)
@quadratic_inequality(lambda w1_w2: w1_w2[1] >= 0)
@quadratic_inequality(lambda w1_w2: w1_w2[1] <= 1)
def penalty(params):
    return 0.0


# Define the wcd function for DC variant
def wdc(r1, r2, P, R):
    # Set the initial guess
    initial_guess = [0.5, 0.5]

    # Optimize
    result = fmin(objective_dc, x0=initial_guess, penalty=penalty, args=(r1, r2, R, P))

    # Extract the optimized values of w1 and w2
    w1_opt, w2_opt = result

    return np.array([w1_opt, w2_opt])


# Define the wcd function for CD variant
def wcd(r1, r2, P, R):
    # Set the initial guess
    initial_guess = [0.5, 0.5]

    # Optimize
    result = fmin(objective_cd, x0=initial_guess, penalty=penalty, args=(r1, r2, R, P))

    # Extract the optimized values of w1 and w2
    w1_opt, w2_opt = result

    return np.array([w1_opt, w2_opt])

# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fAR2)(X, Y)
#
# # Ensure Z contains finite values for contour plot
# Z = np.nan_to_num(Z, nan=np.nanmin(Z))
#
# # Define contour levels at 0.1 intervals
# levels = np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1)
#
# # Plot the mesh grid
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X, Y, Z, levels=levels, cmap='viridis')
# cbar = plt.colorbar(contour, ticks=np.arange(np.nanmin(Z), np.nanmax(Z) + 0.1, 0.1))
# cbar.ax.set_ylabel('fAR2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fAR2(x, y)')
# plt.show()

wRA = wcd(r_exponent(14 / 16), 1, -25, 15)

print(wRA)
