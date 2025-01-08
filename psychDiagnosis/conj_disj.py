from matplotlib import pyplot as plt

from . import EKM
import numpy as np
from . import alpha_cut_power_mean as acpm
import scipy.optimize as opt
import scipy.integrate as integrate
from mystic.solvers import fmin
from mystic.penalty import quadratic_inequality


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
        out = ((1 - w1) * (x ** (r2 * w2)) * (y ** (r2 * (1 - w2))) + w1 * (x ** r2)) ** (1 / r2)
    else:
        out = ((1 - w1) * ((w2 * (x ** r1) + (1 - w2) * (y ** r1)) ** (r2 / r1)) + w1 * (x ** r2)) ** (1 / r2)

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


# wAR = wdc(1, r_exponent(14 / 16), -25, 15)


# print(wAR)
#
# print(dc_delta_plus(wAR[0], wAR[1], 1, r_exponent(14 / 16)))
# print(dc_delta_minus(wAR[0], wAR[1], 1, r_exponent(14 / 16)))
# r2 = r_exponent(14 / 16)


# def fAR2(x, y):
#     return zdc(x, y, wAR[0], wAR[1], 1, r2)


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

# wRA = wcd(r_exponent(14 / 16), 1, -25, 15)


# print(wRA)
#
# print(cd_delta_plus(wRA[0], wRA[1], r_exponent(14 / 16), 1))
# print(cd_delta_minus(wRA[0], wRA[1], r_exponent(14 / 16), 1))
#
# r1 = r_exponent(14 / 16)
# print(r1)

# def fRA2(x, y):
#     return zcd(x, y, wRA[0], wRA[1], r1, 1)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRA2)(X, Y)
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
# cbar.ax.set_ylabel('fRA2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRA2(x, y)')
# plt.show()

# wRRbar = wcd(0, r_exponent(7 / 16), -25, 15)
# r2 = r_exponent(7 / 16)


# print(cd_delta_plus(wrp_bar[0], wrp_bar[1], 0, r_exponent(7 / 16)))
# print(cd_delta_minus(wrp_bar[0], wrp_bar[1], 0, r_exponent(7 / 16)))
# print(r2)


# def fRRbar2(x, y):
#     return zcd(x, y, wRRbar[0], wRRbar[1], 0, r2)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRRbar2)(X, Y)
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
# cbar.ax.set_ylabel('fRRbar2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRRbar2(x, y)')
# plt.show()

# wRbarR = wdc(r_exponent(7 / 16), -10, -25, 15)
# r1 = r_exponent(7 / 16)
#
#
# def fRbarR2(x, y):
#     return zdc(x, y, wRbarR[0], wRbarR[1], r1, -10)


# # Create a mesh grid
# x = np.linspace(0.001, 1, 50)
# y = np.linspace(0.001, 1, 50)
# X, Y = np.meshgrid(x, y)
#
# # Evaluate the function over the mesh grid using the safe wrapper
# Z = np.vectorize(fRbarR2)(X, Y)
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
# cbar.ax.set_ylabel('fRbarR2(x, y)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Contour plot of fRbarR2(x, y)')
# plt.show()


def one_minus(x):
    """
    Compute (1 - x) α-cuts of an array of α-cuts x.
    """
    rows = x.shape[0]
    out = np.zeros_like(x)

    for i in range(rows):
        out[i, 0] = 1 - x[i, 1]
        out[i, 1] = 1 - x[i, 0]
        out[i, 2] = x[i, 2]

    return out


def alpha_w(w, n):
    """
    Compute (n + 1) α-cuts array and its complement for a constant w.
    """
    wout = np.full((n + 1,), w)
    alpha = np.linspace(0, 1, n + 1)

    # Create the augmented array
    augmented = np.column_stack((wout, wout, alpha))

    # Compute the complement
    complement = one_minus(augmented)

    # Combine the results into a single output array
    out = [augmented, complement]

    return out


def alpha_cpa(x, y, r1, r2, P, R):
    """
    Compute α-cuts of DC-variant conjunctive partial absorption operator,
    with x mandatory, y desired IT2 fuzzy variables.
    """
    # Compute w1 and w2 corresponding to P and R
    w = wdc(r1, r2, P, R)

    # Calculate DCδplus and DCδminus to verify correct weights
    dcδplus = dc_delta_plus(w[0], w[1], r1, r2)
    dcδminus = dc_delta_minus(w[0], w[1], r1, r2)

    # Compute the α-cuts of the disjunction of x and y, using weight w_0, exponent r1 >= 1
    disj = acpm.alpha_to_alpha_t2wpm([[x[0], y[0]], [x[1], y[1]]], [[x[0], y[0]], [x[1], y[1]]], alpha_w(w[0], 100)[0],
                                     alpha_w(w[0], 100)[1], r1)

    # Compute the α-cuts of the conjunction of x and disj, using weight w_1, exponent r2 < 1
    out = acpm.alpha_to_alpha_t2wpm([[x[0], disj[0]], [x[1], disj[1]]], [[x[0], disj[0]], [x[1], disj[1]]],
                                    alpha_w(w[1], 100)[0], alpha_w(w[1], 100)[1], r2)

    return out


def alpha_dpa(x, y, r1, r2, P, R):
    """
    Compute α-cuts of CD-variant disjunctive partial absorption operator,
    with x sufficient, y desired IT2 fuzzy variables.
    """
    # Compute w1 and w2 corresponding to P and R
    w = wcd(r1, r2, P, R)

    # Calculate CDδplus and CDδminus to verify correct weights
    cdδplus = cd_delta_plus(w[0], w[1], r1, r2)
    cdδminus = cd_delta_minus(w[0], w[1], r1, r2)

    # Compute the α-cuts of the conjunction of x and y, using weight w_0, exponent r1 < 1
    conj = acpm.alpha_to_alpha_t2wpm([[x[0], y[0]], [x[1], y[1]]], [[x[0], y[0]], [x[1], y[1]]], alpha_w(w[0], 100)[0],
                                     alpha_w(w[0], 100)[1], r1)

    # Compute the α-cuts of the disjunction of x and conj, using weight w_1, exponent r2 >= 1
    out = acpm.alpha_to_alpha_t2wpm([[x[0], conj[0]], [x[1], conj[1]]], [[x[0], conj[0]], [x[1], conj[1]]],
                                    alpha_w(w[1], 100)[0], alpha_w(w[1], 100)[1], r2)

    return out


# x = np.arange(0, 10.01, 0.01)
#
#
# def NVLl(x):
#     return tpz.trap(x, 1, 0, 0, 0.02, 0.33)
#
#
# def VLl(x):
#     return tpz.trap(x, 1, 0, 0, 0.14, 1.82)
#
#
# def Ll(x):
#     return tpz.trap(x, 0.31, 1.9, 2.24, 2.24, 2.51)
#
#
# def MLLl(x):
#     return tpz.trap(x, 0.32, 2.99, 3.31, 3.31, 3.81)
#
#
# def FMLHl(x):
#     return tpz.trap(x, 0.43, 5.79, 6.31, 6.31, 7.21)
#
#
# def MLHl(x):
#     return tpz.trap(x, 0.29, 6.9, 7.21, 7.21, 7.6)
#
#
# def Hl(x):
#     return tpz.trap(x, 1, 7.68, 9.82, 9.82, 10)
#
#
# def EHl(x):
#     return tpz.trap(x, 1, 9.74, 9.98, 9.98, 10)
#
#
# def NVLu(x):
#     return tpz.trap(x, 1, 0, 0, 0.22, 3.16)
#
#
# def VLu(x):
#     return tpz.trap(x, 1, 0, 0, 1.37, 3.95)
#
#
# def Lu(x):
#     return tpz.trap(x, 1, 0.38, 1.63, 3, 4.62)
#
#
# def MLLu(x):
#     return tpz.trap(x, 1, 0.38, 2.25, 4, 5.92)
#
#
# def FMLHu(x):
#     return tpz.trap(x, 1, 2.33, 5.11, 7, 9.59)
#
#
# def MLHu(x):
#     return tpz.trap(x, 1, 4.38, 6.25, 8, 9.62)
#
#
# def Hu(x):
#     return tpz.trap(x, 1, 4.73, 8.82, 10, 10)
#
#
# def EHu(x):
#     return tpz.trap(x, 1, 7.1, 9.8, 10, 10)
#
#
# def VBl(x):
#     return tpz.trap(x, 1, 0, 0, 0.09, 1.32)
#
#
# def Bl(x):
#     return tpz.trap(x, 0.48, 1.79, 2.37, 2.37, 2.71)
#
#
# def SBl(x):
#     return tpz.trap(x, 0.42, 2.79, 3.3, 3.3, 3.71)
#
#
# def Fl(x):
#     return tpz.trap(x, 0.27, 4.79, 5.12, 5.12, 5.35)
#
#
# def SGl(x):
#     return tpz.trap(x, 0.4, 5.89, 6.34, 6.34, 6.81)
#
#
# def Gl(x):
#     return tpz.trap(x, 0.47, 6.79, 7.25, 7.25, 7.91)
#
#
# def VGl(x):
#     return tpz.trap(x, 1, 7.66, 9.82, 9.82, 10)
#
#
# def VBu(x):
#     return tpz.trap(x, 1, 0, 0, 0.59, 3.95)
#
#
# def Bu(x):
#     return tpz.trap(x, 1, 0.28, 2, 3, 5.22)
#
#
# def SBu(x):
#     return tpz.trap(x, 1, 0.98, 2.75, 4, 5.41)
#
#
# def Fu(x):
#     return tpz.trap(x, 1, 2.38, 4.5, 6, 8.18)
#
#
# def SGu(x):
#     return tpz.trap(x, 1, 4.02, 5.65, 7, 8.41)
#
#
# def Gu(x):
#     return tpz.trap(x, 1, 4.38, 6.5, 7.75, 9.62)
#
#
# def VGu(x):
#     return tpz.trap(x, 1, 5.21, 8.27, 10, 10)
#
#
# def Ul(x):
#     return tpz.trap(x, 1, 0, 0, 0.09, 1.15)
#
#
# def MLUl(x):
#     return tpz.trap(x, 0.34, 2.79, 3.21, 3.21, 3.71)
#
#
# def MLIl(x):
#     return tpz.trap(x, 0.33, 5.79, 6.28, 6.28, 6.67)
#
#
# def VIl(x):
#     return tpz.trap(x, 1, 8.68, 9.91, 9.91, 10)
#
#
# def Uu(x):
#     return tpz.trap(x, 1, 0, 0, 0.55, 4.61)
#
#
# def MLUu(x):
#     return tpz.trap(x, 1, 0.42, 2.25, 4, 5.41)
#
#
# def MLIu(x):
#     return tpz.trap(x, 1, 3.38, 5.5, 7.25, 9.02)
#
#
# def VIu(x):
#     return tpz.trap(x, 1, 7.37, 9.36, 10, 10)


def antonym(f, x):
    return f(10 - x)


def complement(f, x):
    return 1 - f(x)


# def z8l(x):
#     return np.array([
#         NVLl(x),
#         VLl(x),
#         Ll(x),
#         MLLl(x),
#         FMLHl(x),
#         MLHl(x),
#         Hl(x),
#         EHl(x)
#     ])
#
#
# def z8u(x):
#     return np.array([
#         NVLu(x),
#         VLu(x),
#         Lu(x),
#         MLLu(x),
#         FMLHu(x),
#         MLHu(x),
#         Hu(x),
#         EHu(x)
#     ])
#
#
# def z7l(x):
#     return np.array([
#         VBl(x),
#         Bl(x),
#         SBl(x),
#         Fl(x),
#         SGl(x),
#         Gl(x),
#         VGl(x)
#     ])
#
#
# def z7u(x):
#     return np.array([
#         VBu(x),
#         Bu(x),
#         SBu(x),
#         Fu(x),
#         SGu(x),
#         Gu(x),
#         VGu(x)
#     ])
#
#
# def z4l(x):
#     return np.array([
#         Ul(x),
#         MLUl(x),
#         MLIl(x),
#         VIl(x)
#     ])
#
#
# def z4u(x):
#     return np.array([
#         Uu(x),
#         MLUu(x),
#         MLIu(x),
#         VIu(x)
#     ])
#
#
# def fNVL(x):
#     return np.array([z8u(x)[0], z8l(x)[0]])
#
#
# def fVL(x):
#     return np.array([z8u(x)[1], z8l(x)[1]])
#
#
# def fL(x):
#     return np.array([z8u(x)[2], z8l(x)[2]])
#
#
# def fMLL(x):
#     return np.array([z8u(x)[3], z8l(x)[3]])
#
#
# def fFMLH(x):
#     return np.array([z8u(x)[4], z8l(x)[4]])
#
#
# def fMLH(x):
#     return np.array([z8u(x)[5], z8l(x)[5]])
#
#
# def fH(x):
#     return np.array([z8u(x)[6], z8l(x)[6]])
#
#
# def fEH(x):
#     return np.array([z8u(x)[7], z8l(x)[7]])
#
#
# def fVB(x):
#     return np.array([z7u(x)[0], z7l(x)[0]])
#
#
# def fB(x):
#     return np.array([z7u(x)[1], z7l(x)[1]])
#
#
# def fSB(x):
#     return np.array([z7u(x)[2], z7l(x)[2]])
#
#
# def fF(x):
#     return np.array([z7u(x)[3], z7l(x)[3]])
#
#
# def fSG(x):
#     return np.array([z7u(x)[4], z7l(x)[4]])
#
#
# def fG(x):
#     return np.array([z7u(x)[5], z7l(x)[5]])
#
#
# def fVG(x):
#     return np.array([z7u(x)[6], z7l(x)[6]])
#
#
# def fU(x):
#     return np.array([z4u(x)[0], z4l(x)[0]])
#
#
# def fMLU(x):
#     return np.array([z4u(x)[1], z4l(x)[1]])
#
#
# def fMLI(x):
#     return np.array([z4u(x)[2], z4l(x)[2]])
#
#
# def fVI(x):
#     return np.array([z4u(x)[3], z4l(x)[3]])
#
#
# alpha_NVL = acul.alpha_t2(fNVL, 1000, 0, 10, 100, 300)
# alpha_VL = acul.alpha_t2(fVL, 1000, 0, 10, 100, 300)
# alpha_L = acul.alpha_t2(fL, 1000, 0, 10, 100, 300)
# alpha_MLL = acul.alpha_t2(fMLL, 1000, 0, 10, 100, 300)
# alpha_FMLH = acul.alpha_t2(fFMLH, 1000, 0, 10, 100, 300)
# alpha_MLH = acul.alpha_t2(fMLH, 1000, 0, 10, 100, 300)
# alpha_H = acul.alpha_t2(fH, 1000, 0, 10, 100, 300)
# alpha_EH = acul.alpha_t2(fEH, 1000, 0, 10, 100, 300)
#
# alpha_VB = acul.alpha_t2(fVB, 1000, 0, 10, 100, 300)
# alpha_B = acul.alpha_t2(fB, 1000, 0, 10, 100, 300)
# alpha_SB = acul.alpha_t2(fSB, 1000, 0, 10, 100, 300)
# alpha_F = acul.alpha_t2(fF, 1000, 0, 10, 100, 300)
# alpha_SG = acul.alpha_t2(fSG, 1000, 0, 10, 100, 300)
# alpha_G = acul.alpha_t2(fG, 1000, 0, 10, 100, 300)
# alpha_VG = acul.alpha_t2(fVG, 1000, 0, 10, 100, 300)
#
# alpha_U = acul.alpha_t2(fU, 1000, 0, 10, 100, 300)
# alpha_MLU = acul.alpha_t2(fMLU, 1000, 0, 10, 100, 300)
# alpha_MLI = acul.alpha_t2(fMLI, 1000, 0, 10, 100, 300)
# alpha_VI = acul.alpha_t2(fVI, 1000, 0, 10, 100, 300)


def alpha_antonym(a):
    rows, cols = a.shape
    out = np.zeros_like(a)

    for i in range(rows):
        out[i, 1] = 10 - a[i, 0]
        out[i, 0] = 10 - a[i, 1]
        out[i, 2] = a[i, 2]

    return out


# rrr = r_exponent(0.8)
#
# r = -1
#
# zAU = [alpha_antonym(alpha_FMLH[0]), alpha_antonym(alpha_FMLH[0]), alpha_EH[0], alpha_G[0]]
# zAL = [alpha_antonym(alpha_FMLH[1]), alpha_antonym(alpha_FMLH[1]), alpha_EH[1], alpha_G[1]]
# wAU = [alpha_U[0], alpha_MLU[0], alpha_VI[0], alpha_MLU[0]]
# wAL = [alpha_U[1], alpha_MLU[1], alpha_VI[1], alpha_MLU[1]]
#
# alpha_wpm = acpm.alpha_to_alpha_t2wpm(zAU, zAL, wAU, wAL, r)
#
# cwpm = t2c.t2_centroid(alpha_wpm[0], alpha_wpm[1], 300)
# dwpm = t2c.defuzz(cwpm)
# cl = cwpm[0]
# cr = cwpm[1]
#
# alpha_lwa = acpm.alpha_to_alpha_t2wpm(zAU, zAL, wAU, wAL, 1)
#
# clwa = t2c.t2_centroid(alpha_lwa[0], alpha_lwa[1], 300)
# dlwa = t2c.defuzz(clwa)
#
#
# def wpmU(x):
#     return mu_sf(x, alpha_wpm[0])
#
#
# def wpmL(x):
#     return mu_sf(x, alpha_wpm[1])
#
#
# def lwaU(x):
#     return mu_sf(x, alpha_lwa[0])
#
#
# def lwaL(x):
#     return mu_sf(x, alpha_lwa[1])
#
#
# specAlcw = cwpm
# specAldw = dwpm
# specAlcl = clwa
# specAldl = dlwa
#
# fou_wpm = fou_points.fouset(wpmU, wpmL, 0, 10, 0.15, 0.05)
#
# foulwa = fou_points.fouset(lwaU, lwaL, 0, 10, 0.11, 0.06)

# j = range(0, fou_wpm.shape[0])
# k = range(0, foulwa.shape[0])
#
#
# # Generate x values
# x_values = np.arange(0, 10.01, 0.01)
#
# # Calculate y values for wpmU, wpmL, lwaU, and lwaL
# y_values_wpmU = [wpmU(x) for x in x_values]
# y_values_wpmL = [wpmL(x) for x in x_values]
# y_values_lwaU = [lwaU(x) for x in x_values]
# y_values_lwaL = [lwaL(x) for x in x_values]
#
# # Assuming fou_wpm and foulwa are defined arrays with the coordinates of the FOU points
# fou_wpm = fou_points.fouset(wpmU, wpmL, 0, 10, 0.15, 0.05)
# foulwa = fou_points.fouset(lwaU, lwaL, 0, 10, 0.11, 0.06)

# # Extracting coordinates for scatter plots
# x_coords_fou_wpm = [coord[0] for coord in fou_wpm]
# y_coords_fou_wpm = [coord[1] for coord in fou_wpm]
# x_coords_fou_lwa = [coord[0] for coord in foulwa]
# y_coords_fou_lwa = [coord[1] for coord in foulwa]
#
# # Create a high-quality plot
# plt.figure(figsize=(12, 8), dpi=300)
#
# # Plot wpmU and wpmL
# plt.plot(x_values, y_values_wpmU, label='wpmU(x)', color='blue', linestyle='-')
# plt.plot(x_values, y_values_wpmL, label='wpmL(x)', color='blue', linestyle='--')
#
# # Scatter plot for fou_wpm points
# plt.scatter(x_coords_fou_wpm, y_coords_fou_wpm, c='blue', s=1, label='fou_wpm')
#
# # Plot lwaU and lwaL
# plt.plot(x_values, y_values_lwaU, label='lwaU(x)', color='red', linestyle='-')
# plt.plot(x_values, y_values_lwaL, label='lwaL(x)', color='red', linestyle='--')
#
# # Scatter plot for foulwa points
# plt.scatter(x_coords_fou_lwa, y_coords_fou_lwa, c='red', s=1, label='fou_lwa')
#
# # Plot the centroids
# plt.scatter(cwpm[0], 1, c='green', label='Centroid Left')
# plt.scatter(cwpm[1], 1, c='green', label='Centroid Right')
#
# # Plot the defuzzified centroid
# plt.scatter(dwpm, 1, c='blue', label='Defuzzified Centroid')
#
# # Customize the plot
# plt.xlabel('x')
# plt.ylabel('Membership Degree')
# plt.title('LWA and WPM Type-2 Fuzzy Sets')
# plt.legend()
# plt.grid(True)
#
# # Display the plot
# plt.show()


wAU1 = 0.653
wAL1 = 0.653
wAU2 = 0.065
wAL2 = 0.065
r1 = 1.449
r2 = -10


def alpha_w_constant(constant):
    w = np.zeros(101)
    alpha = np.zeros(101)

    for i in range(101):
        w[i] = constant
        alpha[i] = i / 100

    # Stack arrays horizontally to create a row-major structure
    out = [0, 0]
    out[0] = np.array([w, w, alpha]).T  # Transpose to switch to row-major
    out[1] = one_minus(out[0])

    return out


# awAU1 = alpha_w_constant(wAU1)
# awAL1 = alpha_w_constant(wAL1)
# awAU2 = alpha_w_constant(wAU2)
# awAL2 = alpha_w_constant(wAL2)
#
# zAU1 = alpha_EH[0]
# zAL1 = alpha_EH[1]
#
# wAU3 = [alpha_U[0], alpha_MLU[0], alpha_MLU[0]]
# wAL3 = [alpha_U[1], alpha_MLU[1], alpha_MLU[1]]
#
# zAU3 = [alpha_antonym(alpha_FMLH[0]), alpha_antonym(alpha_FMLH[0]), alpha_G[0]]
#
# zAL3 = [alpha_antonym(alpha_FMLH[1]), alpha_antonym(alpha_FMLH[1]), alpha_G[1]]
#
# alpha_wpm3 = acpm.alpha_to_alpha_t2wpm(zAU3, zAL3, wAU3, wAL3, -1)
#
# zAU2 = [zAU1, alpha_wpm3[0]]
# zAL2 = [zAL1, alpha_wpm3[1]]
#
# alpha_wpm_disj = acpm.alpha_to_alpha_t2wpm(zAU2, zAL2, awAU1, awAL1, r1)
#
# zAU4 = [zAU1, alpha_wpm_disj[0]]
# zAL4 = [zAL1, alpha_wpm_disj[1]]
#
# alpha_wpm_conj = acpm.alpha_to_alpha_t2wpm(zAU4, zAL4, awAU2, awAL2, r2)
#
#
# def cpawpmU(x):
#     return mu_sf(x, alpha_wpm_conj[0])
#
#
# def cpawpmL(x):
#     return mu_sf(x, alpha_wpm_conj[1])
#
#
# foucpa = fou_points.fouset(cpawpmU, cpawpmL, 0, 10, 0.15, 0.05)
#
# zAU = [alpha_antonym(alpha_FMLH[0]), alpha_antonym(alpha_FMLH[0]), alpha_EH[0], alpha_G[0]]
# zAL = [alpha_antonym(alpha_FMLH[1]), alpha_antonym(alpha_FMLH[1]), alpha_EH[1], alpha_G[1]]
# wAU = [alpha_U[0], alpha_MLU[0], alpha_VI[0], alpha_MLU[0]]
# wAL = [alpha_U[1], alpha_MLU[1], alpha_VI[1], alpha_MLU[1]]
#
# cwpm = t2c.t2_centroid(alpha_wpm_conj[0], alpha_wpm_conj[1], 300)
# dwpm = t2c.defuzz(cwpm)
# cl = cwpm[0]
# cr = cwpm[1]
#
# alpha_lwa = acpm.alpha_to_alpha_t2wpm(zAU, zAL, wAU, wAL, 1)
# clwa = t2c.t2_centroid(alpha_lwa[0], alpha_lwa[1], 300)
# dlwa = t2c.defuzz(clwa)


# def lwaU(x):
#     return mu_sf(x, alpha_lwa[0])
#
#
# def lwaL(x):
#     return mu_sf(x, alpha_lwa[1])
#
# # specAlcw = cwpm
# # specAlcl = clwa
# # specAldw = dwpm
# # specAldl = dlwa
#
# j = range(0, foucpa.shape[0])
# k = range(0, foulwa.shape[0])

# # Generate x values
# x_values = np.arange(0, 10.01, 0.01)
#
# # Calculate y values for wpmU, wpmL, lwaU, and lwaL
# y_values_cpaU = [cpawpmU(x) for x in x_values]
# y_values_cpaL = [cpawpmL(x) for x in x_values]
# y_values_lwaU = [lwaU(x) for x in x_values]
# y_values_lwaL = [lwaL(x) for x in x_values]
#
# # Assuming fou_wpm and foulwa are defined arrays with the coordinates of the FOU points
# fou_wpm = fou_points.fouset(cpawpmU, cpawpmL, 0, 10, 0.15, 0.05)
# foulwa = fou_points.fouset(lwaU, lwaL, 0, 10, 0.11, 0.06)
#
# # Extracting coordinates for scatter plots
# x_coords_foucpa = [coord[0] for coord in foucpa]
# y_coords_foucpa = [coord[1] for coord in foucpa]
# x_coords_fou_lwa = [coord[0] for coord in foulwa]
# y_coords_fou_lwa = [coord[1] for coord in foulwa]
#
# # Create a high-quality plot
# plt.figure(figsize=(12, 8), dpi=300)
#
# # Plot wpmU and wpmL
# plt.plot(x_values, y_values_cpaU, label='wpmU(x)', color='blue', linestyle='-')
# plt.plot(x_values, y_values_cpaL, label='wpmL(x)', color='blue', linestyle='--')
#
# # Scatter plot for foucpa points
# plt.scatter(x_coords_foucpa, y_coords_foucpa, c='blue', s=1, label='fou_cpa')
#
# # Plot lwaU and lwaL
# plt.plot(x_values, y_values_lwaU, label='lwaU(x)', color='red', linestyle='-')
# plt.plot(x_values, y_values_lwaL, label='lwaL(x)', color='red', linestyle='--')
#
# # Scatter plot for foulwa points
# plt.scatter(x_coords_fou_lwa, y_coords_fou_lwa, c='red', s=1, label='fou_lwa')
#
# # Plot the centroids
# plt.scatter(cwpm[0], 1, c='green', label='Centroid Left')
# plt.scatter(cwpm[1], 1, c='green', label='Centroid Right')
#
# # Plot the defuzzified centroid
# plt.scatter(dwpm, 1, c='blue', label='Defuzzified Centroid')
#
# # Customize the plot
# plt.xlabel('x')
# plt.ylabel('Membership Degree')
# plt.title('LWA and WPM Type-2 Fuzzy Sets')
# plt.legend()
# plt.grid(True)
#
# # Display the plot
# plt.show()


# p1 = r_exponent(6 / 16)
# p2 = r_exponent(12 / 16)


def w_table(d, c):
    # Initialize out to array headings
    out = [["Disjunction power = ", d, "Conjunction power = ", c]]
    out.append([" ", " ", " ", " "])
    out.append(["Penalty", "Reward", "w1", "w2"])

    # Define penalty and reward matrices
    penalties = np.array([-10, -15, -20, -25])
    rewards = np.array([10, 15, 20, 25])

    for P in penalties:
        for R in rewards:
            try:
                # Calculate w1 and w2
                w = wdc(d, c, P, R)
            except Exception as e:
                # Handle any exceptions that might occur in wDC
                print(f"An error occurred: {e}")
                return np.array([[P], [R]])

            # Append the results to the output
            out.append([P, R, w[0], w[1]])

    return np.array(out)


# ww = w_table(p1, p2)
#
# print(ww)
