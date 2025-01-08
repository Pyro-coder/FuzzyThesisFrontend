import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import EKM
import trapezoid
import select_f
import fou_points

def alphamaxlmf(ax, aw):
    """
    Finds the array with the smallest number of rows between ax and aw,
    each being a numpy array containing arrays of α-cuts for the x and w LMFs.
    Returns the index of the minimum array and the maximum alpha value from it.
    """
    imax = float('inf')
    amax = np.array(0.0)
    for i in range(len(ax)):
        if len(ax[i]) < imax:
            imax = len(ax[i])
            amax = ax[i][-1][-1]
        if len(aw[i]) < imax:
            imax = len(aw[i])
            amax = aw[i][-1][-1]
    return np.array([imax - 1]), amax

def alphatoalphat2wpm(aux, alx, auw, alw, r):
    n = len(aux)  # Number of terms
    m = len(aux[0])  # Number of alpha-cuts
    
    xuab, wucd, xlab, wlcd = [], [], [], []
    for j in range(m):
        # UMF alpha-cuts
        xuab_j, wucd_j = [], []
        for i in range(n):
            xuab_j.append((aux[i][j][0].item(), aux[i][j][1].item(), aux[i][j][2].item()))
            wucd_j.append((auw[i][j][0].item(), auw[i][j][1].item(), auw[i][j][2].item()))
        xuab.append(xuab_j)
        wucd.append(wucd_j)
        
        # LMF alpha-cuts
        if r != float('-inf') and r != float('inf'):
            hmin, _ = alphamaxlmf(np.array([x[0] for x in alx]), np.array([x[0] for x in alw]))
            hmin = hmin.item()
            xlab_j, wlcd_j = [], []
            for i in range(n):
                xlab_j.append((alx[i][j][0].item(), alx[i][j][1].item(), alx[i][j][2].item()))
                wlcd_j.append((alw[i][j][0].item(), alw[i][j][1].item(), alw[i][j][2].item()))
            xlab.append(xlab_j)
            wlcd.append(wlcd_j)

    # Compute WPM for UMF/LMF alpha-cuts
    zu, zl = [], []
    for j in range(m):
        # UMF
        zu.append(EKM.t2wpm(xuab[j], wucd[j], r))
    for j in range(min(m, int(hmin) + 1)):
        # LMF
        if r not in [float('-inf'), float('inf')]:
            zl.append(EKM.t2wpm(xlab[j], wlcd[j], r))
        else:
            # Handle infinite r cases
            zl.append(xlab[j])  # Directly use xlab for infinite cases

    return [zu, zl]

def z_func(x, params, func_type='trap'):
    result = [trapezoid.trap(x, *p) for p in params]
    return np.array(result)

# Parameters for the membership functions
params_zu = [[1, 0.1, 0.8, 1.2, 2], [1, 0.1, 1, 2, 3]]
params_zl = [[0.8, 0.5, 0.8, 1.2, 1.5], [0.7, 0.5, 1.5, 1.8, 2.5]]
params_wu = [[1, 0, 0.7, 1.3, 2], [1, 0, 1.8, 2.2, 3]]
params_wl = [[0.8, 0, 1, 1, 2], [0.6, 1, 1.5, 2, 2.5]]

# Partial functions for membership calculation with predefined parameters
zu_func = partial(z_func, params=params_zu)
wu_func = partial(z_func, params=params_wu)
zl_func = partial(z_func, params=params_zl)
wl_func = partial(z_func, params=params_wl)

# Select function adaptations for each membership calculation
def z0u(x): return select_f.select_f(zu_func, x, 0)
def z1u(x): return select_f.select_f(zu_func, x, 1)
def w0u(x): return select_f.select_f(wu_func, x, 0)
def w1u(x): return select_f.select_f(wu_func, x, 1)
def z0l(x): return select_f.select_f(zl_func, x, 0)
def z1l(x): return select_f.select_f(zl_func, x, 1)
def w0l(x): return select_f.select_f(wl_func, x, 0)
def w1l(x): return select_f.select_f(wl_func, x, 1)

# Generating FOU sets using the 'fouset' function from the 'fou_points' module
z0fou = fou_points.fouset(z0u, z0l, 0, 10, 0.01, 0.012)
z1fou = fou_points.fouset(z1u, z1l, 0, 10, 0.01, 0.013)
w0fou = fou_points.fouset(w0u, w0l, 0, 10, 0.01, 0.012)
w1fou = fou_points.fouset(w1u, w1l, 0, 10, 0.01, 0.013)

# Plotting function for FOU sets
def plot_fou(fou_set, label, color):
    x_vals, y_vals = zip(*sorted(fou_set[1:]))  # Skipping the header and sorting points by x-axis
    plt.scatter(x_vals, y_vals, color=color, label=label, s=10, alpha=0.5)  # Adding alpha for transparency
    plt.plot(x_vals, y_vals, color=color, linewidth=1)  # Connecting points with lines

plt.figure(figsize=(10, 6))
plot_fou(z0fou, 'z0 FOU', 'blue')
plot_fou(z1fou, 'z1 FOU', 'red')
plot_fou(w0fou, 'w0 FOU', 'green')
plot_fou(w1fou, 'w1 FOU', 'purple')

plt.title('Fuzzy Output Sets Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()
