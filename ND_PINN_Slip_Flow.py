import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import os
tf.keras.backend.set_floatx('float64')


# ============================================================
# GLOBAL SETTINGS (ADD HERE)
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 16

def save_fig(name):
    import os
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(f"{name}.tiff", dpi=300, format="tiff", bbox_inches="tight")
# ============================================================
# RESULTS DIRECTORY
# ============================================================
RESULTS_DIR = "Results_Slip"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ============================================================
# PARAMETERS
# ============================================================
L = 10.0
beta1 =0.1
beta2 =0.1
beta3 =0.1
beta4 =0.1
alpha = 0.5
BiT= 0.5
Pr = 0.71
Nb = 0.1
Nt = 0.1
Sc = 1.0
iter = 15000
Nf = 800

eta = np.linspace(0, 1, Nf).reshape(-1, 1)
eta_tf = tf.convert_to_tensor(eta, dtype=tf.float64)

scale = 1.0 / L

# ============================================================
# MODEL
# ============================================================
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(80, activation='tanh'),
    tf.keras.layers.Dense(80, activation='tanh'),
    tf.keras.layers.Dense(80, activation='tanh'),
    tf.keras.layers.Dense(80, activation='tanh'),
    tf.keras.layers.Dense(4, activation=None)  # f,g,T,P
])

# ============================================================
# DERIVATIVES (3rd order for f,g | 2nd order for T,P)
# ============================================================
def derivatives(x):

    with tf.GradientTape(persistent=True) as t3:
        t3.watch(x)

        with tf.GradientTape(persistent=True) as t2:
            t2.watch(x)

            with tf.GradientTape(persistent=True) as t1:
                t1.watch(x)

                out = model(x)
                f = out[:,0:1]
                g = out[:,1:2]
                T = out[:,2:3]
                P = out[:,3:4]

            f1 = t1.gradient(f, x)
            g1 = t1.gradient(g, x)
            T1 = t1.gradient(T, x)
            P1 = t1.gradient(P, x)

        f2 = t2.gradient(f1, x)
        g2 = t2.gradient(g1, x)
        T2 = t2.gradient(T1, x)
        P2 = t2.gradient(P1, x)

    f3 = t3.gradient(f2, x)
    g3 = t3.gradient(g2, x)

    del t1, t2, t3

    return f,f1,f2,f3, g,g1,g2,g3, T,T1,T2, P,P1,P2


# ============================================================
# LOSS FUNCTION (CORRECTED)
# ============================================================
def loss_fn():

    f,f1,f2,f3, g,g1,g2,g3, T,T1,T2, P,P1,P2 = derivatives(eta_tf)

    # scaling (convert all derivatives to physical derivatives d/dx)
    f1 *= scale
    f2 *= scale**2
    f3 *= scale**3

    g1 *= scale
    g2 *= scale**2
    g3 *= scale**3

    T1 *= scale
    T2 *= scale**2

    P1 *= scale
    P2 *= scale**2

    # ========================================================
    # PDE RESIDUALS (using physical derivatives)
    # ========================================================
    R1 = f3 + (f + g)*f2 - f1**2
    R2 = g3 + (f + g)*g2 - g1**2
    R3 = T2 + Pr*(f+g)*T1 + Pr*Nb*T1*P1 + Pr*Nt*T1**2
    R4 = P2 + Sc*(f+g)*P1 + (Nt/Nb)*T2

    # ========================================================
    # BOUNDARY CONDITIONS (using scaled physical derivatives)
    # ========================================================
    def col(x): return tf.reshape(x, [-1,1])

    # Extract boundary values from the already-computed arrays
    f0 = f[0]      # f(0)
    fL = f[-1]     # f(L) - not used in BC but available
    
    g0 = g[0]      # g(0)
    gL = g[-1]     # g(L) - not used in BC but available
    
    T0 = T[0]      # T(0)
    TL = T[-1]     # T(L)
    
    P0 = P[0]      # P(0)
    PL = P[-1]     # P(L)
    
    # First derivatives at boundaries (already scaled to physical)
    f1_0 = f1[0]   # f'(0)
    f2_0 = f2[0]   # f''(0)
    f1_L = f1[-1]  # f'(L)
    
    g1_0 = g1[0]   # g'(0)
    g2_0 = g2[0]   # g''(0)
    g1_L = g1[-1]  # g'(L)
    
    T1_0 = T1[0]   # T'(0)
    P1_0 = P1[0]   # P'(0)

    # Boundary condition residuals
    bc = tf.concat([
        col(f0),                          # BC1: f(0) = 0
        col(f1_0 - 1.0 - beta1 * f2_0),   # BC2: f'(0) = 1 + beta1*f''(0)
        col(f1_L),                        # BC3: f'(L) = 0
        
        col(g0),                          # BC4: g(0) = 0
        col(g1_0 - alpha - beta2 * g2_0), # BC5: g'(0) = alpha + beta2*g''(0)
        col(g1_L),                        # BC6: g'(L) = 0
        
        col(T0 - 1.0 - beta3 * T1_0),     # BC7: T(0) -1 - beta3 * T'(0)
        col(TL),                          # BC8: T(L) = 0
        
        col(P0 - 1.0 - beta4 * P1_0),     # BC9: P(0) - 1 - beta4*P'(0)
        col(PL)                           # BC10: P(L) = 0
    ], axis=0)

    return (tf.reduce_mean(R1**2) +
            tf.reduce_mean(R2**2) +
            tf.reduce_mean(R3**2) +
            tf.reduce_mean(R4**2) +
            tf.reduce_mean(bc**2))
# ============================================================
# TRAINING (ADAM)
# ============================================================
opt = tf.keras.optimizers.Adam(1e-3)

@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = loss_fn()
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

loss_hist = []
for i in range(iter):
    l = train_step()
    loss_hist.append(l.numpy())
    if i % 100 == 0:
        tf.print("Epoch:", i, "Loss:", l)


# ============================================================
# L-BFGS
# ============================================================
#def get_weights():
#   return tf.concat([tf.reshape(w, [-1]) for w in model.trainable_variables], axis=0)

#def set_weights(w):
#    idx = 0
#    for v in model.trainable_variables:
#        size = tf.size(v)
#        v.assign(tf.reshape(w[idx:idx+size], v.shape))
#        idx += size

#def lbfgs_loss(w):
#    set_weights(w)
#    with tf.GradientTape() as tape:
#        loss = loss_fn()
#    grads = tape.gradient(loss, model.trainable_variables)
#    grads = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
#    return loss, grads

#init = get_weights()

#results = tfp.optimizer.lbfgs_minimize(
#   value_and_gradients_function=lbfgs_loss,
#   initial_position=init,
#   max_iterations=20000
#)

#set_weights(results.position)

# ============================================================
# PDE RESIDUALS (POST-TRAINING DIAGNOSTIC)
# ============================================================
with tf.GradientTape(persistent=True) as t:
    t.watch(eta_tf)
    f,f1,f2,f3, g,g1,g2,g3, T,T1,T2, P,P1,P2 = derivatives(eta_tf)
# Apply SAME scaling as in loss_fn
f1 *= scale
f2 *= scale**2
f3 *= scale**3

g1 *= scale
g2 *= scale**2
g3 *= scale**3

T1 *= scale
T2 *= scale**2

P1 *= scale
P2 *= scale**2

R1 = f3 + (f + g)*f2 - f1**2
R2 = g3 + (f + g)*g2 - g1**2
R3 = T2 + Pr*(f+g)*T1 + Pr*Nb*T1*P1 + Pr*Nt*T1**2
R4 = P2 + Sc*(f+g)*P1 + (Nt/Nb)*T2

R_total = (np.abs(R1.numpy()) +
           np.abs(R2.numpy()) +
           np.abs(R3.numpy()) +
           np.abs(R4.numpy()))
# ============================================================
# PINN RESULTS
# ============================================================
# FIRST define eta_plot (normalized to physical coordinates)
eta_plot = (eta * L).flatten()

out = model(eta_tf).numpy()

f_pinn = out[:,0]
g_pinn = out[:,1]
T_pinn = out[:,2]
P_pinn = out[:,3]

fp_pinn  = f1.numpy().flatten()
gp_pinn  = g1.numpy().flatten()

fpp_pinn = f2.numpy().flatten()
gpp_pinn = g2.numpy().flatten()

Tp_pinn  = T1.numpy().flatten()
Pp_pinn  = P1.numpy().flatten()

Tp_pinn_0 = Tp_pinn[0]
Pp_pinn_0 = Pp_pinn[0]
# ============================================================
# NUMERICAL SOLVER (solve_bvp)
# ============================================================
def odefun(x, y):

    f, fp, fpp, g, gp, gpp, T, Tp, P, Pp = y

    fppp = -(f + g)*fpp + fp**2
    gppp = -(f + g)*gpp + gp**2

    Tpp = -Pr*(f + g)*Tp - Pr*Nb*Tp*Pp - Pr*Nt*Tp**2
    Ppp = -(Sc*(f + g)*Pp) - (Nt/Nb)*Tpp

    return np.vstack((fp,
                      fpp,
                      fppp,
                      gp,
                      gpp,
                      gppp,
                      Tp,
                      Tpp,
                      Pp,
                      Ppp))


def bc(ya, yb):

    return np.array([
        ya[0],          # f(0)
        ya[1] - 1.0 - beta1*ya[2],      # f'(0)
        yb[1],          # f'(L)

        ya[3],          # g(0)
        ya[4] - alpha - beta2*ya[5],    # g'(0)
        yb[4],          # g'(L)

        ya[6] - 1.0 - beta3*ya[7],      # T(0)
        yb[6],          # T(L)

        ya[8] - 1.0 - beta4*ya[9],      # P(0)
        yb[8]           # P(L)
    ])
# ============================================================
# DOMAIN
# ============================================================
x_mesh = np.linspace(0, L, 800)

y_init = np.zeros((10, x_mesh.size))

# initial guesses (important for convergence)
y_init[0] = x_mesh / L
y_init[1] = np.exp(-x_mesh)
y_init[3] = x_mesh / L
y_init[4] = alpha * np.exp(-x_mesh)
y_init[6] = np.exp(-x_mesh)
y_init[8] = np.exp(-x_mesh)
# ============================================================
# SOLVE
# ============================================================
sol = solve_bvp(odefun, bc, x_mesh, y_init)
# ============================================================
# NUMERICAL SOLUTION (same grid)
# ============================================================
f_num_full = sol.sol(x_mesh)[0]
g_num_full = sol.sol(x_mesh)[3]
T_num_full = sol.sol(x_mesh)[6]
P_num_full = sol.sol(x_mesh)[8]
fp_num_full = np.gradient(f_num_full, x_mesh)
gp_num_full = np.gradient(g_num_full, x_mesh)

# Interpolate numerical solution to PINN's physical grid (eta_plot)
f_num = interp1d(x_mesh, f_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)
g_num = interp1d(x_mesh, g_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)
T_num = interp1d(x_mesh, T_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)
P_num = interp1d(x_mesh, P_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)
fp_num = interp1d(x_mesh, fp_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)
gp_num = interp1d(x_mesh, gp_num_full, kind='cubic', fill_value='extrapolate')(eta_plot)

# ============================================================
# ERRORS
# ============================================================
err_f  = np.abs(f_pinn  - f_num)
err_fp = np.abs(fp_pinn - fp_num)

err_g  = np.abs(g_pinn  - g_num)
err_gp = np.abs(gp_pinn - gp_num)

err_T  = np.abs(T_pinn  - T_num)
err_P  = np.abs(P_pinn  - P_num)

# ============================================================
# L1 and L2 ERRORS
# ============================================================

L1_f  = np.mean(err_f)
L2_f  = np.sqrt(np.mean(err_f**2))

L1_fp = np.mean(err_fp)
L2_fp = np.sqrt(np.mean(err_fp**2))

L1_g  = np.mean(err_g)
L2_g  = np.sqrt(np.mean(err_g**2))

L1_gp = np.mean(err_gp)
L2_gp = np.sqrt(np.mean(err_gp**2))

L1_T  = np.mean(err_T)
L2_T  = np.sqrt(np.mean(err_T**2))

L1_P  = np.mean(err_P)
L2_P  = np.sqrt(np.mean(err_P**2))

# ============================================================
bc_labels = [
    r"$f(a)$",
    r"$f'(a)$",
    r"$f'(b)$",
    r"$g(a)$",
    r"$g'(a)$",
    r"$g'(b)$",
    r"$\theta(a)$",
    r"$\theta(b)$",
    r"$\phi(a)$",
    r"$\phi(b)$"
]

bc_error = np.abs(np.array([
    f_pinn[0],
    fp_pinn[0] - 1.0 - beta1*fpp_pinn[0],
    fp_pinn[-1],
    g_pinn[0],
    gp_pinn[0] - alpha - beta2*gpp_pinn[0],
    gp_pinn[-1],
    T_pinn[0] - 1.0 - beta3*Tp_pinn_0,
    T_pinn[-1],
    P_pinn[0] - 1.0 - beta4*Pp_pinn_0,
    P_pinn[-1]
]))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
# ============================================================
# (1) Loss History
# ============================================================
axs[0].plot(loss_hist, 'b-', linewidth=2)
axs[0].set_yscale('log')
axs[0].set_title("    (a)", fontsize=14)
axs[0].set_xlabel("Iteration", fontsize=14)
axs[0].set_ylabel("Loss", fontsize=14)
axs[0].tick_params(labelsize=13)
axs[0].grid(True, alpha=0.3)
# ============================================================
# (2) PDE Residual Distribution
# ============================================================
axs[1].plot(eta_plot, R_total, 'm-', linewidth=2)
axs[1].set_yscale('log')
axs[1].set_title("    (b)", fontsize=14)
axs[1].set_xlabel(r"$\eta$", fontsize=14)
axs[1].set_ylabel("PDE Residuals", fontsize=14)
axs[1].tick_params(labelsize=13)
axs[1].grid(True, alpha=0.3)

# ============================================================
# (1) Boundary Condition Errors
# ============================================================
axs[2].semilogy(bc_labels, bc_error, 'ro-', linewidth=2, markersize=6)
axs[2].set_title("    (c)", fontsize=14)
axs[2].set_ylabel("Boundary Condition Residuals", fontsize=14)
axs[2].tick_params(axis='x', rotation=35, labelsize=13)
axs[2].tick_params(axis='y', labelsize=13)
axs[2].grid(True, which='both', linestyle='--', alpha=0.4)
# ============================================================
# FORMATTING
# ============================================================
plt.tight_layout()
# SAVE
save_fig(f"{RESULTS_DIR}/PINN_Diagnostics_Combined")
plt.show()
# ============================================================
# ALL PROFILES + ALL ERRORS IN ONE FIGURE
# ============================================================
fig, axs = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns

# ---------------- Row 1: Profiles (f, f', g) ----------------
axs[0, 0].plot(eta_plot, f_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[0, 0].plot(eta_plot, f_num, 'k--', linewidth=2, label='Numerical')
axs[0, 0].set_title("    (a)", fontsize=16)
axs[0, 0].set_ylabel(r"$f(\eta)$", fontsize=16)
axs[0, 0].set_xlabel(r"$\eta$", fontsize=16)
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# Error inset for f
axins_f = axs[0, 0].inset_axes([0.60, 0.40, 0.35, 0.30])
axins_f.plot(eta_plot, err_f, 'k', lw=1.2)
axins_f.set_yscale('log')
axins_f.set_title("Error", fontsize=13)
axins_f.tick_params(axis='both', labelsize=13)
axins_f.grid(True, alpha=0.3)
err_min = min(err_f)
err_max = max(err_f)
axins_f.set_yticks([err_min, err_max])
axins_f.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_f.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_f.yaxis.set_minor_locator(plt.NullLocator())

axs[0, 1].plot(eta_plot, fp_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[0, 1].plot(eta_plot, fp_num, 'k--', linewidth=2, label='Numerical')
axs[0, 1].set_title("    (b)", fontsize=16)
axs[0, 1].set_ylabel(r"$f'(\eta)$", fontsize=16)
axs[0, 1].set_xlabel(r"$\eta$", fontsize=16)
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend()

# Error inset for f'
axins_fp = axs[0, 1].inset_axes([0.60, 0.20, 0.35, 0.30])
axins_fp.plot(eta_plot, err_fp, 'k', lw=1.2)
axins_fp.set_yscale('log')
axins_fp.set_title("Error", fontsize=13)
axins_fp.tick_params(axis='both', labelsize=13)
axins_fp.grid(True, alpha=0.3)

err_min = min(err_fp)
err_max = max(err_fp)
axins_fp.set_yticks([err_min, err_max])
axins_fp.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_fp.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_fp.yaxis.set_minor_locator(plt.NullLocator())

axs[0, 2].plot(eta_plot, g_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[0, 2].plot(eta_plot, g_num, 'k--', linewidth=2, label='Numerical')
axs[0, 2].set_title("    (c)", fontsize=14)
axs[0, 2].set_ylabel(r"$g(\eta)$", fontsize=16)
axs[0, 2].set_xlabel(r"$\eta$", fontsize=16)
axs[0, 2].grid(True, alpha=0.3)
axs[0, 2].legend()

# Error inset for g
axins_g = axs[0, 2].inset_axes([0.60, 0.40, 0.35, 0.30])
axins_g.plot(eta_plot, err_g, 'k', lw=1.2)
axins_g.set_yscale('log')
axins_g.set_title("Error", fontsize=13)
axins_g.tick_params(axis='both', labelsize=13)
axins_g.grid(True, alpha=0.3)

err_min = min(err_g)
err_max = max(err_g)
axins_g.set_yticks([err_min, err_max])
axins_g.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_g.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_g.yaxis.set_minor_locator(plt.NullLocator())

# ---------------- Row 2: Profiles (g', θ, φ) ----------------
axs[1, 0].plot(eta_plot, gp_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[1, 0].plot(eta_plot, gp_num, 'k--', linewidth=2, label='Numerical')
axs[1, 0].set_title("    (d)", fontsize=16)
axs[1, 0].set_ylabel(r"$g'(\eta)$", fontsize=16)
axs[1, 0].set_xlabel(r"$\eta$", fontsize=16)
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend()

# Error inset for g'
axins_gp = axs[1, 0].inset_axes([0.60, 0.20, 0.35, 0.30])
axins_gp.plot(eta_plot, err_gp, 'k', lw=1.2)
axins_gp.set_yscale('log')
axins_gp.set_title("Error", fontsize=13)
axins_gp.tick_params(axis='both', labelsize=13)
axins_gp.grid(True, alpha=0.3)

err_min = min(err_gp)
err_max = max(err_gp)
axins_gp.set_yticks([err_min, err_max])
axins_gp.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_gp.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_gp.yaxis.set_minor_locator(plt.NullLocator())


axs[1, 1].plot(eta_plot, T_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[1, 1].plot(eta_plot, T_num, 'k--', linewidth=2, label='Numerical')
axs[1, 1].set_title("    (e)", fontsize=16)
axs[1, 1].set_ylabel(r"$\theta(\eta)$", fontsize=16)
axs[1, 1].set_xlabel(r"$\eta$", fontsize=16)
axs[1, 1].grid(True, alpha=0.3)
axs[1, 1].legend()

# Error inset for θ
axins_T = axs[1, 1].inset_axes([0.60, 0.20, 0.35, 0.30])
axins_T.plot(eta_plot, err_T, 'k', lw=1.2)
axins_T.set_yscale('log')
axins_T.set_title("Error", fontsize=13)
axins_T.tick_params(axis='both', labelsize=13)
axins_T.grid(True, alpha=0.3)

err_min = min(err_T)
err_max = max(err_T)
axins_T.set_yticks([])
axins_T.set_yticks([err_min, err_max])
axins_T.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_T.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_T.yaxis.set_minor_locator(plt.NullLocator())



axs[1, 2].plot(eta_plot, P_pinn, 'r-', linewidth=2, label='ND-PINN')
axs[1, 2].plot(eta_plot, P_num, 'k--', linewidth=2, label='Numerical')
axs[1, 2].set_title("    (f)", fontsize=16)
axs[1, 2].set_ylabel(r"$\phi(\eta)$", fontsize=16)

axs[1, 2].set_xlabel(r"$\eta$", fontsize=16)
axs[1, 2].grid(True, alpha=0.3)
axs[1, 2].legend()

# Error inset for φ
axins_P = axs[1, 2].inset_axes([0.60, 0.20, 0.35, 0.30])
axins_P.plot(eta_plot, err_P, 'k', lw=1.2)
axins_P.set_yscale('log')
axins_P.set_title("Error", fontsize=13)
axins_P.tick_params(axis='both', labelsize=13)
axins_P.grid(True, alpha=0.3)

err_min = min(err_P)
err_max = max(err_P)
axins_P.set_yticks([])
axins_P.set_yticks([err_min, err_max])
axins_P.set_yticklabels([f"$10^{{{int(np.log10(err_min))}}}$", f"$10^{{{int(np.log10(err_max))}}}$"])
axins_P.yaxis.set_major_locator(plt.FixedLocator([err_min, err_max]))
axins_P.yaxis.set_minor_locator(plt.NullLocator())

# ---------------- Formatting ----------------
for ax in axs.flat:
    ax.tick_params(axis='both', labelsize=16, width=1.2, length=5)

# ---------------- Adjust layout ----------------
plt.tight_layout()

# ---------------- AUTO SAVE ----------------
save_fig(f"{RESULTS_DIR}/All_Profiles_Errors")

# ---------------- Show ----------------
plt.show()

print("\n================ L1 & L2 ERROR TABLE ================\n")

print(f"{'Variable':<10}{'L1 Error':<20}{'L2 Error'}")
print("-"*50)

print(f"{'f':<10}{L1_f:<20.3e}{L2_f:.3e}")
print(f"{'f\'':<10}{L1_fp:<20.3e}{L2_fp:.3e}")
print(f"{'g':<10}{L1_g:<20.3e}{L2_g:.3e}")
print(f"{'g\'':<10}{L1_gp:<20.3e}{L2_gp:.3e}")
print(f"{'T':<10}{L1_T:<20.3e}{L2_T:.3e}")
print(f"{'P':<10}{L1_P:<20.3e}{L2_P:.3e}")