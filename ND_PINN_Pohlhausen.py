import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from scipy.integrate import solve_bvp

tf.keras.backend.set_floatx('float64')

# ============================================================
# GLOBAL SETTINGS
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12

def save_fig(name):
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(f"{name}.tiff", dpi=300, format="tiff", bbox_inches="tight")

# ============================================================
# RESULTS DIRECTORY
# ============================================================
RESULTS_DIR = "Results_Pohlhausen_ND_PINN"
os.makedirs(RESULTS_DIR, exist_ok=True)
# ============================================================
# PARAMETERS
# ============================================================
L = 5.0                     # Domain length (approximating infinity)
beta0 = 1.0
beta  = 1.0
Nf = 800
epochs_adam = 20000
learning_rate = 1e-3

# Physical domain for standard PINN
x_phys = np.linspace(0, L, Nf).reshape(-1, 1)
x_tf = tf.convert_to_tensor(x_phys, dtype=tf.float64)

# Normalized domain for ND-PINN
eta = np.linspace(0, 1, Nf).reshape(-1, 1)
eta_tf = tf.convert_to_tensor(eta, dtype=tf.float64)
scale = 1.0 / L

# ============================================================
# HELPER FUNCTION TO CREATE MODEL
# ============================================================
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,)),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(80, activation='tanh'),
        tf.keras.layers.Dense(1, activation=None)
    ])

# ============================================================
# DERIVATIVES FUNCTIONS
# ============================================================
def derivatives_normalized(model, x):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            f = model(x)
        f1 = t1.gradient(f, x)
        f2 = t2.gradient(f1, x)
    f3 = t2.gradient(f2, x)
    del t1, t2
    return f, f1, f2, f3

def derivatives_physical(model, x):
    with tf.GradientTape(persistent=True) as t2:
        t2.watch(x)
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            f = model(x)
        f1 = t1.gradient(f, x)
        f2 = t2.gradient(f1, x)
    f3 = t2.gradient(f2, x)
    del t1, t2
    return f, f1, f2, f3

# ============================================================
# LOSS FUNCTIONS
# ============================================================
def loss_fn_present():
    f, f1, f2, f3 = derivatives_normalized(model_present, eta_tf)
    R = (1/L)**3 * f3 + beta0 * (1/L)**2 * f * f2 + beta * (1 - (1/L)**2 * f1**2)
    f0 = f[0]
    f1_0 = (1/L) * f1[0]
    f1_L = (1/L) * f1[-1]
    bc = tf.concat([f0, f1_0, f1_L - 1.0], axis=0)
    return tf.reduce_mean(tf.square(R)) + tf.reduce_mean(tf.square(bc))

def loss_fn_standard():
    f, f1, f2, f3 = derivatives_physical(model_standard, x_tf)
    R = f3 + beta0 * f * f2 + beta * (1 - f1**2)
    f0 = f[0]
    f1_0 = f1[0]
    f1_L = f1[-1]
    bc = tf.concat([f0, f1_0, f1_L - 1.0], axis=0)
    return tf.reduce_mean(tf.square(R)) + tf.reduce_mean(tf.square(bc))

# ============================================================
# ADAM TRAINING
# ============================================================
def train_adam(model, loss_fn, name):
    opt = tf.keras.optimizers.Adam(learning_rate)
    hist = []
    @tf.function
    def step():
        with tf.GradientTape() as tape:
            loss = loss_fn()
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    for i in range(epochs_adam):
        l = step()
        hist.append(l.numpy())
        if i % 100 == 0:
            tf.print(f"{name} Adam - Epoch:", i, "Loss:", l)
    return hist

# ============================================================
# CREATE MODELS
# ============================================================
model_present = create_model()
model_standard = create_model()

# ============================================================
# TRAIN ND-PINN
# ============================================================
print("="*60)
print("TRAINING ND-PINN (normalized domain)")
print("ODE: f''' + β0*f*f'' + β*(1-f'^2)=0, β0=1, β=1")
print("="*60)
start_time = time.time()
loss_adam_present = train_adam(model_present, loss_fn_present, "ND-PINN")
time_present = time.time() - start_time
print(f"ND-PINN CPU time: {time_present:.2f} seconds")

# ============================================================
# TRAIN STANDARD PINN
# ============================================================
print("\n" + "="*60)
print("TRAINING STANDARD PINN (physical domain)")
print("ODE: f''' + β0*f*f'' + β*(1-f'^2)=0, β0=1, β=1")
print("="*60)
start_time = time.time()
loss_adam_standard = train_adam(model_standard, loss_fn_standard, "Standard PINN")
time_standard = time.time() - start_time
print(f"Standard PINN CPU time: {time_standard:.2f} seconds")

# ============================================================
# POST-TRAINING EVALUATION
# ============================================================
# ND-PINN predictions (physical space)
eta_plot = (eta * L).flatten()
f_present = model_present(eta_tf).numpy().flatten()
with tf.GradientTape(persistent=True) as t:
    t.watch(eta_tf)
    _, f1, f2, _ = derivatives_normalized(model_present, eta_tf)
fp_present = (1/L) * f1.numpy().flatten()
f2p_present = (1/L)**2 * f2.numpy().flatten()

# Standard PINN predictions
x_plot = x_phys.flatten()
f_standard = model_standard(x_tf).numpy().flatten()
with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    _, f1, f2, _ = derivatives_physical(model_standard, x_tf)
fp_standard = f1.numpy().flatten()
f2p_standard = f2.numpy().flatten()

# ============================================================
# NUMERICAL REFERENCE SOLUTION (solve_bvp)
# ============================================================
def ode_system(x, y):
    """ y = [f, f', f''] """
    f, fp, fpp = y
    # f''' = -β0*f*f'' - β*(1 - f'^2)
    fppp = -beta0 * f * fpp - beta * (1 - fp*fp)
    return np.vstack([fp, fpp, fppp])

def bc(ya, yb):
    """ f(0)=0, f'(0)=0, f'(L)=1 """
    return np.array([ya[0], ya[1], yb[1] - 1.0])

# Initial guess
x_mesh = np.linspace(0, L, 100)
# Use a simple cubic guess satisfying BCs
f_guess = 0.5 * x_mesh**2 / L
fp_guess = x_mesh / L
fpp_guess = np.ones_like(x_mesh) / L
y_guess = np.vstack([f_guess, fp_guess, fpp_guess])

sol = solve_bvp(ode_system, bc, x_mesh, y_guess, tol=1e-10, max_nodes=5000)
if not sol.success:
    raise RuntimeError("solve_bvp failed: " + sol.message)

# Interpolate numerical solution onto the evaluation grids
x_num = sol.x
f_num = sol.y[0]
fp_num = sol.y[1]
f2p_num = sol.y[2]

def interp_num(xq):
    return np.interp(xq, x_num, f_num), np.interp(xq, x_num, fp_num), np.interp(xq, x_num, f2p_num)

# Numerical values at ND-PINN grid (physical coordinates)
f_num_present, fp_num_present, f2p_num_present = interp_num(eta_plot)
# Numerical values at Standard PINN grid
f_num_std, fp_num_std, f2p_num_std = interp_num(x_plot)

# ============================================================
# COMPUTE ERRORS
# ============================================================
# L2 errors relative to numerical solution
L2_f_present = np.sqrt(np.mean((f_present - f_num_present)**2))
L2_fp_present = np.sqrt(np.mean((fp_present - fp_num_present)**2))
L2_f_std = np.sqrt(np.mean((f_standard - f_num_std)**2))
L2_fp_std = np.sqrt(np.mean((fp_standard - fp_num_std)**2))

# f''(0) from numerical solution
f2p0_num = f2p_num[0]   # at x=0

# ============================================================
# PDE RESIDUALS (same as loss)
# ============================================================
with tf.GradientTape(persistent=True) as t:
    t.watch(eta_tf)
    _, f1, f2, f3 = derivatives_normalized(model_present, eta_tf)
R_present = (f3.numpy().flatten() / L**3 +
             beta0 * f_present * (f2.numpy().flatten() / L**2) +
             beta * (1 - (f1.numpy().flatten() / L)**2))
L1_present = np.mean(np.abs(R_present))
L2_present = np.sqrt(np.mean(R_present**2))

with tf.GradientTape(persistent=True) as t:
    t.watch(x_tf)
    _, f1, f2, f3 = derivatives_physical(model_standard, x_tf)
R_standard = (f3.numpy().flatten() +
              beta0 * f_standard * f2.numpy().flatten() +
              beta * (1 - f1.numpy().flatten()**2))
L1_std = np.mean(np.abs(R_standard))
L2_std = np.sqrt(np.mean(R_standard**2))

# ============================================================
# PLOTTING
# ============================================================
# Loss history
fig, ax = plt.subplots(1, 1, figsize=(4, 4))
ax.plot(loss_adam_present, color='#1f77b4', linestyle='-', linewidth=1.5, label='ND-PINN')
ax.plot(loss_adam_standard, color='#d62728', linestyle='--', linewidth=1.5, label='Standard PINN')
ax.set_yscale('log')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Adam Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(f"{RESULTS_DIR}/Loss_Residuals")
plt.show()

# ---------- Figure 1: Standard PINN vs Numerical ----------
fig1, axs1 = plt.subplots(1, 2, figsize=(9, 4))

# f profile
axs1[0].plot(x_plot, f_standard, 'r--', lw=2, label='Standard PINN')
axs1[0].plot(x_num, f_num, 'k-', lw=1.8, label='Numerical')
axs1[0].set_xlabel(r"$x$")
axs1[0].set_ylabel(r"$f$")
axs1[0].set_title("(a)")
axs1[0].legend()
axs1[0].grid(True, alpha=0.3)

# f' profile
axs1[1].plot(x_plot, fp_standard, 'r--', lw=2, label='Standard PINN')
axs1[1].plot(x_num, fp_num, 'k-', lw=1.8, label='Numerical')
axs1[1].set_xlabel(r"$x$")
axs1[1].set_ylabel(r"$f'$")
axs1[1].set_title("(b)")
axs1[1].legend()
axs1[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig(f"{RESULTS_DIR}/StandardPINN_vs_Numerical")
plt.show()

# ---------- Figure 2: ND-PINN vs Numerical ----------
fig2, axs2 = plt.subplots(1, 2, figsize=(9, 4))

# f profile (physical coordinate = eta*L)
axs2[0].plot(eta_plot, f_present, 'b--', lw=2, label='ND-PINN')
axs2[0].plot(x_num, f_num, 'k-', lw=1.8, label='Numerical')
axs2[0].set_xlabel(r"$x$")
axs2[0].set_ylabel(r"$f$")
axs2[0].set_title("(a)")
axs2[0].legend()
axs2[0].grid(True, alpha=0.3)

# f' profile
axs2[1].plot(eta_plot, fp_present, 'b--', lw=2, label='ND-PINN')
axs2[1].plot(x_num, fp_num, 'k-', lw=1.8, label='Numerical')
axs2[1].set_xlabel(r"$x$")
axs2[1].set_ylabel(r"$f'$")
axs2[1].set_title("(b)")
axs2[1].legend()
axs2[1].grid(True, alpha=0.3)

plt.tight_layout()
save_fig(f"{RESULTS_DIR}/NDPINN_vs_Numerical")
plt.show()


# ============================================================
# TABULAR COMPARISON (per spatial point)
# ============================================================
# Choose sample points (e.g., 6 equally spaced points between 0 and L)
sample_points = np.linspace(0, L, 6)
# For ND-PINN, we need values at physical coordinates (eta_plot)
# For Standard PINN, we have values at x_plot

# Interpolate numerical solution onto sample points
f_num_samples = np.interp(sample_points, x_num, f_num)
fp_num_samples = np.interp(sample_points, x_num, fp_num)

# Standard PINN values at sample points
f_std_samples = np.interp(sample_points, x_plot, f_standard)
fp_std_samples = np.interp(sample_points, x_plot, fp_standard)

# ND-PINN values at sample points (eta_plot is physical coordinate)
f_present_samples = np.interp(sample_points, eta_plot, f_present)
fp_present_samples = np.interp(sample_points, eta_plot, fp_present)

print("\n" + "="*100)
print("COMPARISON OF f(x) PROFILES AT SELECTED LOCATIONS")
print("="*100)
print(f"{'x':>8} | {'Standard PINN':>14} | {'Numerical':>12} | {'Abs Error (Std-Num)':>20} | {'ND-PINN':>12} | {'Numerical':>12} | {'Abs Error (ND-Num)':>20}")
print("-"*100)
for i, xv in enumerate(sample_points):
    print(f"{xv:8.4f} | {f_std_samples[i]:14.6e} | {f_num_samples[i]:12.6e} | {abs(f_std_samples[i]-f_num_samples[i]):20.6e} | {f_present_samples[i]:12.6e} | {f_num_samples[i]:12.6e} | {abs(f_present_samples[i]-f_num_samples[i]):20.6e}")

print("\n" + "="*100)
print("COMPARISON OF f'(x) PROFILES AT SELECTED LOCATIONS")
print("="*100)
print(f"{'x':>8} | {'Standard PINN':>14} | {'Numerical':>12} | {'Abs Error (Std-Num)':>20} | {'ND-PINN':>12} | {'Numerical':>12} | {'Abs Error (ND-Num)':>20}")
print("-"*100)
for i, xv in enumerate(sample_points):
    print(f"{xv:8.4f} | {fp_std_samples[i]:14.6e} | {fp_num_samples[i]:12.6e} | {abs(fp_std_samples[i]-fp_num_samples[i]):20.6e} | {fp_present_samples[i]:12.6e} | {fp_num_samples[i]:12.6e} | {abs(fp_present_samples[i]-fp_num_samples[i]):20.6e}")

# ============================================================
# ADDITIONAL METRIC SUMMARY (as before, but optional)
# ============================================================
print("\n" + "="*80)
print("ADDITIONAL METRIC SUMMARY (over whole domain)")
print("ODE: f''' + β0*f*f'' + β*(1-f'^2)=0, β0=1, β=1")
print("="*80)

# Create DataFrames for Excel export
# DataFrame 1: Main metrics comparison
metrics_data = {
    'Metric': ['f\'\'(0) value', 'Relative error in f\'\'(0) (%)', 'Mean |PDE residual|', 'L2 norm (residual)', 'CPU Time (s)'],
    'ND-PINN': [f2p_present[0], 100*abs(f2p_present[0]-f2p0_num)/abs(f2p0_num), L1_present, L2_present, time_present],
    'Standard PINN': [f2p_standard[0], 100*abs(f2p_standard[0]-f2p0_num)/abs(f2p0_num), L1_std, L2_std, time_standard],
    'Numerical': [f2p0_num, '—', '—', '—', '—']
}
df_metrics = pd.DataFrame(metrics_data)

# Print to console
print(f"\n{'Metric':<35}{'ND-PINN':<20}{'Standard PINN':<20}{'Numerical':<20}")
print("-"*80)
print(f"{'f\"(0) value':<35}{f2p_present[0]:<20.6f}{f2p_standard[0]:<20.6f}{f2p0_num:<20.6f}")
print(f"{'Relative error in f\"(0) (%)':<35}{100*abs(f2p_present[0]-f2p0_num)/abs(f2p0_num):<20.2f}{100*abs(f2p_standard[0]-f2p0_num)/abs(f2p0_num):<20.2f}{'—':<20}")
print(f"{'Mean |PDE residual|':<35}{L1_present:<20.3e}{L1_std:<20.3e}{'—':<20}")
print(f"{'L2 norm (residual)':<35}{L2_present:<20.3e}{L2_std:<20.3e}{'—':<20}")
print(f"{'CPU Time (s)':<35}{time_present:<20.2f}{time_standard:<20.2f}{'—':<20}")

# DataFrame 2: f(x) comparison at sample points
f_comparison_data = {
    'x': sample_points,
    'Standard PINN (f)': f_std_samples,
    'Numerical (f)': f_num_samples,
    'Absolute Error (Std-Num)': np.abs(f_std_samples - f_num_samples),
    'ND-PINN (f)': f_present_samples,
    'Numerical (f)_ND': f_num_samples,
    'Absolute Error (ND-Num)': np.abs(f_present_samples - f_num_samples)
}
df_f_comparison = pd.DataFrame(f_comparison_data)

# DataFrame 3: f'(x) comparison at sample points
fp_comparison_data = {
    'x': sample_points,
    'Standard PINN (fp)': fp_std_samples,
    'Numerical (fp)': fp_num_samples,
    'Absolute Error (Std-Num)': np.abs(fp_std_samples - fp_num_samples),
    'ND-PINN (fp)': fp_present_samples,
    'Numerical (fp)_ND': fp_num_samples,
    'Absolute Error (ND-Num)': np.abs(fp_present_samples - fp_num_samples)
}
df_fp_comparison = pd.DataFrame(fp_comparison_data)

# Save all DataFrames to Excel with multiple sheets
excel_filename = f"{RESULTS_DIR}/comparison_results.xlsx"
with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='Metrics_Summary', index=False)
    df_f_comparison.to_excel(writer, sheet_name='f_Comparison', index=False)
    df_fp_comparison.to_excel(writer, sheet_name='fp_Comparison', index=False)
    
    # Add metadata sheet with problem description
    metadata = pd.DataFrame({
        'Parameter': ['Problem', 'ODE', 'β₀', 'β', 'Domain', 'Nf (collocation points)', 'Epochs (Adam)', 'Learning Rate'],
        'Value': ['Pohlhausen Boundary Layer', "f''' + β₀*f*f'' + β*(1 - f'²) = 0", beta0, beta, f'[0, {L}]', Nf, epochs_adam, learning_rate]
    })
    metadata.to_excel(writer, sheet_name='Metadata', index=False)

print(f"\n✓ Tables saved to: {excel_filename}")
print(f"  - Sheet 1: Metrics Summary")
print(f"  - Sheet 2: f(x) Comparison at {len(sample_points)} points")
print(f"  - Sheet 3: f'(x) Comparison at {len(sample_points)} points")
print(f"  - Sheet 4: Metadata (problem parameters)")

print("\n" + "="*80)
print("Comparison Complete!")
print("="*80)