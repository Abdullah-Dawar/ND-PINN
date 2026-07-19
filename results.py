# ============================================================
# RESULTS CODE: Load and Compare Standard PINN vs ND-PINN
# Run this AFTER main.py is complete
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import pickle

# Create results folder
results_folder = "burgers_results"
os.makedirs(results_folder, exist_ok=True)

# ============================================================
# 1. LOAD SAVED DATA
# ============================================================
print("Loading saved data...")

# Load grid
grid = np.load("grid_data.npz")
x = grid["x_phys"]
t = grid["t_phys"]
X = grid["X_phys"]
T = grid["T_phys"]
nu = float(grid["nu"])

# Load predictions
pred = np.load("predictions.npz")
u_pinn = pred["u_standard_pinn"]
u_ndpinn = pred["u_ndpinn"]

# Load loss histories
with open('loss_history_pinn.pkl', 'rb') as f:
    losshistory_pinn = pickle.load(f)
with open('loss_history_nd.pkl', 'rb') as f:
    losshistory_nd = pickle.load(f)

print("✓ Data loaded successfully")
print(f"  Grid size: {len(x)} x {len(t)}")
print(f"  nu = {nu:.6f}")
print(f"  Standard PINN range: [{np.min(u_pinn):.4f}, {np.max(u_pinn):.4f}]")
print(f"  ND-PINN range: [{np.min(u_ndpinn):.4f}, {np.max(u_ndpinn):.4f}]")

# ============================================================
# 2. ERROR ANALYSIS
# ============================================================
error = np.abs(u_pinn - u_ndpinn)
L1_error = np.mean(error)
L2_error = np.linalg.norm(u_pinn - u_ndpinn) / (np.linalg.norm(u_ndpinn) + 1e-10)
max_error = np.max(error)
rmse_error = np.sqrt(np.mean(error**2))

print("\n" + "="*60)
print("ERROR STATISTICS (Standard PINN vs ND-PINN)")
print("="*60)
print(f"L1 Error: {L1_error:.3e}")
print(f"L2 Relative Error: {L2_error:.3e}")
print(f"Max Absolute Error: {max_error:.3e}")
print(f"RMSE: {rmse_error:.3e}")
print("="*60)

# ============================================================
# 3. SAVE TO EXCEL
# ============================================================
print("\nSaving results to Excel...")

with pd.ExcelWriter(f'{results_folder}/results.xlsx', engine='openpyxl') as writer:
    # Sheet 1: Error Summary
    pd.DataFrame({
        'Metric': ['L1 Error', 'L2 Error', 'Max Error', 'RMSE', 'Viscosity (ν)'],
        'Value': [L1_error, L2_error, max_error, rmse_error, nu],
        'Scientific': [f'{L1_error:.3e}', f'{L2_error:.3e}', f'{max_error:.3e}', f'{rmse_error:.3e}', f'{nu:.6f}']
    }).to_excel(writer, sheet_name='Error_Summary', index=False)
    
    # Sheet 2: Comparison at selected times
    times = [0.25, 0.5, 0.75]
    data = []
    for time in times:
        idx = np.argmin(np.abs(t - time))
        for j in range(0, len(x), 20):
            data.append([x[j], time, u_pinn[idx, j], u_ndpinn[idx, j], error[idx, j]])
    
    pd.DataFrame(data, columns=['x', 't', 'Standard_PINN', 'ND_PINN', 'Error']).to_excel(
        writer, sheet_name='Comparison', index=False)
    
    # Sheet 3: Training Losses
    min_len = min(len(losshistory_pinn.loss_train), len(losshistory_nd.loss_train))
    pd.DataFrame({
        'Iteration': range(min_len),
        'Standard_PINN_Loss': losshistory_pinn.loss_train[:min_len],
        'ND_PINN_Loss': losshistory_nd.loss_train[:min_len]
    }).to_excel(writer, sheet_name='Training_Losses', index=False)

print(f"✓ Excel saved: {results_folder}/results.xlsx")

# ============================================================
# 4. PLOT 1: Training Losses
# ============================================================
plt.figure(figsize=(10, 6))
plt.semilogy(losshistory_pinn.loss_train, 'b-', lw=2, label='Standard PINN')
plt.semilogy(losshistory_nd.loss_train, 'r-', lw=2, label='ND-PINN')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{results_folder}/training_losses.jpg', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {results_folder}/training_losses.jpg")

# ============================================================
# 5. PLOT 2: Solution Comparison at Different Times
# ============================================================
times = [0.25, 0.5, 0.75]
plt.figure(figsize=(15, 4))

for i, ts in enumerate(times):
    idx = np.argmin(np.abs(t - ts))
    
    plt.subplot(1, 3, i+1)
    plt.plot(x, u_pinn[idx], 'b-', lw=2, label='Standard PINN')
    plt.plot(x, u_ndpinn[idx], 'r--', lw=2, label='ND-PINN', alpha=0.7)
    plt.title(f't = {ts}', fontsize=12)
    plt.xlabel('x', fontsize=11)
    plt.ylabel('u', fontsize=11)
    plt.grid(alpha=0.3)
    plt.ylim(-1.2, 1.2)
    if i == 0:
        plt.legend(fontsize=10)

plt.suptitle('Burgers Equation: Standard PINN vs ND-PINN', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_folder}/solution_comparison.jpg', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {results_folder}/solution_comparison.jpg")

# ============================================================
# 6. PLOT 3: Error Contour
# ============================================================
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, T, error, 100, cmap='magma')
plt.colorbar(contour, label='Absolute Error')
plt.xlabel('x', fontsize=12)
plt.ylabel('t', fontsize=12)
plt.title(f'Error Field |PINN - ND-PINN| (L2: {L2_error:.2e})', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_folder}/error_contour.jpg', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {results_folder}/error_contour.jpg")

# ============================================================
# 7. PLOT 4: 3D Comparison (Improved Colors & Shading)
# ============================================================
fig = plt.figure(figsize=(12, 4))

# ---------- Standard PINN ----------
ax1 = fig.add_subplot(131, projection='3d')
surf1 = ax1.plot_surface(
    X, T, u_pinn,
    cmap='viridis',
    edgecolor='none',
    antialiased=True
)
ax1.view_init(elev=30, azim=-75)   # ← rotation added
ax1.set_title('Standard PINN', fontname='Times New Roman', fontsize=14)
ax1.set_xlabel('$x$', fontsize=13)
ax1.set_ylabel('$t$', fontsize=13)
ax1.set_zlabel('$u(x,t)$', fontsize=13)

# ---------- ND-PINN ----------
ax2 = fig.add_subplot(132, projection='3d')
surf2 = ax2.plot_surface(
    X, T, u_ndpinn,
    cmap='plasma',
    edgecolor='none',
    antialiased=True
)
ax2.view_init(elev=30, azim=-75)   # ← rotation added
ax2.set_title('ND-PINN', fontname='Times New Roman', fontsize=14)
ax2.set_xlabel('$x$', fontsize=13)
ax2.set_ylabel('$t$', fontsize=13)
ax2.set_zlabel('$u(x,t)$', fontsize=13)

# ---------- Error Surface ----------
ax3 = fig.add_subplot(133, projection='3d')
surf3 = ax3.plot_surface(
    X, T, error,
    cmap='inferno',
    edgecolor='none',
    antialiased=True
)
ax3.view_init(elev=30, azim=-165)   # ← rotation added
exp = int(np.floor(np.log10(max_error)))
mant = max_error / 10**exp
ax3.set_title(f'Error (Max = ${mant:.2f}\\times 10^{{{exp}}}$)', fontname='Times New Roman', fontsize=14)
ax3.set_xlabel(r'$x$', fontsize=13)
ax3.set_ylabel(r'$t$', fontsize=13)
ax3.set_zlabel('Error', fontname='Times New Roman', fontsize=13)

plt.tight_layout()
plt.savefig(f'{results_folder}/3d_comparison.jpg', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {results_folder}/3d_comparison.jpg")

# ============================================================
# 8. PLOT 5: Initial Condition Verification
# ============================================================
plt.figure(figsize=(8, 6))
plt.plot(x, u_pinn[0], 'b-', lw=2, label='Standard PINN at t=0')
plt.plot(x, u_ndpinn[0], 'r--', lw=2, label='ND-PINN at t=0')
plt.plot(x, -np.sin(np.pi * x), 'k:', lw=2, label='Exact IC: -sin(πx)', alpha=0.7)
plt.xlabel('x', fontsize=12)
plt.ylabel('u', fontsize=12)
plt.title('Initial Condition Verification', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{results_folder}/initial_condition.jpg', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: {results_folder}/initial_condition.jpg")

# ============================================================
# 9. SAVE ALL DATA
# ============================================================
np.savez(f'{results_folder}/complete_results.npz',
         x=x, t=t, X=X, T=T,
         u_standard_pinn=u_pinn,
         u_ndpinn=u_ndpinn,
         error=error,
         L1_error=L1_error,
         L2_error=L2_error,
         max_error=max_error,
         rmse=rmse_error,
         nu=nu)

print(f"✓ Raw data saved: {results_folder}/complete_results.npz")

# ============================================================
# PLOT: Space–Time Heatmaps
# ============================================================
plt.figure(figsize=(14, 5))

# Standard PINN
plt.subplot(1, 2, 1)
plt.contourf(X, T, u_pinn, 200, cmap='coolwarm')
plt.colorbar(label=r'$u(x,t)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('Standard PINN: Space–Time Solution')

# ND-PINN
plt.subplot(1, 2, 2)
plt.contourf(X, T, u_ndpinn, 200, cmap='coolwarm')
plt.colorbar(label=r'$u(x,t)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('ND-PINN: Space–Time Solution')

plt.tight_layout()
plt.savefig(f'{results_folder}/space_time_heatmaps.jpg', dpi=300)
plt.close()

rel_error = error / (np.abs(u_ndpinn) + 1e-8)

plt.figure(figsize=(8, 6))
plt.contourf(X, T, rel_error, 200, cmap='viridis')
plt.colorbar(label='Relative Error')
plt.xlabel(r'$x$')
plt.ylabel(r'$t$')
plt.title('Relative Error Field')
plt.tight_layout()
plt.savefig(f'{results_folder}/relative_error.jpg', dpi=300)
plt.close()
# ============================================================
# 10. SUMMARY
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nResults saved in: {results_folder}/")
print(f"\nFiles generated:")
print(f"  📊 Excel: results.xlsx")
print(f"  📈 training_losses.jpg")
print(f"  📈 solution_comparison.jpg")
print(f"  📈 error_contour.jpg")
print(f"  📈 3d_comparison.jpg")
print(f"  📈 initial_condition.jpg")
print(f"  💾 complete_results.npz")

print(f"\n{'='*50}")
print("FINAL ERROR METRICS (Standard PINN vs ND-PINN)")
print(f"{'='*50}")
print(f"L1 Error:     {L1_error:.3e}")
print(f"L2 Error:     {L2_error:.3e}")
print(f"Max Error:    {max_error:.3e}")
print(f"RMSE:         {rmse_error:.3e}")
print(f"{'='*50}")

print("\n" + "="*70)
print("RESULTS COMPLETE!")
print("="*70)