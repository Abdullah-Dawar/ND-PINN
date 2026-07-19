# ============================================================
# MAIN CODE: Train Standard PINN and ND-PINN
# Burgers Equation: u_t + u u_x = ν u_xx
# x ∈ [-1,1], t ∈ [0,1], ν = 0.01/π
# ============================================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import deepxde as dde
import numpy as np
import pickle

dde.config.set_default_float("float64")

# ============================================================
# 1. Parameters
# ============================================================
nu = 0.01 / np.pi
x_min, x_max = -1.0, 1.0
t_min, t_max = 0.0, 1.0
x_range = x_max - x_min  # = 2
t_range = t_max - t_min  # = 1
iterations = 50000

# ============================================================
# 2. STANDARD PINN (Physical Domain)
# ============================================================
print("\n" + "="*60)
print("Training Standard PINN (Physical Domain)...")
print("="*60)

def pde_pinn(x, y):
    """PDE in physical coordinates: u_t + u*u_x - ν*u_xx = 0"""
    u = y
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_t = dde.grad.jacobian(y, x, i=0, j=1)
    u_xx = dde.grad.hessian(y, x, i=0, j=0)
    return u_t + u * u_x - nu * u_xx

# Geometry
geom_pinn = dde.geometry.Interval(x_min, x_max)
timedomain_pinn = dde.geometry.TimeDomain(t_min, t_max)
geomtime_pinn = dde.geometry.GeometryXTime(geom_pinn, timedomain_pinn)

# BC and IC
def bc_pinn(x, on_boundary):
    return on_boundary

def ic_pinn(x):
    return -np.sin(np.pi * x[:, 0:1])

bc_pinn_obj = dde.icbc.DirichletBC(geomtime_pinn, lambda x: 0, bc_pinn)
ic_pinn_obj = dde.icbc.IC(geomtime_pinn, ic_pinn, lambda _, on_initial: on_initial)

# Data
data_pinn = dde.data.TimePDE(
    geomtime_pinn, pde_pinn, [bc_pinn_obj, ic_pinn_obj],
    num_domain=20000, num_boundary=1000, num_initial=1000
)

# Model
net_pinn = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot normal")
model_pinn = dde.Model(data_pinn, net_pinn)
model_pinn.compile("adam", lr=1e-3)

# Train
losshistory_pinn, _ = model_pinn.train(iterations=iterations, display_every=100)
print("✓ Standard PINN training completed")

# ============================================================
# 3. ND-PINN (Normalized Domain)
# ============================================================
print("\n" + "="*60)
print("Training ND-PINN (Normalized Domain)...")
print("="*60)

def pde_ndpinn(x_norm, y):
    """PDE in normalized coordinates: u_τ + (1/2)*u*u_ξ - (ν/4)*u_ξξ = 0"""
    u = y
    u_xi = dde.grad.jacobian(y, x_norm, i=0, j=0)
    u_tau = dde.grad.jacobian(y, x_norm, i=0, j=1)
    u_xixi = dde.grad.hessian(y, x_norm, i=0, j=0)
    
    x_scale = x_range  # = 2
    x_scale_sq = x_range**2  # = 4
    
    return u_tau + (1.0 / x_scale) * u * u_xi - (nu / x_scale_sq) * u_xixi

# Geometry
geom_nd = dde.geometry.Interval(0, 1)
timedomain_nd = dde.geometry.TimeDomain(0, 1)
geomtime_nd = dde.geometry.GeometryXTime(geom_nd, timedomain_nd)

# BC and IC
def bc_nd(x_norm, on_boundary):
    return on_boundary

def ic_nd(x_norm):
    xi = x_norm[:, 0:1]
    x_physical = x_min + xi * x_range
    return -np.sin(np.pi * x_physical)

bc_nd_obj = dde.icbc.DirichletBC(geomtime_nd, lambda x: 0, bc_nd)
ic_nd_obj = dde.icbc.IC(geomtime_nd, ic_nd, lambda _, on_initial: on_initial)

# Data
data_nd = dde.data.TimePDE(
    geomtime_nd, pde_ndpinn, [bc_nd_obj, ic_nd_obj],
    num_domain=20000, num_boundary=1000, num_initial=1000
)

# Model
net_nd = dde.nn.FNN([2] + [50] * 4 + [1], "tanh", "Glorot normal")
model_nd = dde.Model(data_nd, net_nd)
model_nd.compile("adam", lr=1e-3)

# Train
losshistory_nd, _ = model_nd.train(iterations=iterations, display_every=100)
print("✓ ND-PINN training completed")

# ============================================================
# 4. Save Models and Data
# ============================================================
print("\nSaving models and data...")

# Save models
model_pinn.save("burgers_pinn_model")
model_nd.save("burgers_ndpinn_model")
print("✓ Models saved")

# Save loss histories
with open('loss_history_pinn.pkl', 'wb') as f:
    pickle.dump(losshistory_pinn, f)
with open('loss_history_nd.pkl', 'wb') as f:
    pickle.dump(losshistory_nd, f)
print("✓ Loss histories saved")

# Generate grid points
x_phys = np.linspace(x_min, x_max, 256)
t_phys = np.linspace(t_min, t_max, 100)
X_phys, T_phys = np.meshgrid(x_phys, t_phys)

# Save grid
np.savez("grid_data.npz", 
         x_phys=x_phys, t_phys=t_phys, 
         X_phys=X_phys, T_phys=T_phys,
         x_min=x_min, x_max=x_max, 
         t_min=t_min, t_max=t_max,
         x_range=x_range, t_range=t_range,
         nu=nu)
print("✓ Grid data saved")

# Generate and save predictions
print("\nGenerating predictions...")

# Standard PINN predictions
X_pinn = np.hstack((X_phys.flatten()[:, None], T_phys.flatten()[:, None]))
u_pinn = model_pinn.predict(X_pinn).reshape(len(t_phys), len(x_phys))

# ND-PINN predictions
xi_norm = (X_phys - x_min) / x_range
tau_norm = T_phys
X_nd = np.hstack((xi_norm.flatten()[:, None], tau_norm.flatten()[:, None]))
u_ndpinn = model_nd.predict(X_nd).reshape(len(t_phys), len(x_phys))

# Save predictions
np.savez("predictions.npz", 
         u_standard_pinn=u_pinn, 
         u_ndpinn=u_ndpinn)
print("✓ Predictions saved")

print("\n" + "="*60)
print("MAIN CODE COMPLETE - Run results.py for visualization")
print("="*60)
