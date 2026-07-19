# ND-PINN  
## Normalized Domain Physics-Informed Neural Network for Accurate and Stable Solution of Differential Equations

This repository contains the implementation of the **Normalized Domain Physics-Informed Neural Network (ND-PINN)** framework for solving differential equations using scientific machine learning.

ND-PINN is a physics-informed neural network approach that introduces a normalized computational domain to improve the stability, robustness, and applicability of PINN-based methods for solving differential equations.

The repository includes implementations of ND-PINN for benchmark and engineering problems, including boundary layer equations and nonlinear partial differential equations.

---

# Repository Contents

The repository contains the following implementations:
ND-PINN/
│
├── ND_PINN_Burgers_Main.py
│ └── Training of Standard PINN and ND-PINN for Burgers equation
│
├── ND_PINN_Burgers_Results.py
│ └── Visualization and comparison of Burgers equation results
│
├── ND_PINN_Pohlhausen.py
│ └── ND-PINN implementation for Pohlhausen boundary layer equation
│
├── ND_PINN_Slip_Flow.py
│ └── ND-PINN implementation for slip-flow boundary layer problem
│
├── README.md
├── LICENSE
└── .gitignore

---

# Implemented Problems

## 1. Pohlhausen Boundary Layer Equation

**File:** `ND_PINN_Pohlhausen.py`

This code implements ND-PINN for the classical Pohlhausen boundary layer equation.

The implementation includes:
- Physics-informed neural network formulation
- Normalized-domain training
- Automatic differentiation
- Boundary condition enforcement
- Comparison with reference solution

---

## 2. Slip-Flow Boundary Layer Problem

**File:** `ND_PINN_Slip_Flow.py`

This code applies ND-PINN to a nonlinear boundary layer flow problem with slip effects.

The implementation includes:
- Nonlinear governing equations
- Multiple physical parameters
- ND-PINN training procedure
- Solution prediction and analysis

---

## 3. Burgers Equation

**Files:** `ND_PINN_Burgers_Main.py`, `ND_PINN_Burgers_Results.py`

This example demonstrates the comparison between:
- Standard PINN trained in the physical domain
- ND-PINN trained in the normalized domain

The workflow includes:
1. Training Standard PINN and ND-PINN models
2. Saving trained models and prediction data
3. Mapping ND-PINN predictions to the physical coordinate system
4. Comparing both approaches through visualization and numerical analysis

The results code generates:
- Solution comparison plots
- Error distributions
- Training loss comparison
- 3D solution visualization
- Exported numerical results

---

# Software Requirements

The implementations are developed using Python.

**Required packages:**
- tensorflow
- deepxde
- numpy
- scipy
- matplotlib
- pandas
- openpyxl

Install dependencies using:

pip install tensorflow deepxde numpy scipy matplotlib pandas openpyxl
How to Run
Pohlhausen Boundary Layer Equation
bash
python ND_PINN_Pohlhausen.py
Slip-Flow Boundary Layer Problem
bash
python ND_PINN_Slip_Flow.py
Burgers Equation
Step 1: Train models

python ND_PINN_Burgers_Main.py
This code trains both Standard PINN and ND-PINN models and saves the required data.

Step 2: Generate results

python ND_PINN_Burgers_Results.py
This code loads the trained results and generates comparison figures and analysis.

Applications
The ND-PINN framework can be extended to various scientific and engineering problems, including:

Boundary layer flows

Fluid mechanics problems

Heat transfer problems

Nonlinear differential equations

Partial differential equations

Scientific machine learning applications

Citation
If you use this repository in your research, please cite:

Dawar, Abdullah. ND-PINN: A Normalized Domain Physics-Informed Neural Network for Accurate and Stable Solution of Differential Equations.
