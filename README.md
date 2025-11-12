# Truss Health Check Analysis - Mini-Project
## EECE 2140: Computing Fundamentals for Engineers
**Fall 2025**  
**Author:** Nathan Tan  
**Email:** tan.nat@northeastern.edu  
**Instructor:** Dr. Fatema Nafa

---

## Project Overview

This project implements a NumPy-based structural analysis system for a 2D planar truss (pedestrian footbridge slice). The program performs finite element analysis to compute nodal displacements, member axial forces and stresses, and evaluates structural safety against allowable stress and Euler buckling limits.

## System Requirements

- **Python Version:** 3.7 or higher
- **Required Packages:**
  - `numpy` (for numerical computations and linear algebra)
  - `math` (standard library - included with Python)

## Installation

### Install Python (if not already installed)
Download from: https://www.python.org/downloads/

### Install Required Package
```bash
pip install numpy
```

Or if using pip3:
```bash
pip3 install numpy
```

## Running the Code

### Basic Execution
```bash
python truss_analysis.py
```

### Save Output to File
```bash
python truss_analysis.py > results.txt
```

### Using Python 3 Explicitly
```bash
python3 truss_analysis.py
```

## File Structure
```
TanNathan_TrussProject/
│
├── truss_analysis.py          # Main analysis code
├── memo.pdf                    # Engineering memo (2 pages)
└── README.md                   # This file
```

## What the Code Does

The program executes the following analysis workflow:

### 1. Baseline Analysis
- Computes geometry (lengths, direction cosines) for all members using vectorized operations
- Assembles global stiffness matrix (8×8 for 4-node truss)
- Applies boundary conditions at supports
- Solves for nodal displacements
- Computes member axial forces and stresses
- Reports forces as tension/compression

### 2. Safety Checks
- **Stress Check:** Verifies |σ| ≤ σ_allow / FOS for all members
- **Buckling Check:** Evaluates compression members against Euler buckling capacity
- Factor of Safety (FOS) = 1.5 for both checks
- Allowable stress = 150 MPa

### 3. Parametric Study 1: Load Factor Sweep
- Tests load factors: λ = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Identifies first failure and governing limit state
- Reports maximum stress at each load level

### 4. Parametric Study 2: Uniform Area Scaling
- Tests area scaling factors: α = [0.8, 0.9, 1.0, 1.1, 1.2]
- Performs binary search to find minimum α that passes all checks
- Determines required area increase for structural adequacy

### 5. Parametric Study 3: Hot-Spot Upgrade
- Tests +10% area increase for each member individually
- Identifies which single member upgrade most reduces global maximum stress
- Quantifies stress reduction benefit

### 6. Test Case Validation
Validates implementation against 4 required test cases:
- **Test 1:** Single bar in tension (analytical verification)
- **Test 2:** Triangular truss under vertical load
- **Test 3:** 4-node pedestrian truss (baseline problem)
- **Test 4:** Zero-load consistency check

### 7. Robustness Tests
- Symmetry verification of global stiffness matrix
- Zero-load produces zero-displacement validation
- Single-element sanity check (u = FL/EA)

## Problem Configuration

### Baseline Truss Geometry
- **Nodes:** 4 nodes at coordinates (0,0), (3,0), (1.5,2.0), (4.5,2.0) [meters]
- **Members:** 5 truss elements connecting nodes
- **Material:** E = 70 GPa (aluminum), A = 4×10⁻⁴ m² for all members
- **Supports:** Node 0 pinned (ux=0, uy=0), Node 1 roller (uy=0)
- **Loading:** Vertical loads at nodes 2 and 3: -12 kN and -8 kN respectively

### Design Criteria
- **Allowable Stress:** σ_allow = 150 MPa
- **Factor of Safety:** FOS = 1.5 (both stress and buckling)
- **Buckling Model:** Circular cross-section equivalent
  - I = πr⁴/4 where r = √(A/π)
  - Effective length factor KL = 1.0 (pinned-pinned assumption)

## Code Features

### Vectorization
The code extensively uses NumPy vectorized operations to avoid Python loops:
- Geometry computation for all members simultaneously
- Element stiffness matrix construction using broadcasting
- Force/stress computation with element-wise operations
- Safety checks using boolean array operations

### Key Functions
- `compute_all_geometry()` - Vectorized member geometry
- `build_k_blocks()` - Construct all 4×4 element stiffness matrices at once
- `assemble_global_K()` - Scatter-add assembly using np.add.at
- `forces_and_stresses()` - Vectorized post-processing
- `stress_failures()` - Vectorized stress check
- `buckling_failures()` - Vectorized buckling check

## Expected Output

The program prints:
1. Nodal displacements (scientific notation)
2. Member forces and stresses with tension/compression labels
3. Safety check results (pass/fail for each member)
4. Load sweep results with failure identification
5. Minimum area scaling factor
6. Best hot-spot upgrade recommendation
7. Test case validation results
8. Summary section formatted for engineering memo

## Interpreting Results

### Displacement Output
```
Node 0: ux=0.000e+00, uy=0.000e+00  (fixed support)
Node 3: ux=1.467e-03, uy=-2.096e-03 (maximum displacement)
```

### Force Output
```
Member 1: N=-12500.00 N (Compression), sigma=-31.25 MPa
```
- Negative N = Compression
- Positive N = Tension

### Safety Output
```
All members PASS stress and buckling.
```
or
```
Member 1 FAILS: stress, buckling
```

## Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'numpy'
```
**Solution:** Install NumPy using `pip install numpy`

### Singular Matrix Error
```
LinAlgError: Singular matrix
```
**Solution:** Check that boundary conditions properly constrain rigid body motion

### Zero-Length Member Error
```
ValueError: Zero length member detected.
```
**Solution:** Verify node coordinates are distinct

## Technical Notes

- All calculations performed in SI units (meters, Newtons, Pascals)
- Numerical tolerances: 1e-9 for displacements, 1e-6 for forces, 1e2 for stresses
- Matrix symmetry verified to tolerance 1e-9
- Binary search converges in 30 iterations for area scaling

## Performance

- Typical runtime: < 1 second for baseline + all parametric studies
- Memory usage: Minimal (small 8×8 global matrix)
- All operations vectorized for efficiency

## References

- Assignment document: "Mini-Project: Truss Health Check with NumPy"
- Course: EECE 2140, Fall 2025, Northeastern University
- Structural analysis theory: Direct Stiffness Method for 2D trusses

## Contact

For questions about this implementation:
- **Student:** Nathan Tan
- **Email:** tan.nat@northeastern.edu
- **Course:** EECE 2140, Fall 2025

---

**Last Updated:** November 12, 2025  
**Version:** 1.0
