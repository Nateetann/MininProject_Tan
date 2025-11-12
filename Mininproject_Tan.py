import numpy as np
import math

# ============================================
# DATA SETUP
# ============================================

# Nodes as NumPy array for vectorization
# Each row represents [x, y] coordinates in meters
nodes = np.array([
    [0.0, 0.0],   # Node 0: left support
    [3.0, 0.0],   # Node 1: right support
    [1.5, 2.0],   # Node 2: top left joint
    [4.5, 2.0],   # Node 3: top right joint
], dtype=float)
N = len(nodes)  # Total number of nodes
ndof = 2 * N    # Total degrees of freedom (2 per node: ux, uy)

# Members: [start_node, end_node, E (Pa), A (m^2)]
# Stores connectivity and material properties for each truss member
members_data = np.array([
    [0, 2, 70e9, 4e-4],  # Member 0: node 0 to node 2
    [1, 2, 70e9, 4e-4],  # Member 1: node 1 to node 2
    [1, 3, 70e9, 4e-4],  # Member 2: node 1 to node 3
    [2, 3, 70e9, 4e-4],  # Member 3: node 2 to node 3
    [0, 1, 70e9, 4e-4],  # Member 4: node 0 to node 1
], dtype=float)
M = len(members_data)  # Total number of members

# DOF order: [ux0, uy0, ux1, uy1, ux2, uy2, ux3, uy3]
# Fixed DOFs represent boundary conditions (supports)
fixed_dofs = np.array([0, 1, 3])  # ux_0, uy_0, uy_1 fixed (pinned at node 0, roller at node 1)

# External loads: initialize force vector with zeros
# Positive x is right, positive y is up
f = np.zeros(ndof)
f[4] = 0.0      # No horizontal force at node 2
f[5] = -12000.0 # Downward force at node 2 (12 kN)
f[6] = 0.0      # No horizontal force at node 3
f[7] = -8000.0  # Downward force at node 3 (8 kN)

# Safety parameters for structural design
sigma_allow = 150e6  # Allowable stress in Pa (150 MPa)
FOS = 1.5            # Factor of safety

# Print buckling assumptions for documentation
print("Buckling model: solid circular equivalent, I = π r^4 / 4 with r = √(A/π); KL = 1.0 (pinned pinned).")
print(f"Stress FOS = {FOS}, Buckling FOS = {FOS}")

# ============================================
# GEOMETRY AND DOF HELPERS
# ============================================

# VECTORIZED: Compute geometry for all members simultaneously
# Uses NumPy advanced indexing to avoid Python loops over members
# Args: members_data (M, 4) array of [start, end, E, A], nodes (N, 2) array of [x, y]
# Returns: L, c, s arrays (M,) for length, cos(θ), sin(θ), plus start/end indices
def compute_all_geometry(members_data, nodes):
    # VECTORIZED: Extract all start and end node indices at once
    start_idx = members_data[:, 0].astype(int)  # Shape (M,)
    end_idx   = members_data[:, 1].astype(int)  # Shape (M,)
    
    # VECTORIZED: Use advanced indexing to get coordinates for all members
    start_xy  = nodes[start_idx]  # (M, 2) - coordinates of all start nodes
    end_xy    = nodes[end_idx]    # (M, 2) - coordinates of all end nodes
    
    # VECTORIZED: Compute dx, dy for all members at once
    dx = end_xy[:, 0] - start_xy[:, 0]  # x-components of member vectors
    dy = end_xy[:, 1] - start_xy[:, 1]  # y-components of member vectors
    
    # VECTORIZED: Compute all member lengths simultaneously
    L  = np.sqrt(dx*dx + dy*dy)  # Euclidean distance for all members
    
    # Validation: ensure no zero-length members exist
    if np.any(L == 0.0):
        raise ValueError("Zero length member detected.")
    
    # VECTORIZED: Compute direction cosines for all members
    c = dx / L  # cos(θ) for all members
    s = dy / L  # sin(θ) for all members
    
    return L, c, s, start_idx, end_idx

# VECTORIZED: Build DOF mapping table for all members
# For each member, maps local DOFs [ux_a, uy_a, ux_b, uy_b] to global DOF indices
# Args: start_idx, end_idx (M,) arrays of node indices
# Returns: dofs (M, 4) array where each row contains global DOF indices
def build_member_dofs(start_idx, end_idx):
    # VECTORIZED: Build DOF table using stack operation
    # Each node i has DOFs [2*i, 2*i+1] for [ux, uy]
    dofs = np.stack([2*start_idx, 2*start_idx + 1, 2*end_idx, 2*end_idx + 1], axis=1)
    return dofs

# ============================================
# ELEMENT AND GLOBAL STIFFNESS
# ============================================

# VECTORIZED: Construct 4x4 element stiffness matrices for ALL members at once
# Uses NumPy broadcasting to avoid loops over members
# For a 2D truss element, k = (EA/L) * [[ c²,  cs, -c², -cs],
#                                        [ cs,  s², -cs, -s²],
#                                        [-c², -cs,  c²,  cs],
#                                        [-cs, -s²,  cs,  s²]]
# Args: E, A, L, c, s are all (M,) arrays
# Returns: k_blocks (M, 4, 4) array of element stiffness matrices
def build_k_blocks(E, A, L, c, s):
    # VECTORIZED: Compute EA/L for all members at once
    EA_L = (E * A) / L  # Shape (M,)
    
    # VECTORIZED: Compute all matrix components using broadcasting
    cc = c * c  # c² for all members
    ss = s * s  # s² for all members
    cs = c * s  # c*s for all members

    # Build individual matrix entries (each is shape (M,))
    k11 = cc;  k12 = cs;  k13 = -cc; k14 = -cs
    k21 = cs;  k22 = ss;  k23 = -cs; k24 = -ss
    k31 = -cc; k32 = -cs; k33 = cc;  k34 = cs
    k41 = -cs; k42 = -ss; k43 = cs;  k44 = ss

    # VECTORIZED: Stack all entries into (M, 4, 4) array using nested stacks
    # This creates a 4x4 matrix for each of the M members simultaneously
    k_blocks = np.stack([
        np.stack([k11, k12, k13, k14], axis=1),  # Row 1 for all members
        np.stack([k21, k22, k23, k24], axis=1),  # Row 2 for all members
        np.stack([k31, k32, k33, k34], axis=1),  # Row 3 for all members
        np.stack([k41, k42, k43, k44], axis=1),  # Row 4 for all members
    ], axis=1)  # Final shape: (M, 4, 4)

    # VECTORIZED: Scale each member's 4x4 block by its EA/L value using broadcasting
    k_blocks = k_blocks * EA_L[:, None, None]  # Broadcasting over last two dimensions
    return k_blocks

# VECTORIZED: Assemble global stiffness matrix using scatter-add operations
# Uses np.add.at for efficient accumulation into the global K matrix
# Args: ndof (total DOFs), dofs (M,4) DOF indices, k_blocks (M,4,4) element stiffnesses
# Returns: K (ndof, ndof) global stiffness matrix
def assemble_global_K(ndof, dofs, k_blocks):
    # Initialize global stiffness matrix with zeros
    K = np.zeros((ndof, ndof))
    
    # VECTORIZED ASSEMBLY: Loop only over local row index i = 0..3
    # This is much more efficient than looping over all M members and all 16 entries
    for i in range(4):
        rows = dofs[:, i]           # (M,) - global row indices for local row i
        cols = dofs                 # (M, 4) - all global column indices for each member
        vals = k_blocks[:, i, :]    # (M, 4) - values from row i of each member's k-matrix
        
        # VECTORIZED: np.add.at performs scatter-add operation
        # Accumulates vals into K at positions (rows, cols) for all members at once
        np.add.at(K, (rows[:, None], cols), vals)
    return K

# Check if a matrix is symmetric within numerical tolerance
# Args: A (n,n) array, tol (float) absolute tolerance
# Returns: boolean indicating symmetry
def is_symmetric(A, tol=1e-9):
    return np.allclose(A, A.T, atol=tol)

# ============================================
# SOLVE AND MEMBER RESULTS
# ============================================

# Apply boundary conditions and solve the linear system Ku = f
# Args: K (ndof, ndof) stiffness, f (ndof,) loads, fixed_dofs array, ndof scalar
# Returns: u (ndof,) displacement vector with fixed DOFs = 0
def apply_bc_and_solve(K, f, fixed_dofs, ndof):
    # VECTORIZED: Identify free (unconstrained) DOFs using set difference
    all_dofs = np.arange(ndof)
    free = np.setdiff1d(all_dofs, fixed_dofs)
    
    # VECTORIZED: Extract reduced system using advanced indexing
    Kff = K[np.ix_(free, free)]  # Stiffness submatrix for free DOFs
    ff  = f[free]                # Load vector for free DOFs
    
    # Solve reduced system: Kff * uf = ff
    uf  = np.linalg.solve(Kff, ff)
    
    # VECTORIZED: Reconstruct full displacement vector
    u   = np.zeros(ndof)  # Initialize with zeros (fixed DOFs stay zero)
    u[free] = uf          # Place computed displacements at free DOFs
    return u

# VECTORIZED: Compute axial forces and stresses for all members simultaneously
# Uses element-wise operations and dot products via np.sum
# Args: members_data (M,4), L, c, s (M,), start_idx, end_idx (M,), u (ndof,)
# Returns: N (M,) axial forces, sigma (M,) stresses
def forces_and_stresses(members_data, L, c, s, start_idx, end_idx, u):
    # Extract material properties for all members
    E = members_data[:, 2]  # Young's modulus (M,)
    A = members_data[:, 3]  # Cross-sectional area (M,)
    
    # VECTORIZED: Collect member end displacements for all members at once
    # Build (M, 4) array where each row is [ux_a, uy_a, ux_b, uy_b]
    u_members = np.column_stack([
        u[2*start_idx],      # ux at start node for all members
        u[2*start_idx + 1],  # uy at start node for all members
        u[2*end_idx],        # ux at end node for all members
        u[2*end_idx + 1]     # uy at end node for all members
    ])  # Shape: (M, 4)
    
    # VECTORIZED: Build transformation matrix for all members
    # Each row is [-c, -s, c, s] for computing axial displacement
    T = np.column_stack([-c, -s, c, s])  # Shape: (M, 4)
    
    # VECTORIZED: Compute EA/L for all members
    EA_L = (E * A) / L  # Shape: (M,)
    
    # VECTORIZED: Compute axial forces using element-wise multiplication and sum
    # N = (EA/L) * (T · u_members) where · is dot product for each row
    N = EA_L * np.sum(T * u_members, axis=1)  # Element-wise multiply then sum across columns
    
    # VECTORIZED: Compute stresses for all members
    sigma = N / A  # Stress = Force / Area for all members
    
    return N, sigma

# ============================================
# BASELINE ANALYSIS
# ============================================

print("Computing geometry for all members...")
# VECTORIZED: Compute all member geometry at once
L, c, s, start_idx, end_idx = compute_all_geometry(members_data, nodes)

# Build DOF mapping table
dofs = build_member_dofs(start_idx, end_idx)

# VECTORIZED: Build element stiffness matrices for all members
k_blocks = build_k_blocks(members_data[:, 2], members_data[:, 3], L, c, s)

print("Assembling global stiffness...")
# VECTORIZED: Assemble global K using scatter-add
K = assemble_global_K(ndof, dofs, k_blocks)

# Verify symmetry of global stiffness matrix
print(f"K symmetric: {is_symmetric(K)}")

print("Solving baseline...")
# Solve for displacements
u = apply_bc_and_solve(K, f, fixed_dofs, ndof)

# Display nodal displacements
print("\nDisplacements (m):")
for i in range(N):
    print(f"Node {i}: ux={u[2*i]:.10e}, uy={u[2*i+1]:.10e}")

# VECTORIZED: Compute forces and stresses for all members
member_forces, member_stresses = forces_and_stresses(members_data, L, c, s, start_idx, end_idx, u)

# Display member forces and stresses
print("\nMember forces and stresses:")
for m in range(M):
    # Determine tension/compression/zero
    kind = "Tension" if member_forces[m] > 0 else ("Compression" if member_forces[m] < 0 else "Zero")
    print(f"Member {m}: N={member_forces[m]:.2f} N ({kind}), sigma={member_stresses[m]/1e6:.2f} MPa")

# ============================================
# SAFETY CHECKS
# ============================================

# VECTORIZED: Check stress failure for all members at once
# Args: stresses (M,), sigma_allow, FOS
# Returns: boolean array (M,) indicating which members fail stress check
def stress_failures(stresses, sigma_allow, FOS):
    limit = sigma_allow / FOS  # Design stress limit
    return np.abs(stresses) > limit  # VECTORIZED: Check all members simultaneously

# VECTORIZED: Check buckling failure for all compression members
# Args: forces (M,), members_data (M,4), L (M,), FOS
# Returns: boolean array (M,) indicating buckling failures, Pcr (M,) critical loads
def buckling_failures(forces, members_data, L, FOS):
    # Extract properties
    A = members_data[:, 3]  # Cross-sectional areas
    E = members_data[:, 2]  # Young's modulus
    
    # VECTORIZED: Compute section properties for all members
    # Assume circular cross-section: A = πr², I = πr⁴/4
    r = np.sqrt(A / np.pi)      # Equivalent radius for all members
    I = np.pi * (r**4) / 4.0    # Second moment of area for all members
    
    # VECTORIZED: Compute Euler critical buckling load for all members
    # Pcr = π²EI / (KL)² with KL = 1.0 for pinned-pinned
    Pcr = (np.pi**2) * E * I / (L*L)  # Critical load for all members
    
    # VECTORIZED: Check buckling only for compression members
    comp = forces < 0.0                    # Boolean array: True for compression
    exceed = np.abs(forces) > (Pcr / FOS)  # Boolean array: True if exceeds buckling capacity
    
    # Return members that are both in compression AND exceed buckling capacity
    return comp & exceed, Pcr

print("\nSafety check:")
limit = sigma_allow / FOS
print(f"Design stress limit = {limit/1e6:.2f} MPa")

# VECTORIZED: Check all failures using vectorized functions
sfail = stress_failures(member_stresses, sigma_allow, FOS)    # Stress failure array
bfail, Pcr = buckling_failures(member_forces, members_data, L, FOS)  # Buckling failure array

# Report failures for each member
any_fail = False
for m in range(M):
    msgs = []
    if sfail[m]:
        msgs.append("stress")
    if bfail[m]:
        msgs.append("buckling")
    if msgs:
        any_fail = True
        print(f"Member {m} FAILS: {', '.join(msgs)}")

if not any_fail:
    print("All members PASS stress and buckling.")

# ============================================
# PARAMETRIC STUDY 1: LOAD SWEEP
# ============================================

print("\n" + "="*60)
print("PARAMETRIC STUDY 1: Load Factor Sweep")
print("="*60)

# Define load scaling factors to test
load_factors = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
first_fail_lam = None      # Track first load factor that causes failure
first_fail_member = None   # Track which member fails first

# Loop through each load factor
for lam in load_factors:
    # VECTORIZED: Scale load vector
    u_lam = apply_bc_and_solve(K, f * lam, fixed_dofs, ndof)
    
    # VECTORIZED: Compute forces and stresses for scaled loads
    N_lam, sig_lam = forces_and_stresses(members_data, L, c, s, start_idx, end_idx, u_lam)
    
    # VECTORIZED: Find maximum stress across all members
    max_abs_sig = float(np.max(np.abs(sig_lam)))
    
    # VECTORIZED: Check for failures using vectorized functions
    stress_fail = stress_failures(sig_lam, sigma_allow, FOS)
    buck_fail, Pcr_lam = buckling_failures(N_lam, members_data, L, FOS)
    any_fail_mask = stress_fail | buck_fail  # Combine both failure modes
    
    # Check if any member fails
    if np.any(any_fail_mask):
        # VECTORIZED: Find first failing member using argmax
        idx = int(np.argmax(any_fail_mask))  # Returns index of first True value
        print(f"λ={lam:.2f}: max |σ|={max_abs_sig/1e6:.2f} MPa, first failure member {idx}")
        
        # Record first failure across all load factors
        if first_fail_lam is None:
            first_fail_lam = lam
            first_fail_member = idx
    else:
        print(f"λ={lam:.2f}: max |σ|={max_abs_sig/1e6:.2f} MPa, no failures")

# ============================================
# PARAMETRIC STUDY 2: UNIFORM AREA SCALING
# ============================================

print("\n" + "="*60)
print("PARAMETRIC STUDY 2: Uniform area scaling")
print("="*60)

# Check if all members pass with uniform area scaling factor alpha
# Args: alpha (scaling factor), members_data, nodes, f, fixed_dofs
# Returns: boolean indicating if all members pass
def passes_with_alpha(alpha, members_data, nodes, f, fixed_dofs):
    # VECTORIZED: Scale all member areas at once
    ms = members_data.copy()
    ms[:, 3] *= alpha  # Multiply all areas by alpha
    
    # Recompute geometry (L, c, s don't change with area, but recompute for consistency)
    Ls, cs, ss, si, ei = compute_all_geometry(ms, nodes)
    
    # Build DOF table
    dofs_s = build_member_dofs(si, ei)
    
    # VECTORIZED: Build element stiffness with scaled areas
    kb = build_k_blocks(ms[:, 2], ms[:, 3], Ls, cs, ss)
    
    # Assemble and solve
    Ks = assemble_global_K(2*len(nodes), dofs_s, kb)
    us = apply_bc_and_solve(Ks, f, fixed_dofs, 2*len(nodes))
    
    # VECTORIZED: Compute forces and stresses
    N_s, sig_s = forces_and_stresses(ms, Ls, cs, ss, si, ei, us)
    
    # VECTORIZED: Check failures
    s_fail = stress_failures(sig_s, sigma_allow, FOS)
    b_fail, _ = buckling_failures(N_s, ms, Ls, FOS)
    
    # Return True if NO failures (all pass)
    return not np.any(s_fail | b_fail)

# Test discrete area scaling factors
for alpha in [0.8, 0.9, 1.0, 1.1, 1.2]:
    print(f"α={alpha:.2f}: {'PASSES' if passes_with_alpha(alpha, members_data, nodes, f, fixed_dofs) else 'FAILS'}")

# Binary search for minimum alpha that satisfies all checks
lo, hi = 0.1, 10.0  # Search bounds
for _ in range(30):  # 30 iterations for convergence
    mid = 0.5 * (lo + hi)
    if passes_with_alpha(mid, members_data, nodes, f, fixed_dofs):
        hi = mid  # Solution exists at or below mid
    else:
        lo = mid  # Need higher alpha
alpha_min = hi
print(f"Minimum α that passes ≈ {alpha_min:.4f}")

# ============================================
# PARAMETRIC STUDY 3: SINGLE MEMBER +10% AREA
# ============================================

print("\n" + "="*60)
print("PARAMETRIC STUDY 3: Single member +10% area upgrade")
print("="*60)

# Helper function: compute maximum absolute stress for given member configuration
# Args: ms (M,4) member data array
# Returns: maximum absolute stress across all members (float)
def max_abs_stress_for(ms):
    # Recompute geometry
    Lx, cx, sx, si, ei = compute_all_geometry(ms, nodes)
    dofs_x = build_member_dofs(si, ei)
    
    # Build stiffness and solve
    kx = build_k_blocks(ms[:, 2], ms[:, 3], Lx, cx, sx)
    Kx = assemble_global_K(ndof, dofs_x, kx)
    ux = apply_bc_and_solve(Kx, f, fixed_dofs, ndof)
    
    # VECTORIZED: Compute stresses
    _, sigx = forces_and_stresses(ms, Lx, cx, sx, si, ei, ux)
    
    # VECTORIZED: Return maximum absolute stress
    return float(np.max(np.abs(sigx)))

# Compute baseline maximum stress
baseline_max = max_abs_stress_for(members_data)
print(f"Baseline max |σ|: {baseline_max/1e6:.2f} MPa")

# Test upgrading each member individually
reductions = np.zeros(M)  # Store stress reduction for each upgrade

for idx in range(M):
    # VECTORIZED: Create modified member data with one area upgraded by 10%
    ms = members_data.copy()
    ms[idx, 3] *= 1.10  # Increase area of member idx by 10%
    
    # Compute maximum stress with upgrade
    up = max_abs_stress_for(ms)
    
    # Calculate reduction in maximum stress
    reductions[idx] = baseline_max - up
    print(f"Upgrade member {idx}: max |σ|={up/1e6:.2f} MPa, reduction={reductions[idx]/1e6:.2f} MPa")

# VECTORIZED: Find best upgrade using argmax
best_idx = int(np.argmax(reductions))
best_reduction = float(reductions[best_idx])

print(f"Best upgrade: member {best_idx}")
print(f"Stress reduction: {best_reduction/1e6:.2f} MPa ({100*best_reduction/baseline_max:.1f}%)")

# ============================================
# ROBUSTNESS TESTS
# ============================================

print("\n" + "="*60)
print("ROBUSTNESS TESTS")
print("="*60)

# Helper function for numerical assertions
# Args: x, y (values to compare), tol (tolerance), msg (error message)
def assert_close(x, y, tol=1e-9, msg=""):
    if abs(x - y) > tol:
        raise AssertionError(msg if msg else f"Not close: {x} vs {y}")

# Test 1: Verify global stiffness matrix is symmetric
print("Test 1: Global K is symmetric...")
if not is_symmetric(K):
    raise AssertionError("K must be symmetric")
print("  ✓ PASSED")

# Test 2: Zero loads should produce zero displacements at free DOFs
print("Test 2: Zero loads produce zero displacements...")
u0 = apply_bc_and_solve(K, np.zeros(ndof), fixed_dofs, ndof)
free = np.setdiff1d(np.arange(ndof), fixed_dofs)
if not np.allclose(u0[free], 0.0, atol=1e-12):
    raise AssertionError("Zero loads should give zero free displacements")
print("  ✓ PASSED")

# Test 3: Single bar sanity check - verify u = FL/(EA) and N = F
print("Test 3: Two node single bar sanity check...")
# Create simple 2-node bar
nodes_t = np.array([[0.0, 0.0], [2.0, 0.0]])
members_t = np.array([[0, 1, 200e9, 1e-4]], dtype=float)

# Compute geometry
L_t, c_t, s_t, si_t, ei_t = compute_all_geometry(members_t, nodes_t)
dofs_t = build_member_dofs(si_t, ei_t)

# Build stiffness
k_t = build_k_blocks(members_t[:, 2], members_t[:, 3], L_t, c_t, s_t)
K_t = assemble_global_K(4, dofs_t, k_t)

# Apply load and solve
fixed_t = np.array([0, 1, 3])  # Fix node 0 completely, fix uy at node 1
f_t = np.zeros(4)
Fapp = 10000.0
f_t[2] = Fapp  # Apply force in x-direction at node 1

u_t = apply_bc_and_solve(K_t, f_t, fixed_t, 4)

# Check displacement: u = FL/(EA)
u_expected = Fapp * L_t[0] / (members_t[0, 2] * members_t[0, 3])
assert_close(u_t[2], u_expected, 1e-9, "Bar displacement mismatch")

# Check force: N should equal applied force
N_t, _ = forces_and_stresses(members_t, L_t, c_t, s_t, si_t, ei_t, u_t)
assert_close(N_t[0], Fapp, 1e-6, "Bar force should equal applied load")
print("  ✓ PASSED")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)

# ============================================
# APPENDIX TEST HARNESS
# ============================================

print("\n" + "="*60)
print("REQUIRED TEST CASES (Appendix)")
print("="*60)

# Generic test case runner
# Args: name (string), nodes_tc (list), members_tc (list), fixed_tc (list),
#       loads_tc (dict), expected values for validation
# Returns: computed u, N, sigma arrays
def run_test_case(name, nodes_tc, members_tc, fixed_tc, loads_tc,
                  expected_u=None, expected_N=None, expected_sigma=None):
    print(f"\n{name}")
    print("-" * 40)
    
    # Convert to arrays
    nodes_arr = np.array(nodes_tc, dtype=float)
    members_arr = np.array(members_tc, dtype=float)
    nd = 2 * len(nodes_arr)  # Total DOFs
    
    # Build load vector from dictionary
    f_tc = np.zeros(nd)
    for dof, val in loads_tc.items():
        f_tc[int(dof)] = val

    # Solve test case
    L_tc, c_tc, s_tc, si_tc, ei_tc = compute_all_geometry(members_arr, nodes_arr)
    dofs_tc = build_member_dofs(si_tc, ei_tc)
    k_tc = build_k_blocks(members_arr[:, 2], members_arr[:, 3], L_tc, c_tc, s_tc)
    K_tc = assemble_global_K(nd, dofs_tc, k_tc)
    u_tc = apply_bc_and_solve(K_tc, f_tc, np.array(fixed_tc, dtype=int), nd)
    N_tc, sigma_tc = forces_and_stresses(members_arr, L_tc, c_tc, s_tc, si_tc, ei_tc, u_tc)

    # Print results
    print(f"Displacements: {u_tc}")
    print(f"Axial Forces: {N_tc}")
    print(f"Stresses: {sigma_tc}")
    print(f"max |u|: {np.max(np.abs(u_tc)):.10f}")
    print(f"max |N|: {np.max(np.abs(N_tc)):.2f}")
    
    # Identify compression members
    comp = np.where(N_tc < 0)[0]
    print(f"Members in compression: {list(map(int, comp))}")

    # Validate against expected values if provided
    if expected_u is not None:
        print("Displacements match:",
              np.allclose(u_tc, expected_u, atol=1e-9))
    if expected_N is not None:
        print("Forces match:",
              np.allclose(N_tc, expected_N, atol=1e-6))
    if expected_sigma is not None:
        print("Stresses match:",
              np.allclose(sigma_tc, expected_sigma, atol=1e2))
    return u_tc, N_tc, sigma_tc

# Test Case 1: Single Bar in Tension
tc1_nodes = [[0, 0], [2, 0]]
tc1_members = [[0, 1, 200e9, 1e-4]]
tc1_fixed = [0, 1, 3]
tc1_loads = {2: 10000}
tc1_exp_u = np.array([0, 0, 0.001, 0], dtype=float)
tc1_exp_N = np.array([10000], dtype=float)
tc1_exp_sigma = np.array([1.0e8], dtype=float)
run_test_case("Test Case 1: Single Bar in Tension",
              tc1_nodes, tc1_members, tc1_fixed, tc1_loads,
              tc1_exp_u, tc1_exp_N, tc1_exp_sigma)

# Test Case 2: Triangular Truss Under Vertical Load
tc2_nodes = [[0, 0], [3, 0], [1.5, 2.0]]
tc2_members = [
    [0, 2, 70e9, 4e-4],
    [1, 2, 70e9, 4e-4],
    [0, 1, 70e9, 4e-4]
]
tc2_fixed = [0, 1, 3]
tc2_loads = {5: -10000}
tc2_exp_u = np.array([0, 0, 0.0004017857, 0, 0.0002008929, -0.0008482143], dtype=float)
tc2_exp_N = np.array([-6250, -6250, 3750], dtype=float)
tc2_exp_sigma = np.array([-1.5625e7, -1.5625e7, 9.375e6], dtype=float)
run_test_case("Test Case 2: Triangular Truss",
              tc2_nodes, tc2_members, tc2_fixed, tc2_loads,
              tc2_exp_u, tc2_exp_N, tc2_exp_sigma)

# Test Case 3: 4-Node / 5-Member Pedestrian Truss Slice (our main problem)
tc3_nodes = [[0, 0], [3, 0], [1.5, 2.0], [4.5, 2.0]]
tc3_members = [
    [0, 2, 70e9, 4e-4],
    [1, 2, 70e9, 4e-4],
    [1, 3, 70e9, 4e-4],
    [2, 3, 70e9, 4e-4],
    [0, 1, 70e9, 4e-4]
]
tc3_fixed = [0, 1, 3]
tc3_loads = {5: -12000, 7: -8000}
tc3_exp_u = np.array([0, 0, 0.0001607143, 0, 0.0008244048, -0.0008973214,
                      0.0014672619, -0.0020959821], dtype=float)
tc3_exp_N = np.array([-2500, -12500, -10000, 6000, 1500], dtype=float)
tc3_exp_sigma = np.array([-6.25e6, -3.125e7, -2.5e7, 1.5e7, 3.75e6], dtype=float)
run_test_case("Test Case 3: 4 Node Pedestrian Truss",
              tc3_nodes, tc3_members, tc3_fixed, tc3_loads,
              tc3_exp_u, tc3_exp_N, tc3_exp_sigma)

# Test Case 4: Zero-Load Consistency Check
tc4_nodes = [[0, 0], [2, 0], [1, 1]]
tc4_members = [
    [0, 2, 100e9, 2e-4],
    [1, 2, 100e9, 2e-4],
    [0, 1, 100e9, 2e-4]
]
tc4_fixed = [0, 1, 3]
tc4_loads = {}  # No loads
tc4_exp_u = np.zeros(6, dtype=float)
tc4_exp_N = np.zeros(3, dtype=float)
tc4_exp_sigma = np.zeros(3, dtype=float)
run_test_case("Test Case 4: Zero Load Consistency",
              tc4_nodes, tc4_members, tc4_fixed, tc4_loads,
              tc4_exp_u, tc4_exp_N, tc4_exp_sigma)

# ============================================
# MEMO READY SUMMARY
# ============================================

print("\n----- SUMMARY (paste into memo) -----")

# VECTORIZED: Compute displacement magnitudes for all nodes
uxuy = u.reshape(-1, 2)  # Reshape to (N, 2) for [ux, uy] pairs
u_mags = np.sqrt(np.sum(uxuy**2, axis=1))  # Magnitude = sqrt(ux² + uy²) for each node
node_umax = int(np.argmax(u_mags))  # Find node with maximum displacement
umax = float(u_mags[node_umax])

# Compute buckling properties for margin calculations
A_arr = members_data[:, 3]
E_arr = members_data[:, 2]
r_eq = np.sqrt(A_arr / np.pi)
I_eq = np.pi * r_eq**4 / 4.0
Pcr = (np.pi**2) * E_arr * I_eq / (L**2)

# VECTORIZED: Compute safety margins for all members
stress_margin = (sigma_allow / FOS) - np.abs(member_stresses)  # Positive = safe
buck_margin  = np.full(M, np.inf)  # Initialize with infinity (tension members)
comp = member_forces < 0.0  # Boolean array for compression members
buck_margin[comp] = (Pcr[comp] / FOS) - np.abs(member_forces[comp])

# Report maximum displacement
print(f"Max |u| = {umax:.6e} m at node {node_umax}")

# Report failing members with margins
fails = []
for m in range(M):
    fm = []
    if np.abs(member_stresses[m]) > (sigma_allow / FOS):
        fm.append(f"stress margin = {stress_margin[m]/1e6:.2f} MPa")
    if (member_forces[m] < 0.0) and (np.abs(member_forces[m]) > Pcr[m]/FOS):
        fm.append(f"buckling margin = {buck_margin[m]:.2f} N")
    if fm:
        fails.append(f"Member {m}: " + ", ".join(fm))

if fails:
    print("Failing members:")
    for line in fails:
        print("  " + line)
else:
    print("All members pass baseline stress and buckling.")

# Report load sweep results
if first_fail_lam is not None:
    print(f"Load sweep: first failure λ = {first_fail_lam} at member {first_fail_member}")
else:
    print("Load sweep: no failures in tested range.")

# Report area scaling and upgrade results
print(f"Uniform α: minimum α that passes ≈ {alpha_min:.4f}")
print(f"Best +10% area upgrade: member {best_idx}, Δ max |σ| = {best_reduction/1e6:.2f} MPa "
      f"({100*best_reduction/baseline_max:.1f}%)")
print("----- END SUMMARY -----")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)