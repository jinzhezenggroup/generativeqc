# Decision: explicit MINAO ensemble admission

Status: implemented; supersedes the admission/cap discussion in
[the raw MINAO decision](2026-10-07-minao-initial-guess.md)
Date: 2026-10-07

## Problem

A projected superposition of overlapping atomic orbitals is positive but need
not have occupations at most two. Trace-normalized water/STO-3G has a maximum
occupation of 2.8138572253, so the strict shared seed gate correctly rejects it.
The previous numeric cap also omitted the shared validator's 12 simultaneous
square matrices, including its Jacobi eigensolver copies.

## Decision

Keep raw pinned occupied-ANO data and projection unchanged. Only MINAO's
admission stage trace-normalizes D, diagonalizes S^(1/2) D S^(1/2), and solves
min ||f-lambda||^2 subject to 0<=f<=2 and sum(f)=N. Its KKT solution is
f_i=clamp(lambda_i+shift,0,2), found with 128 bounded bisection steps and a
bounded rounding-residual correction. Reconstruct with S^(-1/2) and run the
unchanged strict global validator. This is an ensemble seed, not a claim of
raw PySCF initialization equality or a determinant. Filling previously empty
metric directions is intentional when required by the constrained minimum.

Insufficient AO electron capacity and invalid/singular metrics fail closed.
No preliminary Fock/J/K/XC work, global tolerance change or target-model change
is introduced. Explicit/imported/warm seeds never enter this construction.

The numeric cap and public planner mirror the same conservative inventory:
16*n*n doubles; 2*n*ns projection doubles; 8*n+ns linear slots; 512*(n+ns)
bytes for AO expansions/source shells; 256*atoms bytes for source/center
storage; 16*source_primitives bytes; and 8192 bytes for bounded through-g
single-shell and overlap scratch. Source occupations/shells are reserved
before population. The square allowance covers caller X/raw/output plus
validator 12 matrices; construction itself has a smaller peak. This sums
phase bounds and intentionally overestimates overlap. Target owner and
allocator/runtime overhead are outside the narrow cap. This repair is not
proof of a previous whole-plan OOM.

## Rejected alternatives

Relaxing the shared occupation gate would weaken imported/explicit density
safety. Clipping alone loses electrons; rescaling after clipping can again
exceed two. Replacing raw MINAO with a Fock-built guess changes its definition
and adds the workload this provider avoids. Raw projection remains available
separately for reference comparisons.

## Evidence and limits

Independent NumPy active-set and Gaussian-overlap fixtures cover water in two
bases, methane, OH-/F-/Cl-, Na+ and H2. Native final target energy/density,
exact-cap/fallback, batch/checkpoint and source-controlled CUDA host tests
qualify the stated boundaries. No real GPU run or timing claim is made.
