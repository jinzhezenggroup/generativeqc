# Native progressive geometry optimization

Use `generativeqc optimize` for Cartesian **local-minimum** searches with the
Python-free C++ runtime. At every trial and accepted geometry the optimizer
requests the **same native method's complete analytic energy and forces**.
It does not take finite differences of energies or silently change the target
functional, correlated Hamiltonian, basis, precision or spin state.

## Automatic method ladder

```console
# Automatic GFN2-xTB (CPU) -> PBE -> conventional CCSD(T) (selected backend).
generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --basis def2-svp --output water-ccsdt.xyz --json

# Also works for expensive DFT; use a cheaper Gaussian basis for early PBE.
generativeqc optimize water.xyz --method pbe0-rks --backend cuda \
  --basis def2-tzvp --preopt-basis def2-svp --output water-pbe0.xyz

# DF-CCSD(T) keeps its required CUDA, auxiliary basis and FP64 target contract.
generativeqc optimize water.xyz --method 'df-ccsd(t)' --backend cuda \
  --basis def2-svp --density-fitting cuda --auxiliary-basis def2-svp \
  --correlation-memory-bytes 8589934592 --output water-df-ccsdt.xyz
```

Automatic stages are, in order: GFN2-xTB, an optional spin-aware PBE step for
hybrids/expensive DFT or correlated targets, and the exact requested final
method. RHF, UHF, LDA and PBE targets omit the PBE middle stage. The GFN2
preliminary stage uses CPU even when the final target is CUDA. Automatically
chosen *preliminary* stages can be skipped only if native execution reports
**NOT_IMPLEMENTED**. A failed SCF/CC convergence, nonfinite forces, resource
failure, or failed optimization **never** authorizes silently proceeding.
Explicitly requested stages cannot be skipped at all.

Use a hand-picked progression or inspect the stages without executing:

```console
generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --stages 'gfn2-xtb,pbe-rks,ccsd(t)' --basis cc-pvdz --plan-only

generativeqc optimize water.xyz --method 'ccsd(t)' --backend cuda \
  --stages 'pbe-rks,ccsd(t)' --basis cc-pvdz --opt-fmax 0.0003 \
  --opt-max-steps 80 --opt-max-step 0.35 --json
```

The **last stage must match `--method`**. Method aliases are canonicalized
(`ccsd(t)` is `rccsd(t)`; `df-ccsd(t)` is `df-rccsd(t)`).
`--preopt-basis` applies only to preliminary Gaussian methods; by default
they use the target basis. GFN2 always has its intrinsic basis.
The final step always recomputes and independently converges the *target*
method's gradient at the actual accepted final coordinates.

The optimizer uses Cartesian L-BFGS and backtracking Armijo with a maximum
per-atom displacement; curvature history is reset when the energy surface
changes. Input XYZ defaults to Angstrom (`--units bohr` is also supported);
`--opt-fmax` is the maximum **atomic force norm** in Hartree/Bohr (default
`3e-4`) and `--opt-max-step` is in Bohr (default `0.35`). Preliminary
stages use looser tolerances. Running out of steps or failing a line search
is **not** convergence. No CCSD(T) analytic Hessian is needed for this search.

Successful `--json` output includes final energy, maximum force, Bohr
coordinates, and per-stage status, iterations and evaluation counts.
`--output` creates a separate Angstrom XYZ file after final convergence only,
and refuses to overwrite an existing file. XYZ uses atomic numbers, supported
by the native CLI reader. Without an output path, the final coordinates are
available on stdout.

## Limits and accuracy

This is not a transition-state, periodic-cell or global-conformer optimizer.
An optimized stationary point is not automatically a true minimum: characterize
the structure by vibrational/Hessian analysis when necessary. Currently there
is no automatic cross-geometry SCF/CC amplitude warm start or persistent
checkpoint; expensive CCSD(T) optimizations may require many cold native
energy-plus-force endpoints.

Conventional CCSD(T) forces retain their [qualified AO / closed-shell domain]
(../developer/rccsd_t_api.md). DF-CCSD(T) has a separate
[CUDA/FP64 response model](../developer/df_ccsdt_gradient.md).
Unrestricted CCSD(T) energy availability does *not* imply unrestricted
analytic nuclear forces.
