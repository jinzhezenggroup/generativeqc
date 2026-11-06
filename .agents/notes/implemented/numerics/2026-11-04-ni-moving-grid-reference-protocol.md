# Decision: retain a matched, independently converged Ni moving-grid oracle

Status: implemented (validation protocol only)
Date: 2026-11-04

## Scope

This protocol generates independent references for the explicit-grid spherical
NiH4(2-) singlet sentinel in #2843. It does not implement a production derivative,
qualify a native Ni HVP/Hessian, or promote a public Calculator endpoint. No
external density or derivative enters production execution.

Use the independently pinned `tests/data/global_hybrid/def2-svp-ni-h.json` and
its manifest/license checks, preserving 51 real-spherical AOs (56 Cartesian
workspace functions). Coordinates in Bohr are:

| Atom | x | y | z |
| --- | ---: | ---: | ---: |
| Ni | 0.01 | -0.02 | 0.03 |
| H | 2.90 | 0.10 | 0.03 |
| H | -3.05 | 0.05 | -0.02 |
| H | 0.03 | 2.97 | 0.07 |
| H | -0.04 | -3.10 | -0.03 |

Charge is -2, multiplicity 1. Resolve the actual B3LYP MethodIR (VWN-RPA,
physical 1/5 full-range exchange), not its internal native carrier. The explicit
grid is `(16, 6, 12)`, 5760 points; it is not native default-grid qualification.

## Matched independent Hamiltonian and grid

1. Converge the native primal state at energy 1e-13, density 1e-11, maximum
   240 iterations. Borrow only basis/primal-grid inputs and detached values for
   comparison; close its exported source before releasing native owners.
2. Build PySCF 2.14.0 spherical RKS from the same pinned raw basis. Construct its
   XC code from the MethodIR semilocal components and physical exchange fraction.
   Use exact four-center integrals (`direct_scf=False`), not DF substitution.
3. Recover each element's atom-centered offsets and *unpartitioned* atomic weights
   from the primal grid. Feed these inputs to PySCF's `grids_response_cc`, with
   `radii_adjust=None` and `original_becke`. Independent partition/motion
   arithmetic then supplies the displaced grid and analytic moving-grid gradient
   (`gradient.grid_response=True`). Compare central points at 2e-12 absolute and
   weights at 2e-10 absolute / 2e-12 relative; large far-tail weights make an
   absolute-only comparison misleading.
4. Independently converge every reference geometry at energy 1e-12,
   orbital-gradient control 1e-10, maximum 240 cycles. Fresh atomic guesses work
   for all three directional steps. Raw-matrix guesses may start from the
   *independent central PySCF density*, never the native density. On failed SCF,
   retain the failure and try bounded Newton recovery (80 macro cycles,
   `ah_lindep=1e-24`); then fresh atomic/minao guesses at unchanged controls.
   Accept no result with a false convergence flag. Post-check the actual orbital
   gradient below 2e-9 and a maximum displaced/central reference density change
   below 0.02 to guard the local electronic branch.
5. Before derivatives, require native/reference energy agreement 2e-10 Hartree,
   overlap 2e-12 absolute, and density 2e-9 absolute. The earlier matched-state
   probe independently established energy error 4.55e-12 Hartree, density error
   1.96e-10 and a 0.1433784 Hartree occupied/virtual gap.

The initial minao-only and seeded probes sometimes exhausted SCF despite small
reported orbital gradients. Do not reinterpret those flags as convergence,
silently relax gates, seed with native density, or discard the failed receipts.
Alternative independent guesses address convergence without changing chemistry.

## Frozen derivative gates and results

Use the seed-2843 random `(5,3)` direction scaled by 0.03 and remove its mean
translation. Its norm is 0.13978350893712545. Directional gradient steps are
`3e-3, 1e-3, 3e-4`; raw Cartesian-column steps are `1e-3, 5e-4, 3e-4`.
Freeze 1e-5 Hartree/Bohr² for native maximum raw-matrix error and maximum HVP
error divided by direction norm. Do not symmetrize raw matrices or repair
translation/rotation before acceptance.

The independent three-step directional reference completed in 42.13 s. All 15
raw columns at all three steps also completed, including bounded guess retries;
the resumed raw-column phase took 391.55 s after central/directional setup.
Independent raw asymmetry decreases from 1.72e-7 to 1.61e-8; translation residual
decreases from 1.86e-7 to 1.65e-8. Raw H@v versus the independent directional
reference has normalized maximum differences 3.63e-7, 9.25e-8 and 3.42e-8.
These establish oracle consistency, **not native derivative parity**.

Local recipes/receipts are retained under `/tmp/gqc-2843-2849/`:
`probe-ni-gradient-reference-atom-v4.py`, `probe-ni-gradient-reference.py`,
`probe-ni-raw-reference.py`, `ni-independent-gradient-hvp-v4.log`,
`ni-independent-gradient-hvp.npz`, `ni-independent-raw-hessian-v2.log`, and
`ni-independent-raw-hessian.npz`. Earlier reference versions, logs and the first
partial raw matrix remain separate. A production/default-grid or installed
qualification must not substitute this explicit-grid oracle-only receipt.

## Remaining native boundary

On source 21a3ceaee, the complete native probe's response converges in 17
iterations / 34 actions at 8.11e-12, but the initial bounded scalar-compilation
trial still reaches the 1800-second whole-probe limit during one-electron
second-source assembly. It returns no HVP. All individual compiled tiles remain
under their existing finite compiler deadline; bounded compiler inputs do not
bound aggregate cold compilation work. Preserve the cache and the failure rather
than claiming completion or an endpoint speedup.

An additional native-default-grid probe uses 59,872 points, converges primal SCF
to -1510.3058118719334 Hartree in 91.82 s, then explicitly refuses the low-level
512 MiB grid lease with `AO jet/motion workspace exceeds grid_budget_bytes`.
It does not silently lower the grid. This is state/resource-admission evidence,
not a default-grid HVP/Hessian or installed public endpoint pass.
