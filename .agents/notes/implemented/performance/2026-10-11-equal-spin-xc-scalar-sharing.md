# Decision: share polarized XC scalar work on a producer-proven equal-spin slice

Status: implemented local opt-in candidate; complete endpoint promotion pending
Date: 2026-10-11

## Problem

PBE0 has a specialized restricted stationary evaluator, whereas native B3LYP,
M06-2X, WB97M-V and r2SCAN SCF point consumers still evaluate distinct alpha/beta
scalar DAGs after the RKS producer has constructed identical per-spin inputs.
Blindly extending PBE0's stationary witness to meta-GGAs would not establish the
additional tau-input contract. A semilocal kernel improvement also cannot be
substituted for the requested complete functional endpoint optimization.

This task starts from fetched master `a9b991f72`, not the worktree's initial
September snapshot. The original objective remains B3LYP, M06-2X, the wB97
family, r2SCAN-3c and related methods, including complete endpoint qualification.

## Decision

Use the compiler's existing scalar Graph and structural substitution **after**
building polarized first derivatives. Bind beta rho, diagonal sigma and tau to
their alpha counterparts, retaining the separate sigma_ab feature and all
original derivative roots. Identical subgraphs can then be evaluated once;
there is no new scientific formula, precision change or algebraic reassociation.

Emit equal-spin siblings of the canonical B3LYP/WB97M-V point programs and of
the component-wise r2SCAN and split-hybrid worker adapters. Preserve their
thresholds, floors, Fermi-hole constraints and energy rescaling. Give the bound
variants separate expression identities while retaining the original identities.

The native RKS point kernel, unlike an arbitrary stationary task, directly
constructs both spins from the same half-density, gradient and tau channels.
Compile a separate equal-spin point specialization and select it only for this
producer, including the batched layout. UKS, responses, stationary force points
and LDA/PBE remain on the incumbent. The explicit selector
`GENERATIVEQC_CUDA_XC_EQUAL_SPIN=1` is opt-in pending complete endpoint evidence;
unset/`0` retains the reference schedule. There is no new resident-storage
precondition or device allocation.

## Rejected alternatives

- Differentiate a restricted energy and copy its derivatives to both spins:
  that changes the independent partial/chain-rule contract.
- Bind sigma_ab to sigma_aa: diagonal work-input floors can break that equality
  even for an originally equal-gradient input. Keep sigma_ab independent.
- Use a method label, a tolerance or PBE0's rho/gradient-only stationary witness
  as proof of a meta-GGA tau binding.
- Promote based on point throughput or claim r2SCAN-3c/WB97M-V complete-method
  speedups without D4/gCP or range-exchange/VV10 work.

## Evidence

`tests/python/test_equal_spin_xc_codegen.py` executes the production host
selector, checks scalar-work removal, compiles the actual emitted point bodies
as strict host C++, and compares every derivative against both the incumbent
and optional independent pinned Libxc. Work-boundary tests include vacuum,
density screens, zero sigma/tau, Fermi-hole clipping and an independent cross
sigma. The initial complete host run has 57 passes and one unallocated-device
skip; the expanded focused equal-spin suite has 12 passes with PySCF 2.14.0.

An exploratory Slurm RTX 5090 point-only probe on node4 evaluates 262,144 points
per launch. Five post-priming interleaved observations per arm show approximately
16%/25%/3%/11% point-kernel reductions for B3LYP/r2SCAN/WB97M-V/M06-2X.
Outputs match the incumbent bit-for-bit for all 32 independent physical inputs;
maximum independent Libxc energy/partial errors are 4.22e-15/2.20e-14.
These are **not complete endpoint claims**, boundary-device qualification or a
default-promotion gate. The initial generated header freeze precedes additional
identity-only declarations. Its raw sources, timings and report are retained
under ignored `.artifacts/equal-spin-xc/initial-point-probe/`; later source-matched
measurements must remain separate. Slurm accounting is disabled on node4, so a
successful local process result is not fabricated into an accounting receipt.

The final focused host run passes 62 tests with one unallocated-device skip.
Compiler structure validation checks 514 modules with zero dependency errors;
focused Ruff and diff whitespace checks pass. A source-matched follow-up Slurm
job 739 on node4 records successful completion inside its allocation, unchanged
device visibility `0`, and zero-error memcheck/initcheck summaries. Its five
post-priming point-only pairs give 15.869%/25.183%/3.363%/10.885% reductions in
the same method order. Every output remains bitwise identical to the incumbent
and within the unchanged independent Libxc point gates. Current sources,
binary hashes, raw observations and sanitizer summaries are under
`.artifacts/equal-spin-xc/current-point-probe/`. These remain ordinary physical
point probes, not molecular or extreme-boundary device qualification.

The complete production XC contraction source also compiles with NVCC 12.9.86,
C++20, sm_120, `--fmad=false`, verified ccache and the authentic native headers.
The standalone probe needs the surrounding grid-policy/semilocal declarations;
earlier C++17 and incomplete-prelude diagnostics are retained, not counted as
successful compilation. The full frozen-source library build starts separately
as CPU-only Slurm job 740 with a finite 45-minute limit. Its completion and full
SCF/force execution must be observed independently before making endpoint claims.

The native point compilation also identifies a second bottleneck: functional
selection remains a runtime argument inside feature-width-specialized kernels.
Both general and equal-spin single-tile GGA/MGGA entries use 255 registers.
GGA general/equal-spin stack frames are 368/400 bytes, with spill stores
672/272 and loads 312/208 bytes; MGGA frames are 392/376, stores 704/328 and
loads 280/224 bytes. These are static ptxas resource reports, not executed
memory traffic. They explain why the isolated method-bound probe cannot stand
in for native endpoint performance. If endpoint profiling confirms this cost,
bind the already-admitted mathematical functional at kernel compilation rather
than changing the formula or claiming that equal-spin CSE alone fixes spills.

The opt-in native lowering now implements that additional binding. Its separate
[functional-binding note](2026-10-11-native-xc-functional-binding.md) preserves
the lowering decision and source-matched evidence; the unbound compilation
resources above remain the historical phase-1 diagnostic, not the current
candidate's resource report.

An ignored exploratory complete E+F driver is staged as
`.artifacts/equal-spin-xc/endpoint_probe.py` and on node4. It preserves all cold,
warm and moved observations, actual SCF/Fock diagnostics, forces, correction
records and resource diagnostics. Its independent molecular gates are explicitly
`not-run`; passing baseline/candidate consistency or timing is not promotion.
It uses a small explicit pilot grid for ordinary functionals and retains the
r2SCAN-3c method defaults, so those pilot rows cannot qualify production grids.

Source reconstruction, generated headers, native compilation diagnostics and
subsequent probes are staged under ignored `.artifacts/equal-spin-xc/` and
`/data/jzzeng/gqc-equal-spin-20261011/` on node4. All GPU work uses `srun`; the
compilation uses verified ccache 4.5.1 and the existing `/data/jzzeng/ccache`.

## Consequences and remaining work

The generated code and AOT kernel inventory grow; cold setup, compile duration,
register resources and complete endpoint timings need measurement. No broad
functional performance goal is complete yet. In particular, other wB97 variants
have not been admitted or benchmarked by this change, and r2SCAN-3c needs its
complete prescribed composition. Independent molecular energy/force gates,
large-system cold/warm/moved cohorts, sanitizer coverage and promotion decisions
remain required. Revisit the opt-in policy only after those gates pass.
