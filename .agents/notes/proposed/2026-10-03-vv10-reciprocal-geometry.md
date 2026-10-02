# Proposal: reuse feature reciprocals in the actual VV10 radial closure

Status: proposed; independent host/device gates pass, endpoint comparison pending
Date: 2026-10-03

## Problem and actual dispatch

The full24 profile spends about 19 seconds in the force's VV10 pair closure.
The profiler names Vv10Variant=1, and the enum and WB97M-V method manifest both
identify VV10 (rVV10=2). The earlier rVV10-only experiment improved an unrelated
microbenchmark, so its absence of WB97M-V endpoint benefit did not test this idea.
See the rejected-force note for that diagnosis and retained evidence.

## Decision and bounds

Add a separately versioned compiler-owned VV10 geometry policy. It reuses the
existing feature reciprocals 1/g_i and 1/(g_i+g_j) in the radial logarithmic
factor, adding only 1/g_j. There are four FP64 divisions rather than six.
Energy, density and gradient-feature IR roots retain the same logical hashes
and ordered operations. Only the radial root changes rounding. The CUDA wrapper
selects it for VV10 with feature and geometry demand; SCF feature-only, CPU raw
and all rVV10 closures retain the original path.

Admit squared distance in [0, 2^32] and omega/kappa inputs in [2^-32, 2^32].
These bounds keep all denominators, reciprocals and intermediates finite and
normal. The unused row inverse is not an admission precondition for VV10.
Other values, including exceptional scales, execute the original ordered
closure. No threshold, pair traversal/reduction order, precision mode or output
failure check changes. The small native helper is dispatch into generated code;
its ownership shard now explicitly classifies this existing helper as runtime.

## Evidence

All 67 pair-codegen tests pass through verified ccache, including standalone
deterministic generation, unchanged energy/feature root hashes, exact fallback
results at and around domain boundaries, legacy FP64/exception classification,
and independent 90-digit energy finite differences across the admitted exponent
interval. The compiler checker finds no errors in 410 modules.
Node1 Slurm job 5402 passes all six independent complete RKS/UKS WB97M-V and
reconverged displaced-energy tests in 193.73 seconds. Every molecular sample
retains the 1e-8 Eh / 1e-7 Eh/Bohr gate.

The same job is measuring geometry baseline versus candidate full24 endpoints,
with three warm repeats and matched GPU4PySCF density masks. Results are pending;
there is no endpoint benefit or large-system performance claim yet.
The candidate native library SHA256 is
`4d2bababa025188f40d6f31bf24ba55001298cbcfa30da4e53d5e62f03ba1bb2`.
Source archives, candidate patch, build/ccache receipts and results are retained
under ignored `.artifacts/wb97m-vv10-reciprocal/` and the remote task directory.

## Alternatives and revisit conditions

Multiplying three reciprocals to form phi would also change energy and feature
rounding and exceptional behavior. This proposal deliberately keeps those roots.
Unbounded replacement of division is not justified by molecular test success;
the ordered fallback and independent scalar tests remain required.
Promote only after repeatable complete endpoint benefit, geometry rebuild and
larger-size qualification. A kernel-only improvement is insufficient.
