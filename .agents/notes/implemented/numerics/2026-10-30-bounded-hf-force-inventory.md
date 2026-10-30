# Decision: bounded HF force coverage follows callable derivative consumers

Status: implemented
Date: 2026-10-30

## Problem

The README HF Direct 48/96-atom cases had correct energies but force errors of
approximately 8.48/9.68 Eh/Bohr on the portable A100 and SM120 builds. Forcing
bounded streaming on the 12-atom case reproduced the failure. Its first atom's
force was `[-2.559206675, -4.321427863, -0.552791831]`, whereas the independent
strict PySCF result is approximately `[-0.00810025, -0.02278001, -0.01324232]`.
A staged diagnostic isolated the missing two-electron force contributions;
the nuclear and one-electron terms were consistent with the converged density.

The bounded dispatcher combined the generated Fock inventory with its force
coverage mask, while both generated force queues iterate the separate force
registry. Fock-only rows were therefore neither executed nor admitted to the
bounded fallback. Queue planning also reserved force slices for these rows,
based on an obsolete assumption that they had callable dormant force symbols.

## Decision and invariants

Only the actual force registry authorizes a generated derivative consumer.
Retain the existing native dddd route as the sole Fock-advertised exception:
its native derivative is explicitly submitted. Native paged/resident low-order
consumers are also included in coverage, exactly once. All other classes remain
visible to the bounded exact fallback. A tuned profile still fails closed on
unexpected gaps; explicit inventory filters and portable profiles retain the
correctness fallback. Queue slices are reserved only for callable consumers.

This changes coverage, not integral equations, screening, normalization,
precision, or convergence. It applies to RHF and UHF without GPU-name selection.
The temporary force-stage downloads used to diagnose the problem were removed.

## Evidence

`tests/python/test_rhf_bounded_materialized.py` adds contracted def2-SVP water
with independent PySCF E/F references, RHF/UHF, automatic/disabled force
inventories, cold/warm execution and changed geometry. Disabling only the force
inventory deliberately leaves Fock kernels available. Strict regression gates
are 2e-9 Eh and 2e-8 Eh/Bohr. The complete file passes 20 tests on both RTX 5090
and RTX PRO 6000 (Slurm 1888/3763); the four new cases also pass on A100
(LSF 216478).

The 12/48/96-atom README HF Direct endpoint probes retain all four calls on
5090 and PRO6000. Maximum differences from the existing independent HF oracle
are respectively 6.38e-7, 1.81e-7 and 4.89e-8 Eh/Bohr; energies differ by less
than 3.2e-11 Eh. These probes use README stopping defaults, while the regression
tests use the stricter explicit tolerances above. The corrected core binary
SHA256 is `d1c30471cfba0926e19f314d406d0e9317a9833a84cc629a2becb70de5e2f665`.
Raw records, source/binary archives and scheduler receipts remain under ignored
`.artifacts/ecnu-stationary-parity-20261030/` and the allocated nodes' matching
run directories. Previously measured bounded HF timings omitted scientific work
and cannot qualify a speedup or serve as a correct performance baseline.

## Rejected diagnoses

Strict validation GEMM, disabled force-density screening, disabled generated
force kernels, final-Fock rebuild, disabled resident psss and disabled Schwarz
scheduling did not repair the missing force. Do not repeat these controls as a
substitute for auditing consumer coverage. Future registry changes must preserve
independent Fock/force inventories and the explicit bounded fallback.
