# Native CPU hybrid follow-up: retained scope at user stop

This is a historical qualification record for the existing-results PR requested
on 2026-11-14. It is not full acceptance of issues #2844--#2848. Further chemistry
and scale work was explicitly stopped by the user; do not interpret the pending
jobs below as work that this goal will automatically resume.

Current execution and acceptance rules remain owned by
[the Hessian contract](../../../docs/developer/dft_hessian.md),
[geometry optimization](../../../docs/user/geometry_optimization.md), and
[the validation guide](../../../docs/maintainer/validation.md).

## Delivered implementation

- Linked native CPU FP64 B3LYP RKS/UKS total Hessian/HVP composition, with generated
  AOT integral, moving-grid XC, response and two-body D3(BJ) sources; explicit
  source identity, resource/work refusal and transactional output contracts.
- Canonical spherical Ni/I/Cs controls, residual scalar-ECP curvature and shared
  CPU molecular evaluation, without importing independent reference engines in
  production or admitting public molecular ECP saddles.
- Fixed-method saddle execution and native CLI/API controls, including bounded
  attempts, root continuity, stationary index/mode checks and failure recovery.
- Independent cold-state, projected-response, finite-difference and mode controls;
  reference-only DIIS/Newton conditioning does not alter physical acceptance gates.

The PR preserves the accumulated implementation rather than only the last
reference-helper changes. It does not claim that every delivered component has
completed every issue-wide endpoint qualification.

## Observed results and limits

The machine-readable [evidence record](evidence.json) embeds original receipts;
their original pending/accepted flags are preserved rather than rewritten.

| Observation | What it does and does not establish |
| --- | --- |
| Five accepted cold installed proper cube-rotation cases: Cs RKS/UKS, I RKS/UKS, mixed stationary Ni/I RKS | Physical state, raw full Hessian and public HVP covariance for rotations preserving the finite default grid. Not arbitrary-angle or exact continuum rotational invariance. |
| Independent 25-atom all-electron central reference: physical orbital-gradient norm `3.3260817238274695e-12` | Genuine cold independent state and gradient. Not native scale, state matching or H/HVP acceptance. Four displaced references completed; four were cancelled and eight were never started. |
| Independent 25-atom Ni/I-ECP central reference: norm `3.2280734254209157e-12` | Genuine cold independent state and gradient. Native public energy/force was cancelled before returning, so the practical ECP force gate remains open. |
| Native NiO/H2 final observation: 12 accepted steps, maximum gradient `0.00018600427053553416`, index two | Neither a stationary point nor a first-order saddle. Final independent state/mode/reaction audits never started. |
| Public all-electron scale HVP: exit 124 | Its two-hour bound expired during the cold primal; no HVP was returned. This is not a diagnosis of mathematical nonconvergence. |
| Remaining native scale/TS jobs and dependent waiter: exit 143 | Explicit user-requested cancellation, not numerical rejection or acceptance. The waiter and reference manager were stopped before their dependent work. |

The original method, basis, default grid and strict physical gates were not
relaxed. The mixed stationary Ni/I raw continuum rotational residual remains
about `0.0020083`; projected mode analysis is explicitly a convention, not a
repair or exact-invariance claim. Both public ECP saddle guards remain closed.

## Source and regression provenance

The measured native source identity is
`daa312d601759a05805a5c29c703806cffca82ca57db62f77d7d84fef71f1ff0`;
the installed core SHA-256 is
`e99b4290141db29765d53329fdb4f454d67a3842bc5b0121fd1dce09c2302927`.
The consolidation commit preserves this source identity. These are historical
measurements, not qualification of a later integration or rebuild.

The bounded closing regression passes **183 tests**, with **six explicitly
skipped opt-in heavy cases**. The separate heavy cube-rotation executions are
not inferred from those skips. Compiler structure checks cover 612 modules with
zero dependency errors; Ruff lint, formatting and whitespace checks pass.

To reproduce the closing regression against an already built matching CPU
library, set `GENERATIVEQC_LIBRARY` explicitly and run:

```bash
PYTHONPATH="$PWD:$PWD/python:$PWD/tests/python" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 180s python -m pytest -q \
  tests/python/test_native_canonical_heavy_hessian.py \
  tests/python/test_molecular_saddle_spin_roots.py \
  tests/python/test_global_hybrid_ecp_basis.py \
  tests/python/test_independent_diis.py tests/python/test_independent_newton.py \
  tests/python/test_source_identity_manifest.py \
  tests/python/test_installed_cpu_hessian_validation.py \
  tests/python/test_native_cpu_ecp_admission.py
```

## Retention and review disposition

Full producer inputs, logs, arrays, terminal statuses and cancellation receipts
are retained in the ignored **local-only** archive
`build/qualification/host-root-registered-continuation-20261114/user-stop-final-supplement-v1.tar.xz`.
Its SHA-256 is
`c81ebbbb00a5eaf02b069259132c137266d2947f2088fcd4e5435e93b692317c`.
Every archived file was reread and checked against its original; the compact
record identifies the local retention manifest and prior immutable archives.
This PR does not publish those large archives or create a Release/asset/tag.
The compact record is not a substitute for raw-array review or independent
reproduction when promoting further endpoints.

Upstream `master` at `17f9317a2` includes later work and the squash of prior PR
#3062. Integrating it produced 32 conflicting paths, including scientific code,
tests and generated provenance. That merge was aborted without changing the
qualified tree; resolving it safely is a separate integration gate. Submit this
preserved contribution as a **draft**, not as merge-ready or a full fix.

Keep #2844--#2848 open. Practical 25--40-atom public derivative/force acceptance,
the representative requested-level Ni reaction TS and public ECP saddle admission
remain unresolved. No further long-running work is scheduled by this handoff.
