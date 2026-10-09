# OH UHF spatial diagnostic fixture (#1791)

The JSON fixture retains new, actual CPU primary densities and independent
PySCF tensors for neutral OH/STO-3G: O(0,0,0), H(0,0,1.834) Bohr and the 1.01
moved bond 1.85234 Bohr. Alpha/beta electron counts are 5/4. Native scalar and
OpenBLAS providers use separate source-matched libraries; their identities,
oracle/library versions, hashes, controls, and actual SCF work are recorded.

These are new executions, not recovery of the e3f66381/c9d9f8ef binaries or
original matrices. All four historical raw FAIL records remain in the original
`cpu-bounded-scalar-hf-20261003` capsule and are copied unchanged in this fixture.
Current raw D/projector outcomes are reported at the unchanged absolute 1e-7
gate. The separate spatial certificate cannot change those decisions.

`tools/oh_uhf_symmetry.py` constructs only an axial spatial rotation in the
explicit spherical O 2p=(x,y,z) block. It verifies geometry, S/H/full ERI,
Fock covariance, counts, idempotency, physical commutators and energies of the
original and transformed states. Qualification uses one common angle for both
spins, including full spin densities and metric projectors. It never transforms
the endpoint's retained production density in place.

The runner reads the actual primary warm-state density immediately after each
energy+force endpoint, before closing the same batch. A later independent
PySCF solve uses exact bundled primitives and confirmed AO ordering; no native
reference-export solve supplies a replacement primary density. The fixture
also includes an independently solved, gapped H2 state to preserve strict raw
comparison outside the supported OH domain.

## Recorded CPU results

Captured from source `7c07fa309cf7f9123cde697e460169e481bfca01` with two
distinct library hashes and the same verified native scientific source identity.
All four raw D/projector comparisons are **FAIL**; all separate spatial and force
gates are **PASS**. The provider labels bind the build configuration, not a claim
that every small-matrix operation calls a BLAS routine. The API's
`cpu_reference` backend name denotes this native C++ CPU HF endpoint; PySCF
supplies the subsequent independent oracle only.

| Build provider | Geometry | Raw D max error | Rotated D max error | Iterations / Fock builds |
| --- | --- | ---: | ---: | ---: |
| scalar | original | 0.8194010709038728 | 7.86222e-11 | 9 / 12 |
| scalar | moved | 0.467758639658178 | 4.47768e-11 | 9 / 11 |
| OpenBLAS | original | 0.8194010709038728 | 7.86222e-11 | 9 / 12 |
| OpenBLAS | moved | 0.467758639658178 | 4.47768e-11 | 9 / 11 |

Across these new states, physical residuals are at most 1.62936e-11,
energy discrepancies at most 5.68435e-14 Hartree, and force errors at most
5.08394e-12 Hartree/Bohr. The original endpoint's 12 Fock builds differ from
the historical 11; actual new counts are retained rather than copied from the
capsule. `sccache 0.16.0` wrapped all 213 compile commands per build. Scalar
had 213 misses; OpenBLAS added 212 misses and one hit, with no compile failures.
The fixture retains the before/after statistics and command hashes.

The dedicated qz regression passed all 14 tests, including molecular
Hamiltonian mutation and occupied-to-virtual promotion rejection. This verifies
the restricted diagnostic; it does not resolve the determinant policy.

## Reproduction

Build clean CPU libraries with the repository compiler-cache rules, one with
`GENERATIVEQC_CPU_LINALG_PROVIDER=scalar`, one with `openblas`. Preserve the
launcher version, before/after statistics, actual commands and library/build
identities. Set `PYTHONPATH` to the exact source's `python/` and repository root,
and `GENERATIVEQC_LIBRARY` to the selected library. Use PySCF 2.14.0 and explicit
one-thread OpenMP/BLAS settings. For each provider, in a separate process:

```bash
taskset -c CPU /absolute/venv/bin/python -m tools.run_oh_uhf_symmetry \
  --source /absolute/source --source-sha EXACT_HEAD \
  --library-sha ACTUAL_SHA256 --provider scalar --output /absolute/new/scalar.json
```

Repeat with `openblas` and its own library/hash/output. The runner checks source,
native scientific identity, provider, primitive and AO bindings, and refuses an
existing output. Retain every failed attempt. The JSON fixture combines the
two reports without altering their recorded matrix values or raw decisions.

Run `python -m pytest tests/python/test_oh_uhf_symmetry.py -q` to reconstruct
certificate metrics from retained molecular matrices and exercise adversarial
non-equivalent integer states, broken Hamiltonian symmetries, incompatible spin
rotations, unsupported/nondegenerate fixtures, and strict tolerance boundaries.
Synthetic algebra checks are explicitly identified and are separate from the
molecular CPU evidence.

The command's successful exit indicates the separate diagnostic/force gates
passed. It does not assert raw-density PASS, a unique canonical determinant,
GPU qualification, performance improvement, or complete allocation coverage.
The canonical determinant/equivalence-class policy remains unresolved in
#1791; this slice references rather than closes the issue.

## Current comparison policy

The separate [OH comparison policy](../../docs/maintainer/oh_uhf_comparison.md)
now accepts independently certified spatial equivalence in this frozen domain.
The original capture, certificate and runner remain unchanged and retain their
historical unresolved-policy text and raw FAIL decisions. The policy evaluator
recomputes the certificate from retained tensors without altering that evidence.
