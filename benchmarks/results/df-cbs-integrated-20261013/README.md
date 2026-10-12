# Integrated correlation-only DF-CBS GPU requalification

This tree combines implementation `5911c23b02b9ebf4f55ba6bededd5ecf9454a633`
with remote CBS head `c2ef4302b19daa6632651d12e04adac44ac24779`, including landed
master's guarded triples-scalar fusion, serial Lambda scratch reuse, exact DF
final-validation submission and conventional MP2 staging/allocation recovery.
The grouped CC parent inherits the guarded complete-force defaults; standalone
and companion MP2 still omit triples and Lambda. These are new source-matched
GPU runs, not a relabelling of the pre-integration receipts in
`../df-cbs-qm7-20261012/`.

## Scientific model

```text
HF(CBS) + DF-MP2 correlation(CBS) + [DF-CCSD(T) - DF-MP2](cc-pVDZ)
```

Conventional exact RHF supplies every reference, with all electrons and FP64.
DZ/TZ orbital and RI auxiliary snapshots are explicit; energy and forces share
fixed geometry-independent extrapolation weights. Production uses native CUDA
correlation and analytic physical/orbital/nuclear pullbacks, not PySCF or
finite-difference forces. PySCF 2.14.0 is an independently reconverged oracle.

## Complete endpoints

| QM7 case | Orbital / auxiliary functions | Native E+F (s) | Energy error (Eh) | Max directional error (Eh/Bohr) |
| --- | --- | ---: | ---: | ---: |
| Index 1, C2H6, complete DZ/TZ DF-CBS | DZ 58 / 196; TZ 144 / 342 | 39.0866 | 3.68e-12 | 5.05e-9 |
| Index 5604, C7H16, DF-MP2/cc-pVDZ | 178 / 616 | 112.3420 | 7.39e-13 | 7.98e-9 |

Each is one complete first `singlepoint` E+F call, including SCF, sources,
correlation, response, publication and group cleanup. Calculator construction,
input conversion and oracle calculations are excluded. These are not replay
medians or speedup comparisons. Translation errors are `1.02e-12` and
`1.62e-12 Eh/Bohr`; the full native force arrays and all CBS components are
retained in the JSON. Directional acceptance uses independent displacements of
`1e-4` and `3e-5 Bohr`, not a full Cartesian finite-difference inventory.

Both JSON records are `qualified`: energy error <= `5e-8 Eh`, directional error
<= `5e-6 Eh/Bohr` at both steps, translation <= `1e-6 Eh/Bohr`, convergence and
actual CUDA provenance. The original QM7 checksum and Bohr geometries are
retained unchanged; the dataset itself remains an ignored local input.
Endpoint JSON is stored as lossless `.json.gz`, preserving the exact original
bytes and every source hash, force/component array and oracle comparison.
The C7H16 filename's `heptane` label does not infer a specific isomer.

The ethane 58-AO correction genuinely qualifies the DF owner beyond the
conventional 56-AO force admission. C7H16 qualifies MP2, not full large-molecule
CBS or all QM7 cases. Admission still requires numerical/resource gates.
Budgets are 4 GiB per ethane group and 8 GiB for C7H16; maximum reported planned
capacities are 1,233,237,165 and 1,972,222,310 bytes. These are not measured
whole-process/device allocation peaks. DF source storage is `O(nAO² naux)`;
generated MP2 adjoint scratch remains `O(nocc² nvir²)`.

## Reuse and validation

- Integrated Slurm GPU suite: **8 passed**. Physical water DZ/TZ DF-CBS,
  H2/water/degenerate methane DF-MP2, same-source CC/MP2/HF companions,
  source-only auxiliary g roles, atomic failure publication and budget recovery.
- Relevant integrated host/compiler/API/frontend/default-forwarding selection:
  **233 passed, 48 skipped**. An initial run had two compiled test-probe errors
  because the new demand type was absent from their stubs. The probes now use
  the actual declaration and verify factors-only forwarding for both modes;
  the complete selection is rerun, not reported as passing from the failed log.
- CI routing/time-budget/retention tests: **208 passed** after installing the
  project's declared `pytest-xdist` test prerequisite. Compiler/API/docstring,
  Ruff, formatting, retention and publication-compaction checks pass.
- Host and CUDA libraries rebuild with verified compiler caches. Raw initial
  probe failure, final pass, build and cache receipts are retained separately.

The traces retain two converged target-basis RHF owners and two physical DF
source builds for ethane. CC and small-basis MP2 share the original source
identity; the high-basis MP2 source executes zero CC core-block GEMMs. Parent
cotangents/preconditioners stay charged while the companion borrows the source.
Method-specific response solves and derivative contractions remain separate;
preliminary guesses and host orchestration are still real endpoint work.

## Provenance and reproduction

Actual CUDA library SHA256:
`b15f41470aef63e45e0f1a71f0a9d6119d39bea12296481bba5092a0c695403a`.
Each JSON retains 26 source/qualification file hashes, checked against the
integrated file bytes, including the changed upstream owners. It does not use
the exported remote checkout's stale Git HEAD as scientific provenance.

Runs use `node4` (`n4` SSH alias), Slurm `main`, one `gpu:5090:1`, CUDA 12.9.86,
SM120 and finite limits: 20 minutes for the GPU suite, 30 for the two QM7
endpoints plus their independent oracles. Preserve Slurm's device visibility.
Runtime/package and before/after cache receipts are retained; cache statistics
are server observations, not isolated compilation-speedup measurements.

Use the reproduction commands in `../df-cbs-qm7-20261012/README.md` with fresh
integrated trace/output paths. In particular, set `GENERATIVEQC_PROFILE=off`,
one BLAS/OpenMP thread, the built native library path and
`LD_PRELOAD=$PWD/build/libgenerativeqc.so` with CUDA 12.9 on `LD_LIBRARY_PATH`.
The preload prevents this host's incompatible CUDA 12.4 Python providers from
loading first; it is an environment prerequisite, not a CPU/reference fallback.
