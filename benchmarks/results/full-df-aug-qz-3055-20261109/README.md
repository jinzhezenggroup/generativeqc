# Full-DF augmented-QZ CUDA qualification for #3055

This is source-matched correctness evidence, not a release-performance or
speedup qualification. The build enables CUDA fast compile and disables optional
stationary/direct AOT profiles. No release/tag or published binary is created.

## Identity and environment

- Base source: `e16c84a00de8664836da846ed0e2300e4dbd896a`.
- Implementation patch SHA-256:
  `8065f3f2bd509b57f84c79648a792d31a5fcb39156c3d70323a7e3cc56659bdf`.
  `implementation.patch.xz` applies to that base; `source-files.sha256` identifies
  the changed implementation, generator, test and oracle files. Documentation
  and evidence additions are outside this implementation digest.
- Loaded native-library SHA-256:
  `6a89c622ea5fbe7e8a96984bc6683e02f686a668d2c49bc4c57043b20028038a`.
- CMake/Ninja Release, CUDA 12.9.86, SM 120, explicit sccache 0.16.0 CXX/CUDA
  launchers. Complete configured flags are in `CMakeCache.txt.xz`; launcher
  commands and before/after cumulative cache counters are retained separately.
  Cache counters can include other work using the same shared cache server.
- Python 3.13.9, PySCF 2.14.0, NumPy 2.3.5. Production uses the native CUDA
  library; PySCF/libcint is used only by the independent test/oracle process.
- Node1: RTX 5090, 32607 MiB, driver 580.95.05, Slurm job 16947, visibility 1.
  Node2: RTX PRO 6000 Blackwell, 97887 MiB, driver 595.99.02, Slurm job 4218,
  visibility 1. Every GPU execution, including memcheck, is inside finite `srun`.
  No GPU work is executed on the maintained local node3.

## Reproduction

`run-molecular.sh` and `run-probes.sh` preserve the executed Slurm commands,
allocation limits, opt-in test flags, staged library and Python search paths.
Their absolute checkout paths describe the measurement staging directories;
adjust those paths to a checkout of this implementation with the identified
native build. Preserve Slurm's assigned `CUDA_VISIBLE_DEVICES`.

The molecular tests use unmodified PySCF orbital and JKFIT basis records,
spherical AOs, all electrons correlated and no counterpoise correction:

- CN-, charge -1, singlet: the exact issue geometry, converted to Bohr using
  0.529177210903 Angstrom/Bohr; complete aug-QZ, 160 AOs, and its full JKFIT.
- CH+, charge +1, singlet: C at (0,0,0), H at (0.3,-0.4,2.4) Bohr; complete
  aug-QZ, 126 AOs, and its full JKFIT. The separate CCSD(T) endpoint returns
  RHF/MP2 companions from the same stationary reference/source.
- CH+ T/Q CBS, with the correction separately at augmented DZ and TZ. HF
  exponent 1.63 and MP2-correlation exponent 3 give actual HF weights
  `(-0.24367215584898735, 1.2436721558489874)` and MP2-correlation weights
  `(-27/37, 64/37)`. Component RHF total-energy coefficients differ from HF
  weights because the correlation extrapolation subtracts its RHF references.

Component JSON files retain input basis decimals and provenance checksums,
root/occupation/metric diagnostics, absolute native and independent energies,
analytic forces, both directional finite differences, native numeric capacities,
complete endpoint timings and semantic source/transform work. Raw traces are
compressed independently; no production reference-oracle fallback is enabled.

## Numerical and resource gates

Standalone component energy error must be below 5e-9 Hartree; CBS component and
assembled energy error below 1e-8 Hartree. Independently reconverged displaced
energies at both 1e-4 and 3e-5 Bohr must agree with analytic directional forces
within 3e-6 Hartree/Bohr, separately for every component and for the assembled
CBS force. Translation tolerance is 3e-7 Hartree/Bohr. Replay and energy-only
calls are separately checked. Tests fail rather than accept a different root.
`summarize.py` additionally reconstructs independent complete Cartesian forces
from each energy slope using diatomic rotational invariance and the nonzero
bond projection of the recorded displacement. It checks every Cartesian
component against the same 3e-6 gate; `summary.json` retains these comparisons.

The correlation numeric budget is 2 GiB, not a total process GPU-memory bound.
`measured_endpoint_peak_bytes=0` in native diagnostics means unavailable; it
must not be interpreted as a measured zero. `memory_observation` instead samples
the allocated process through NVML every 100 ms, including CUDA libraries and
driver/context overhead. This sampled process peak can exceed the numeric
budget and can miss short peaks. Raw per-process samples are retained alongside
the endpoint JSON. Do not confuse the planned numeric bound, owned buffers,
host staging and actual sampled device-process memory.

## Completed supporting gates

- `final-molecular.log.xz`: five higher-angular molecular tests passed on node1:
  separate QZ CCSD(T)/MP2 endpoints, T/Q CBS with DZ/TZ corrections, and the
  orbital-g/low-l-fitting policy regression. Both finite-difference steps pass.
- `qualified-host.log.xz`: 778 passed, 932 optional CUDA/variant skips.
- `qualified-budgets.log.xz`: 62 passed, 2 optional native-probe skips.
- `final-probes.log.xz`: 293 independent CUDA value/derivative checks passed,
  216 expected variant skips; four weighted S/T/V gradient tests passed under
  Compute Sanitizer, zero errors and zero leaked allocations.
- `final-low-l-molecular.log.xz`: five complete lower-angular CUDA E/F,
  companion, replay and energy-only regressions passed on node2 (Slurm 4220),
  including spherical/Cartesian MP2 and an independent CCSD(T) force endpoint.
- Public API docstrings, compiler structure (602 modules, zero dependency
  errors), Ruff and Sphinx warnings-as-errors pass.

## Complete endpoint measurements

| Endpoint, including requested E/F companions | Native wall (s) | Sampled process GPU peak (GiB) | Max component/assembled energy error (Eh) | Max reconstructed Cartesian force error (Eh/Bohr) |
| --- | ---: | ---: | ---: | ---: |
| CH+ aug-QZ CCSD(T), MP2, RHF | 76.3873 | 5.2695 | 1.171e-11 | 4.157e-9 |
| CN- aug-QZ MP2, RHF | 90.0398 | 5.3379 | 9.664e-13 | 3.604e-8 |
| CH+ T/Q CBS, DZ correction | 67.8448 | 5.0781 | 6.992e-12 | 3.047e-8 |
| CH+ T/Q CBS, TZ correction | 64.7059 | 5.0781 | 7.866e-12 | 3.032e-8 |

The first endpoint starts the process's CUDA context; later endpoints share
that warm context but create/release their own native method owners. Timings
exclude independent CPU oracles and NVML sampler startup/teardown. The full
five-test molecular run takes 1869.67 s, mostly independent reconverged CC
references. The smaller CBS corrections are not QZ CC force calculations.

The QZ CH+ CC provider reports one source scan, 126 source reads, 3,460,968
source values, 1,716,445,890 transform FMAs, 133 transform stages and five MO
blocks. CN- MP2's correlation-source scope reports 160 rows, 7,270,400 source
values, 162 transform GEMMs and 4,391,321,600 transform summands. The latter
reuses the reference source (zero new source constructions/factorizations in
that scope), not a hidden oracle or repeated full integral construction.
`native-progress.jsonl.xz` retains all endpoint phases, work observations,
shared-source identities and replay/energy-only execution. CC/Lambda/orbital
iterations and response residuals remain in the component JSON diagnostics.

## Retained failed development experiment

`molecular-gpu-5.log.xz` is an earlier development-binary run, not final source
evidence. CH+ at H=(0.3,-0.4,2.0) Bohr passed independent QZ CCSD(T), MP2 and
RHF energies, both force-difference steps, replay and energy-only checks; CN-
also passed. Its CBS DZ correction instead converged to a different native HF
root (reference energy -37.63051739899468 Hartree). The independent same-input
reference was lower. The test correctly failed; no tolerance was relaxed.
`root-probe.py` / `root-scan.log.xz` record a native-versus-independent DZ
geometry scan used to select the common-root CH+ geometry for final qualification.
SCF residual convergence alone is not a ground-state or root-identity guarantee.

Heavy-atom orbital h/auxiliary i and Q/5 remain explicitly unsupported; see the
separately staged extension in `docs/user/ccsdt_cbs.md`.

## Exact offline storage and replay

All 36 original files, including this original README, are retained in the
existing `generativeqc.exact-fp64-evidence.v3` container. No receipt was dropped.
`sha256.txt` authenticates both compressed streams; every reconstructed file
has its original decompressed UTF-8 SHA256 inside the container. The 6,090
FP64 values retain their exact bits and original JSON formatting. All 49,399
native progress JSONL records retain their raw C++ numeric spelling as text.
Failed development runs, exact basis decimals, implementation patch and source
manifest remain recoverable. This representation does not upgrade historical
CUDA evidence to current-source or release-performance qualification.

From the repository root, authenticate, restore and replay all four independent
E/F gates with:

```sh
python benchmarks/results/full-df-aug-qz-3055-20261109/replay.py
```

To restore every original file for inspection, pass `--output` with a new empty
directory. Names omit the former `.xz` suffix because recovered bytes are the
original decompressed texts. The recovered original `summarize.py` replays the
unchanged scientific thresholds. Run it with assertions enabled. Original
compressed storage was 1,118,571 bytes; the three authenticated streams total
549,209 bytes, before this readable guide and replay helper.
