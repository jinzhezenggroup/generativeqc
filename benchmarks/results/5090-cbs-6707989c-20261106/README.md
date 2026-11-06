# RTX 5090 CCSD(T)/CBS: stopped at the source-comparability gate

Status: **INCOMPLETE; build and measurements NOT_RUN**. This is a preflight
report for the proposed Figure 5 companion cohort, not a measured RTX 5090 curve.
The user required stopping if changes after the H100 source affect the
closed-shell CBS default path. That condition is met by #2982. Freeze a final
integrated source and remeasure **both H100 and RTX 5090** before making the
requested current-implementation comparison. No replacement source was selected
or executed in this task.

## Source audit, 2026-11-06

The requested unmodified source is
`6707989cce434b0ca4ef43f9920d5f91525f30c1`, tree
`f9cb8c86cf88eda9ce7f7e858d0d7108c3baffdf`. The audit endpoint is upstream master
`7109b3d208d23352f266e2bf7440fdfcc71df974`. The 23 intervening first-parent
commits and exact relevant patches are retained in `records.json`; replay reads
their immutable Git objects, without network access or a native library.

| Change after measured source | Closed-shell CBS assessment |
| --- | --- |
| #2982, `c630ceaeb18846fca5a98f3b621f67fc34938b2a` | **STOP**. Despite its unrestricted title, this changes the restricted CUDA DF-HF reference driver used by full-DF CBS. The retained default fallback now normalizes DIIS, validates/projects roundoff-level Fock asymmetry, and uses bounded 50/50 startup density damping through iteration 32. Reference failure propagation also changes. |
| #3007, `94e1eaf529d91cbb2aa7a163367caeb30beae1d8` | Changes default CUDA DF Hückel seed validation products when the target has at least 512 AOs and the existing OpenBLAS provider is available. Other sizes and an unavailable provider retain scalar products. No assertion that this cohort exercises that size branch is needed for the stop decision. |
| #3008, `1943a24b7c851a9c2bf4f68a1026c9f637c98f12` | Changes omitted public HF/KS controls and the native policy receipt. It does **not by itself** replace CBS-owned explicit convergence controls: `cbs.py` is byte-identical across the audit range, supplies its own controls, and retains the density ceiling of `1e-11` and physical CC residual cap. A public-default change alone is not proof of changed CBS numerical results. |

The relevant call chain in the pinned source is
`CCSDTCBSCalculator.singlepoint` (full-DF grouped components) -> native correlated
reference construction -> `hf::DFReference::solve` ->
`run_rhf_density_fitting_cuda_bucket_cached` ->
`run_rhf_density_fitting_cuda_bucket_impl`. See pinned `python/generativeqc/cbs.py`,
`src/hf/df_reference.cpp`, and `src/scf/rhf.cpp`. The latter tries the compact
device SCF first and, on a supported nonconverged/rejected compact solve, retains
the host-orchestrated DIIS path with device J/K. That built-in fallback is part of
the default algorithm, independently of the separate `density_fitting_host_retry`
flag, which the DF reference sets false. It is not an automatic experiment retry.

The #2982 changes are therefore reachable by a closed-shell full-DF CBS consumer;
they cannot be dismissed as UHF-only. This static audit does **not** establish
that a particular H100 receipt took that branch, that any of the 20 inputs would
take it on RTX 5090, or that their converged energies/forces would differ. No such
numerical or performance claim is made. The conservative source-path stop is the
user's requested admission rule. Further implementation auditing cannot turn it
into permission to run the old-source cohort.

## Frozen protocol and planned work

The planned experiment has the **same source and protocol as the H100 cohort**
in [the H100 README](../h100-cbs-5090-cohort-20261105/README.md); this is a plan,
not an executed equivalence claim. Scientific records are read-only from
`cbs-routine-standard-20261025` at
`4370d6c4b1bf4033818f31a9fcc622d3045a33bf`, SHA-256
`03b2722d016c13daaf6025a021de18e0ac333ebbb6efe94943c57748a57f5d3a`.

The planned population is 13 small-molecule cases (11 full-DF, two
correlation-only DF), including augmented orbital and JKFIT bases for diffuse
hydroxide, and QM7 C1-C7 at zero-based indices 0, 1, 8, 48, 183, 293, 5604.
All-electron cc-pVDZ/cc-pVTZ extrapolation and the cc-pVDZ correction use the
matching JKFIT bases. Default and tight observations remain separate.

Planned counts are 73 primary E+F calls, 20 separate startup E+F calls, and 13
energy-only calls. Completed counts are all zero because admission stopped.
Every timed call would construct a new outer CBS owner and include inner
preparation; basis-table preparation, imports, process/CUDA startup and
compilation remain outside timing. Each process's startup call stays separate.

The declared controls remain one math thread and `CUBLAS_WORKSPACE_CONFIG=:16:8`
before imports, with numeric budgets 32 GiB for QM7 and 8 GiB for small molecules.
The old 5090 README already declares 32 GiB for QM7. No memory admission was
attempted here, no budget was changed, and no physical OOM or resource rejection
is inferred from the card's nominal VRAM capacity.

All original gates remain: energy `1e-7 Eh`; maximum force `3e-6 Eh/Bohr`;
composite RMS and finite-difference stability `1e-6 Eh/Bohr`; constituent energy
locality `0.01 Eh`; energy-only parity `1e-8 Eh`.

| QM7 case | Atoms | Default median (s) | Two-call range (s) | CCSD iterations, first default | Final CC residual |
| --- | ---: | --- | --- | --- | --- |
| C1 | 5 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C2 | 8 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C3 | 11 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C4 | 14 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C5 | 17 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C6 | 20 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |
| C7 | 23 | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

| Population | Energy error (Eh) | Force maximum (Eh/Bohr) | Force RMS (Eh/Bohr) |
| --- | --- | --- | --- |
| full-df--water | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--water-stretched | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--ammonia | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--ammonium | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--hydroxide | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--formaldehyde | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--nitrogen | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--water-dimer | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--hydroxide-diffuse | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--hydrogen-stretched | NOT_RUN | NOT_RUN | NOT_RUN |
| correlation-df--water-stretched | NOT_RUN | NOT_RUN | NOT_RUN |
| correlation-df--ammonium | NOT_RUN | NOT_RUN | NOT_RUN |
| full-df--nitrogen-stretched | NOT_RUN | NOT_RUN | NOT_RUN |

Two calls would provide **descriptive medians and ranges only**. A future paired
comparison applies **only to this configuration**, without a general hardware
speedup claim. The historical H100/5090 timings are not substituted for missing
new measurements.

## Build, allocation and reader

SM120/CUDA 12.9 build: **NOT_RUN**, because source gate 1 failed before gate 2.
The proposed 5090 core identity, binary hash, actual device model/UUID/driver,
image/OS identity and Slurm job/step are all **UNAVAILABLE** (`null` in the
summary). No GPU allocation, compile, experiment retry, patch or production edit
was performed. The existing H100 core hash is historical evidence and is not a
5090 binary identity. The original 5090 publication is unchanged.

Run the stdlib-only CPU reader from a full Git clone:

```sh
python benchmarks/results/5090-cbs-6707989c-20261106/replay.py
```

It authenticates frozen inputs, the inspected Git range and retained diffs,
checks the explicit stop state, and reports `audit_integrity: PASS` separately
from `scientific_status: NOT_RUN`. It performs no imports of GenerativeQC,
NumPy, CUDA, compilers or scientific solvers. This report supplies no Figure 5
measurement points and does not qualify either GPU on a final integrated source.
