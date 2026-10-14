# Decision: budget-selected fitted composite geometry

Status: implemented within the independently checked small-domain boundary
Date: 2026-10-14

## Scientific boundary

The existing stationary consumers already own full-range DF response, exact
long-range exchange and additive canonical 3c corrections. Reuse their shared
FP32 semilocal geometry, not a second equation or a name-only capability flag.
WB97M-V admission checks the complete range/VV10 MethodIR and live full-range
DF J/K proof, with the default 1e-10 metric threshold. Only its semilocal
geometry narrows; DF response, exact long-range K, VV10, SCF and AO production
remain strict. Its distinct schema-4 qualification requires the composite
owner's `mixed_rsh_df_provider` work evidence.

Canonical r2SCAN-3c DF requires both the electronic r2SCAN graph and the exact
complete 3c graph. The public constructor already validates the defining basis.
The consumer publishes electronic `r2scan-df` evidence; only the facade can
qualify the complete `r2scan-3c-df` model after successful typed D4 and gCP
results. Neither correction is relabeled as FP32. Modified corrected graphs
retain explicit audits.

The current small domains and budget floors are unchanged. This is not HF DF,
all-size README coverage, all-FP32 arithmetic or evidence of a speedup.

## Independent reference

The optional GPU adapter in `benchmarks.pbe0_precision.reference` now installs
full-density rebuilding on the final object **after** density fitting. For WB
it reuses the separately qualified README adapter: full-range fitted J/K with
exact long-range K in both SCF and analytic forces. A stock all-fitted RSH
oracle changes the Hamiltonian and is not valid. The CPU adapter therefore
rejects fitted WB rather than silently checking a different model. Ordinary
semilocal CPU references and the default backend are unchanged.

3c independently combines GPU4PySCF r2SCAN/JKFIT response with upstream dftd4
4.3.0 and the separate gCP reference. All reference generation is outside
production. Tests include complete scalar, prepared, frozen and explicitly
moved endpoints; real pending-work strict retry; failed-peer isolation and
energy-only provenance expiry. Host admission and policy checks pass 174 tests.

## First qualification attempt: compiler prerequisite failure

Slurm 8428 failed the first two WB fixtures **before native force execution**:
the runtime sccache/NVCC host probe could not execute `cc1plus`. The independent
GPU oracles had completed. No precision gate was weakened. Retain its XML/log
and distinguish this environment failure from numerical rejection. Slurm 8429
was interrupted during its independent timing reference before repeating the
same invalid compiler environment; its partial raw directory remains retained.

The retry explicitly identifies GCC 12 with `NVCC_CCBIN`, preserving compiler
cache identity and Slurm visibility. Both cache launchers are version-verified.
Native library remains the n1 `f993f2ba55c378b4c8102a682226ad8ed7193903d1dc674632445b3fec886bf6`
build; Python sources are isolated under `fitted-composite-source`.
Slurm 8430 passed the entire water scalar/prepared/frozen/moved fixture with
that explicit compiler. The source-matched Slurm 8432 subsequently passed all
nine fitted tests: four WB models/representations, three 3c models and both
real pending-work recovery/evidence-lifetime checks. Its following direct WB
regression also passed all nine tests (18 passed in the whole 8432 allocation).
The final XML and complete job log are retained locally. Slurm 8437 passed the additional coarse-grid
3c and both tight-budget audit tests (3 passed, 9 deselected). These are public
selection tests, not diagnostic overrides or acceptance of a Fast keyword.

## Complete actual README endpoints

Slurm 8433 separately ran the actual README CLI for both independent reference
and native engines on an assigned n1 RTX 5090. All 48 reference and **96 native**
calls completed and passed. The checker re-verifies scientific-protocol equality
and all independent norms, retaining raw input hashes in
`fitted-composite-8433-verified.json`. Each warm series includes all five frozen
replays, not just a favorable subset. No profiler or extra oracle runs inside
the native endpoints; corrections and all actual retries stay timed.

| Model / AOs | Phase | Ordinary FP64 s | Fast s | FP64/Fast |
| --- | --- | ---: | ---: | ---: |
| WB97M-V DF / 24 | Cold | 6.962992 | 6.934239 | 1.0041 |
| WB97M-V DF / 24 | Frozen warm | 0.725088 | 0.657673 | 1.1025 |
| WB97M-V DF / 24 | Moved | 5.280122 | 5.254745 | 1.0048 |
| WB97M-V DF / 24 | Moved frozen | 0.725049 | 0.656985 | 1.1036 |
| WB97M-V DF / 96 | Cold | 50.725618 | 50.464935 | 1.0052 |
| WB97M-V DF / 96 | Frozen warm | 5.679719 | 5.401134 | 1.0516 |
| WB97M-V DF / 96 | Moved | 37.842829 | 37.545854 | 1.0079 |
| WB97M-V DF / 96 | Moved frozen | 5.673710 | 5.398585 | 1.0510 |
| r2SCAN-3c DF / 34 | Cold | 2.793337 | 2.758306 | 1.0127 |
| r2SCAN-3c DF / 34 | Frozen warm | 0.131590 | 0.112628 | 1.1684 |
| r2SCAN-3c DF / 34 | Moved | 0.982886 | 0.992583 | **0.9902** |
| r2SCAN-3c DF / 34 | Moved frozen | 0.130789 | 0.110259 | 1.1862 |
| r2SCAN-3c DF / 68 | Cold | 3.596435 | 3.508436 | 1.0251 |
| r2SCAN-3c DF / 68 | Frozen warm | 0.306969 | 0.230681 | 1.3307 |
| r2SCAN-3c DF / 68 | Moved | 1.623949 | 1.503509 | 1.0801 |
| r2SCAN-3c DF / 68 | Moved frozen | 0.308856 | 0.230270 | 1.3413 |

Cold/moved Fock counts match between modes: WB 13/11 and 19/14, 3c 13/9 and
15/11. Every warm call has one Fock, every Fast call one solve and complete
returned FP32 semilocal geometry (24,576 points per atom). SCF and correction
counts are not hidden by the precision facade. Fast maximum independent force
errors are 4.548e-8/7.833e-8 Eh/bohr for WB and 1.539e-7/1.693e-7 for 3c;
maximum RMS errors are 2.747e-8/2.953e-8 and 1.060e-7/1.133e-7 respectively.
All are below the explicit 1e-4 maximum and 3e-5 RMS allowances; energy errors
are below 4.377e-12 Eh. Ordinary FP64 passes the unchanged 1e-8 Eh / 1e-7
Eh/bohr strict gates. This is not a mathematical error certificate.

The losing 34-AO moved phase is retained. Cold gains are small; neither the
stronger 68-AO warm ratio nor mixed-point counts establish all-FP32 SCF, a 2–7x
endpoint claim or benefits at every size. The old API-cold, before-improved-FP64
comparators are not used. HF DF and the larger README sizes remain open.

Full raw files, failed prerequisites, source tarballs, XML and source hashes
remain in ignored local and n1 `readme-budget-fast-20261013` directories.
The timed admission source hashes are `5ec1f486b30009bd9f43a5223c87f3fc017662dde4ce50dbd26d125d2b3cfd85`
and `efe0ef725afc0e213c7ff2df38014d4d6e4a78453d9b9a22b7b36e286d5eb36d`
for fitted/composite consumers. Later docstrings and the added composite
source-inventory entry do not relabel the earlier raw receipts.

This supersedes the corrected/RSH-DF exclusions in
`2026-10-14-public-df-budget-gradient.md` and the private-only boundary in
`../../proposed/2026-10-14-composite-fp32-geometry.md`, only for these canonical
models and bounded domains. Their earlier negative timing populations remain
valid historical observations.
