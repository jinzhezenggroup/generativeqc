# Decision: share Rys XY tables in qualified strict raw-K streaming tasks

Status: implemented
Date: 2026-11-08

## Problem

The remaining 96-atom PBE0/B3LYP Direct cold gap is not explained by extra
target SCF steps: both engines take nine. A cold-only A100 CUPTI allocation
(LSF229364) attributes 3.716408 seconds to six raw-K streaming classes:
DPPP, PPPP, DPDS, DDPS, DPDP and DDPP. Their component lanes recompute the
same primitive/root XY recurrence tables. The corresponding DF endpoint
does not execute these kernels; this diagnosis does not explain a DF gap.

## Decision

The compiler emits a separate `shared_rys_xy_streaming_fock` companion
capability. It reuses the existing scientific component task while keeping
the incumbent streaming enumeration, screening and positive-K scatter.
Capability admission is independent of paged-HF shared-XY admission.

Preparation intersects compiled capabilities with actual kernel-attribute
and occupancy checks. Automatic selection additionally requires the visible
device's actual scalar FP32:FP64 performance ratio in `(0,2]`. Architecture
strings and product names are not profitability rules. Compilation targets
identify available deployments, not interchangeable scientific qualification.
There is no new opt-in or runtime tuning benchmark.

Unknown/reduced-rate FP64 devices, absent capabilities, resource refusal,
UHF, mixed precision, compensated output and explicit alternative lowerings
retain the incumbent. Unsupported eligibility is determined before enqueue;
an execution error is not retried after partially accumulating the output.
No CPU/reference-engine work enters the production endpoint.

## Evidence

Final normal Release source identity:
`7fbcd2d2e35fac47b64517f70bba10e6af2d65c3e74112801a6c159a5a7bc51c`.
ELF SHA256:
`9a37512dc5f96f8a5d0ca78cae46c33323cb8a693620ff3bc93958f33c82534f`.
This is build16854, not the earlier unmeasured30dd source snapshot.
The control is clean parent `e462d0426e62d0e9558fb6939f19d8e4a28bd51b`,
including master `1eeb9a20e7d4350f2dbbeaa736f59144def5348b`.
The qualified source overlay excludes the inherited unfinished HF
adaptive/prepared-Fock and additional paged-HF default changes.

Both cards use fresh-process ABBA, two complete cold endpoints per arm and
three frozen-own-density warms per process. Timing includes construction,
preparation, SCF, analytic forces and host export; no interposer is loaded.

| Device / Direct96 method | Control mean | Candidate mean | Cold change |
|---|---:|---:|---:|
| A100 / PBE0 | 12.686775 s | 11.760297 s | -7.30% |
| A100 / B3LYP | 12.864098 s | 11.876885 s | -7.67% |
| RTX5090 / PBE0 | 16.803305 s | 16.909077 s | +0.63% |
| RTX5090 / B3LYP | 17.226281 s | 17.207576 s | -0.11% |

A100 LSF229664 and 5090 Slurm16855 exit0. All paired warms pass the unchanged
3% regression gate; the largest5090 ratio is1.00749. Target iterations/Focks
remain `[9,1,1,1]`; initial-guess work is reported separately. No quartet or
recurrence total was acquired, so the kernel launch census is not one.
CUPTI proves six shared-XY streaming kernels execute nine times each on
A100 and none execute on5090; dropped/starved/error counters are zero.

The unchanged README independent dense-GPU, CPU/GPU spot and grid-stability
gates pass for96 atoms and PBE0 at48 atoms. Acceptance remains E/atom1e-5,
Fmax1e-4 and Frms3e-5. Maximum native parity differences are approximately
1.37e-12 energy and5.16e-13 force. These are native-vs-native differences,
not the larger independent-engine quadrature differences.

The six-class independent positive raw-K Libcint fixture passes144 matrix
predicates on each card (A100229399 and509016846);5090 memcheck/racecheck/
synccheck pass. The reference contracts `K[first,third] += D[second,fourth]
* ERI` over unique eightfold symmetries, not combined J-K/2 cancellation.
The reusable source preparer is
`tools/generativeqc_validation/shared_rys_xy_fixture.py --consumer raw-k`.
Scientific inputs and expected arrays from the executed fixture are retained;
regenerating independent inputs on another BLAS build can change last bits.

## Rejected alternatives and limits

An earlier frozen-core shared-XY interposer pilot saves7.82-8.53%. Those
numbers are not assigned to the later normal native ELF.

Changing just DPPP/PPPP to weighted-Z after XY saves only0.86-1.06%
(approximately0.1 second) in LSF229399. Retain that mechanism observation
without broadening the separately qualified production weighted-Z target
guard or introducing an unqualified opt-in. A5090 weighted-Z guard/raw-K
fixture is not A100 production weighted-Z qualification.

A100 B3LYP GPU4PySCF is11.637084 seconds in the native allocation, still
approximately2.06% faster. PBE0 reference cold is anomalously142.657065
seconds while warms are approximately3.18 seconds. Its full record remains;
there is no measured cause and it is not used to claim a tenfold win.
Earlier independent PBE0 reference allocations are approximately11.0-11.1
seconds. This change does not establish all-README leadership, nor new DF,
HF, omega-B97M-V, other-device or other-method qualification.

## Consequences and revisit conditions

The extra companion code is justified by a complete-endpoint A100 gain and
bounded resource fallbacks.5090 keeps its incumbent default. Retain the
scientific and per-warm gates when extending class/device coverage.
Do not extend paged-HF admission through this independent capability.

After this gain, prioritize the independently attributed direct-force
spatial-SPD owner (approximately0.693 seconds), AO radial work (approximately
0.476 seconds over10540 calls), and grid/Becke work rather than the small
weighted-Z increment. Revisit based on complete cold endpoints, not their
isolated kernel timings.

The scoped qualification is retained under
`benchmarks/results/dft-shared-rys-streaming-20261108/`. Exact-FP64 decoding
authenticates all135 original retained texts and every scientific value.
Full AOT compiler stdout remains checksum/size-indexed in the ignored original
build archive and complete diagnostic capsule; identity, configuration,
compiler-cache statistics and restore receipts are checked in. Earlier pilots
are not substituted for normal native qualification.

To fit the unchanged85-MiB aggregate budget, the historical seed-products
container uses the existing exact-FP64 v3 codec:411288 to221361 bytes.
Every original container/embedded JSON byte is reconstructed and authenticated,
including all139 populations and their failures. This is storage only, not a
new numerical qualification. No other historical README summary is migrated.
The historical seed report's original dependency checksum for independent-force
tiles is already stale: both the pre-migration report/container and the migrated
reader fail at the same dependency. Do not bypass it or claim that full legacy
report passes. Exact original-byte reconstruction and the new DFT independent
qualification pass separately.

## References

- GitHub issue2976, native evidence comment6722254219.
- `benchmarks/results/dft-shared-rys-streaming-20261108/replay.py`.
- Original A100 native archive SHA256:
  `cf854cadc805b845881bd38c3c29798a97730cc61b283a264b185e7cfd71f17f`.
- `docs/maintainer/performance_engineering.md`.
