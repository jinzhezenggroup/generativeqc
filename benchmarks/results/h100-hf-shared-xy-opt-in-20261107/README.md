# Default-off shared-Rys HF companion pilot

`GENERATIVEQC_CUDA_HF_SHARED_RYS_XY=1` requests the bounded SM90 restricted-FP64
DPPP/PPPP materialized Fock companions. Absent, empty or `0` keeps the incumbent;
other values fail before preparation. Unsupported valid requests retain the
existing owner. Prepared CUDA resource/restart identity binds this selection;
CPU identity, ordinary defaults, queue capacities and force ownership are unchanged.

On the measured H100, one complete cold E+analytic-force+host-export pair improves
11.9574236162 to 10.6378763989 seconds: **1.3195472173 seconds / 11.03538%**.
The fixed order is off then on, with fresh owners. This is one descriptive pair,
not a timing distribution or a fresh GPU4PySCF comparison.

| Call | Off seconds | On seconds | On/off |
| --- | ---: | ---: | ---: |
| Cold | 11.9574236162 | 10.6378763989 | 0.88964619 |
| Warm 1 | 1.5849162303 | 1.4804518931 | 0.93408842 |
| Warm 2 | 1.5878549442 | 1.4758048840 | 0.92943306 |
| Warm 3 | 1.5829769485 | 1.4775780626 | 0.93341730 |

All three paired warm ratios pass the predeclared 1.05 guard. `summary.json`
contains the exact retained seconds and full recomputed metrics. Constructor,
prepare, Huckel initialization, SCF, analytic forces and host export are included
in cold. Warm uses the same owner's frozen engine-local cold density. Both arms
use native_selector 0, original Standard settings, one math thread and independent
copies of the same prior artifact cache, with tracing disabled.

The main cold solve has 11 iterations/11 Focks in each arm. The common Huckel
initializer separately records seven preliminary iterations/Focks, also inside
cold. Every warm has one iteration/Fock. Precision, settings, initial-guess work
(excluding its timing), incremental policy and warm behavior agree. Full operator
inventory is not claimed where the retained precision receipt marks it incomplete.

## Source and numerical scope

Measured commit `a694beec91645e0c7002bcb84976e2041ae02fa3`, tree
`73f7c92a0f8508e0953f0eae45a7ba90ba4e6655`; native input identity
`fea322c023b121519d3b35b50ed92fa022f6096ccdeed3b5b1fcf20f9598bc18`.
Core SHA256 is `136cc845ed20041978dd6f98bafdd2610b84c93a40724298a8573f8493545443`.
Physical device: `GPU-2b5c4936-338e-88e5-49e3-e4d3cceccf52`, H100 80GB HBM3,
SM90, MIG disabled, driver595.58.03. Exact runtime, packages, allocation, geometry,
controls and compiler/cache provenance remain in the original records.

All eight complete 96x3 force arrays pass the retained independent dense HF96
oracle gates. Worst errors are 3.17376e-13 Eh/atom, 4.88026e-8 Fmax and
2.52591e-8 Frms. All four strict off/on comparisons pass1e-9 Eh/atom,
1e-7 Fmax and3e-8 Frms; maxima are6.63173e-14,3.84814e-13 and9.75856e-14.
Original CPU-spot/reference reuse qualification is retained. No new reference or
96-atom CPU oracle was run, and no old reference latency is used as a denominator.

The original pilot status remains `INTEGRATION_PILOT_CORRECTNESS_PASS_PENDING_TIMING_DECISION`.
The separate completed independent review accepts this scoped default-off
publication. Neither record is rewritten to claim default promotion, current
publication-tree device measurement, the original all30 acceptance or closure of
the historical approximately49% HF gap.

The reviewed 17 source/policy/compiler/identity/test/note paths are integrated onto
main `17babea1beeced50a11254684b879cbf26157d72`, preserving the merged WB #3028
producer and other-owned SCF/VV10 work. The publication policy entry adds this
new evidence without promoting the control. This integrated tree has not been
measured on a GPU; the timings remain bound to `a694`, not the publication tree.

## Retained evidence and offline audit

`evidence.json.xz` losslessly retains all23 original pilot files,286045 bytes,
including both raw arrays/plans/qualifications and all16 stdout/stderr streams.
It also retains the original build/dispatch/source bindings, producer source,
completed review and exact original oracle capsule. Each member retains its
original whole-file size and SHA256. No numerical/adverse field is discarded.
Large binaries, generated shards, build/resource streams and runtime caches remain
in the original owner; hashes identify them, not recover their omitted bytes.

Run with standard Python3.10 or newer:

```sh
python benchmarks/results/h100-hf-shared-xy-opt-in-20261107/replay.py
```

The reader authenticates the capsule and provenance chain, recomputes all eight
dense and four strict full-array comparisons, cold saving, all three warm guards,
work/control equality and scope. It executes no native library, CUDA, SCF or
oracle and is not a device reproduction. The exact older
[synthetic/sanitizer qualification](../h100-shared-rys-xy-phase-a-20261106/README.md)
remains source-bound to b56. Its original missing-tool failure is retained there.
The later failed native capture and separate authenticated CPU recovery/actual
queue replay remain in their bound owner histories; this pilot does not relabel
those earlier sources or add terminal-gather work.
