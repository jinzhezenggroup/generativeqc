# Decision: reject the output-owned DF generation/whitening pipeline

Status: rejected; no production/default performance change
Date: 2026-11-09

## Problem and mechanism

The qualified PBE0 DF96 cohort retains a cold gap of0.10745377093553543s:
native8.43015567958355s versus fresh GPU4PySCF8.322701908648014s. A separate
intrusive profile locates approximately1.039s in resident DF materialization:
15 raw panels and15 in-place Cholesky solves. A two-stream host-only proposal
generates each panel directly into its disjoint retained factor interval, orders
the incumbent plan/cuBLAS-stream TRSM with bounded events, and joins both streams
before return. This removes an8,542,322,688-byte staging copy without increasing
numeric scratch/cache or changing mathematical work, factor gauge or panel sizes.
Unsupported layouts retain serial, and optional resource fallback precedes work.

The single pilot's gate was declared before GPU observation: require independent
dense E/F, strict native solver/precision/semantic work, complete unrelated kernel
count parity,15 raw+15 solve panels and materialization-span mean savings above
max(168.603113591671ms, twice summed absolute MADs). This is a mechanism screen,
not complete cold/warm endpoint or default qualification. No priority/tile/device
retiming rescue or evidence-budget increase is permitted after failure.

## Loader failure and correction

Original A100 LSF230607 completed native control but its unchanged host clone
crashed at cleanup; the actual pipeline never executed. Its core mapped two full
native libraries. CPU-only230618 reproduced the overlay's SONAME dependency
resolving through inherited executable DT_RPATH to the old GGA image rather than
the intended qualified image. Preserve the original failure/intent. This proves
contamination, not an isolated native crash stack or cause of every old failure.

CPU Slurm16956 compiled the same source against all767 authenticated baseline
headers, with verified sccache0.16.0; both raw DSOs remained byte-identical to the
earlier compilation. Therefore the examined newer headers did not alter these
particular DSOs. Both dependencies were then bound to the absolute intended
native ELF. CPU-only230632 proved each loader dependency list contains exactly
that one native owner without running constructors or science. The corrected
pilot additionally checks real process device/inode identities at construction
and retention, refusing mixed full-native owners.

Corrected clone SHA256:
`da6f34c584422d95ea8d6f83af9b19fb554d44fc5d58dc5648bd868740095acb`.
Corrected pipeline SHA256:
`ee8d24c6daf955e5642cc2435e726f1c7aabb9d8ea833ddd8c94ae763ba7ed51`.
The mechanism source is unchanged, not a new tile or hardware-selected variant.

## Corrected A100 result

Finite exclusive LSF230633 exits0. Native and unchanged-clone preflight passes
before candidate execution. All five cold-only intrusive profiles pass retained
dense independent E/F, strict native solver/precision/work, panel counters,
exact complete unrelated kernel counts and single-image ownership. Collector
loss/errors are zero; all input/header/image/device receipts authenticate.
The first candidate has exactly equal energy and force maximum difference
1.4832579608992091e-13 against native control.

| Inclusive materialization event span, ms | First | Second | Mean |
|---|---:|---:|---:|
| Native control |1031.30371|1034.54907|1032.92639|
| Pipeline |1035.96448|1038.74255|1037.353515|

Mean saving is **-4.427125ms**, against the unchanged168.603113591671ms floor;
twice summed absolute MADs is6.02343ms. The unchanged-clone span is1029.09167ms.
The original qualification gate fails. Do not run clean endpoints, physical5090
qualification, another priority/tile sweep or promote this exact pipeline.

Inclusive TRSM-region sums rise from418.92278391/420.76854558ms in native arms to
942.38791421/950.33301846ms in pipeline arms. These intervals include competing
stream work/waits; they are not exclusive kernel time and do not prove a specific
hardware bottleneck. Removing a copy and adding streams did not reduce elapsed
materialization. Intrusive whole-cold times are preserved but not a clean gain.

## Evidence, scope and consequences

The complete corrected archive is1,798,439bytes, SHA256
`1fa9d9fe2c124322a7a17c73146a0e10bd5c64b42397e9593e3e709f09970ffd`.
Original failed archive is1,210,351bytes, SHA256
`344ba9faffa615ddcdca5ed6113d7885612cd2f0a0c3fae20e636a552ec48153`.
Ignored `.artifacts/dft-resident-pipeline-20261109/` and its separate
`dft-resident-pipeline-loader-fixed-20261109/` owner retain original payloads,
all full E/F arrays, collectors, headers/build/cache/dependency receipts,
unmodified gates and authenticated rejection analysis. No result is replaced.

The native base is sourceb11d2832 / ELFa0da0cdb, master76bb87c76 plus PR3056,
not a relabeled current-master production build. Current fetched master2a859ebb9
already contains merged PR3056. Its qualified cold savings remain unchanged.
The only retained code change is the shared README benchmark's single-native-owner
integrity guard; it is not a performance improvement or solver default change.
No production CPU science, SM/GPU-name selector, opt-in, enlarged cap, Release,
asset or tag is introduced. Full non-CBS leadership remains open, DFT first.

## References

- #2976 comments6734659292,6735476715 and6735666273 (original plan, failure,
  corrected predeclared protocol), followed by the result closeout.
- `../implemented/compatibility/2026-11-09-single-native-benchmark-owner.md`.
