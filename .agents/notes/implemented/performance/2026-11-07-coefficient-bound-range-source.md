# Decision: coefficient-bound resident full/LR exchange

Status: implemented
Date: 2026-11-07

## Problem

The retained A100 Direct96 WB97M-V case already admits every LR bucket:
1,873,221,716 FP64 values / 14,985,773,728 bytes. Its independent full-range
exchange still evaluates integrals on each density. Streaming Full/LR fusion
cannot remove that work while the complete LR lease owns replay. Disabling the
effective LR cache would measure a different and deliberately degraded baseline.

## Decision

Before the first consumer, replace a complete GPU LR source by
`LR + (c_full / c_LR) Full`, using the same plan's canonical rows. Its one K
contraction supplies the existing range-exchange slot; the ordinary full-K slot
is cleared. Unchanged Fock/energy assembly applies `c_LR` exactly once. This
eliminates a separately recomputed full K, not either scientific radial term.

Construction uses the existing compiler-owned full and LR producers, followed by
the shared semantic FP64 vector-accumulation provider. A temporary full source is
independently finite-audited and released before SCF. This is not a fused-radial
source-building recurrence.

## Invariants

- Strict restricted full Direct only; no DF, mixed, incremental, graph-chunk,
  application-ledger, host-cache or disk-cache extension is implied.
- Complete admitted LR residency, unchanged persistent source ceiling, and a
  separately bounded transient allocation alongside the complete consumer.
- Fixed full/range term identities, coefficients, omega, geometry and canonical
  rows. Raw LR/J borrowers cannot consume a coefficient-bound source.
- Capacity refusal before mutation retains ordinary LR replay. A failure after
  the first in-place write poisons the source and propagates numerical failure;
  it must never be presented as optional refusal.
- Independent raw component profiling and incumbent warm rebasing retain their
  original providers. Analytic derivatives do not borrow composite value data.
- Count actual K contractions, not the two logical radial terms represented by
  one contraction. Trace metadata also retains the radial-term multiplicity.
- No CPU/PySCF/oracle work enters the production endpoint.

Default enablement is limited to the existing automatic repeated cold range-
source domain, with the additional strict restricted complete-GPU/transient
guards. Explicit `GENERATIVEQC_KS_COMBINED_RANGE_SOURCE=0` before preparing a new
owner restores raw LR replay. Explicit `1` never bypasses resource/identity gates.

## Pilot evidence

Prototype native identity: `d2172162b5a713a1d301ce4d0aefa100b1325a5d64f2ef3210f8cb17d2f9398c`.
SM80/SM120 library SHA256: `68912cf573d10259a01c0b6c975833393925673742b43b5157a90e92f1742869`.

Slurm16623 passes independent raw Full/LR matrix gates at two geometries/sizes,
signed coefficients, zero/nonzero screening, immutable identity and insufficient
transient-capacity checks. Compute Sanitizer memcheck reports zero errors.
Complete constructor+prepare+SCF+analytic-force+host-export Direct96 WB97M-V:
5090 cold 179.062530 -> 179.437876 s; own-density warm 13.704064 -> 13.695091 s.
Both arms retain 16 cold iterations/Focks and 32 K actions: the composite route
is not admitted on this resource envelope. Energy/atom difference is zero;
maximum force discrepancy is below 8e-13 Hartree/Bohr. This is fallback/no-
regression evidence, not evidence of composite-path acceleration on 5090.

A100 LSF226270 is DONE/exit0. Its independently gated complete cold improves
96.271742 -> 89.834377 s (6.69%, saving 6.437364 s), and own-density first warm
14.961188 -> 13.908842 s (7.03%). K contractions decrease 32 -> 16 cold and 2 -> 1
warm, with unchanged 16/1 Focks, 11 provisional + 5 target cold Focks, four cold
VV10 sweeps and zero new warm sweeps. Energy/atom difference is 2.84e-14 Ha;
maximum paired force difference is 1.14e-12 Ha/Bohr. Both images also pass the
retained independent full E/F numerical oracle, not a new latency comparison.

Slurm16627 additionally passes both raw and combined f-shell fixtures. Slurm16625
skipped due to missing compiler/cache environment and is not a passed gate.

The final default candidate also retires a failed mutated lease, validates spin
on repeated preparation and routes BLAS through the semantic Tensor owner. Source
5573279078e2 was built but never GPU qualified after the vendor-boundary gate
rejected its method-local BLAS calls; it is not a performance measurement.
Final native source is b9c0e6d609dfae6ef2dc9721e896c13e19a4d239b3d029ae264de4c59f67e83b.
Final-source GPU evidence is retained separately from this pilot. Slurm16630
failed host authentication because its source archive had not been extracted;
no oracle or endpoint ran in that allocation.

## Final default qualification

Native source b9c0e6d609dfae6ef2dc9721e896c13e19a4d239b3d029ae264de4c59f67e83b,
ELF SHA256 2b867f8aba56371f155af043c3b34bdec7b75c73321ba1dfd44ef52ea4741fc9:

| Allocation | Raw LR cold | Default cold | Raw LR warm | Default warm |
| --- | ---: | ---: | ---: | ---: |
| A100 / LSF226273 | 96.189766 | 89.776354 | 15.019811 | 13.879518 |
| RTX5090 / Slurm16631 | 178.361599 | 178.619370 | 13.620170 | 13.624872 |

A100 saves 6.413413 s / 6.67% complete cold and 7.59% first warm. RTX5090 changes
by +0.14% cold / +0.03% warm with its incumbent resource fallback: no meaningful
regression or composite-path acceleration is claimed there. Both retain 16 cold
iterations/Focks, one warm Fock, and unchanged VV10 work. A100 K actions halve;
RTX5090 actions do not change. Four final independent raw/composite fixtures
including f shells pass with zero skips; small-case memcheck reports zero errors.
A100 separately passes the 96-AO signed composite source oracle.
The latest master sync to 17babea1b adds benchmark documents only and preserves
this exact native source identity; it does not require relabeling earlier timings.

The compressed benchmark publication retains complete E/F, build/source receipts,
native reconstruction patches, semantic trace values, independent oracle gates
and superseded results. Its offline reader also checks the pre-existing
independent numerical endpoint oracle, never reusing its old reference latency
for a new ranking. Corruption tests keep acceptance active under Python `-O`.

## Revisit when

Extend only if complete cold improves without shifting cost into warm,
changing convergence/precision/grids, or weakening the independent numerical
gates. A streaming/bucketwise construction would need its own transient bound,
post-mutation failure policy and complete endpoint evidence.

## References

- Tracking issue #2976, checkpoint comment6699812138.
- Qualified pilot checkpoint comment6700363351.
- `tests/python/test_range_resident_values_cuda.py` and its native probe.
- `benchmarks/results/combined-range-source-20261107/replay.py`.
- Ignored evidence: `.artifacts/cold-weighted-rsh-20261107/`.
