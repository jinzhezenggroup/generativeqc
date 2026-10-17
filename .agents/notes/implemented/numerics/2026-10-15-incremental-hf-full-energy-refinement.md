# Decision: retire screened HF deltas at physical stationarity

Status: implemented; experimental opt-in, not default promotion
Date: 2026-10-15

## Problem

The [full-only warm repair](../performance/2026-10-15-incremental-full-only-warm-finalization.md)
removes redundant final builds when no delta was executed. It does not prevent
screened incremental HF energies from changing between full and reconstructed
operators near a stationary density. Retained n4 Slurm diagnostic job 997
reproduces a 384-AO moved-geometry RHF failure: 100 returned iterations and a
discarded 100-iteration attempt, with no forces. At the last returned step,
density RMS is 7.831e-11 and the pre-DIIS physical maximum is 2.342e-10, both
below 1e-9, but iteration |delta E| is 5.862e-9 Eh against the existing
roundoff-guarded 5.319e-12-Eh energy limit. Independent endpoint energy error
is a different quantity and cannot turn this into successful convergence.

The first attempt spends iterations 14 through 100 failing only energy; its
retry does so from iteration 19 through 100. Successful diagnostic reruns do
not erase this intermittent failure. Waiting for a screened energy gate before
allowing an ordinary full-density check can therefore waste the complete
iteration/retry budget. KS already has a separate full-energy refinement policy.

## Decision

For screened CUDA RHF/UHF only, after actual delta work and once the current
density-step and pre-DIIS physical-maximum gates pass, latch that system into
ordinary full-density builds for the remainder of its solve. Enter this phase
even if the reconstructed energy happens to pass its gate: an accidental
cross-operator agreement must not substitute for full-energy confirmation.

Entering refinement does not mark convergence. Discard the reconstructed
energy baseline and keep the accepted density proposal and DIIS history.
The first full evaluation establishes a fresh energy baseline; only a later
full evaluation can pass the unchanged energy, density and physical criteria.
The controller remains full-only if stationarity subsequently worsens. There
is no return to deltas until a new solve resets all per-item update state.

Reserve a second sentinel in the existing update-counter buffer, separate from
the unanchored first-build sentinel. Preparation reads that immutable per-item
state across matrix CTAs; finalization preserves it. Convergence owns the
transition after all matrix work. This needs no new device allocation, host
polling, reference/CPU work, or extra graph launch. It is per physical system,
including both UHF spins, not a bucket-wide penalty on unrelated neighbors.

Full provider outputs are already physical F[D]. Finalization preserves them
bitwise rather than needlessly adding and subtracting hcore, which can erase a
representable value and inject additional tail noise. Only actual delta results
are reconstructed against an anchor.

## Invariants and bounds

- Keep the original max-iteration bound. Full refinement is charged inside it,
  not hidden as additional work after exhausting 100 delta iterations. A full
  solver that still fails returns honest nonconvergence, not a manufactured
  successful energy/force endpoint.
- Keep all tolerances, finite checks, physical residual checks and independent
  energy/force acceptance gates. A small density step alone cannot trigger or
  certify this path; approximate/coarse items must first reach target precision.
- Keep actual full/delta work counts. The legacy `periodic_rebuilds` field
  continues to mean every full refresh after the first anchor, now also
  including full-energy refinement, rather than cadence-only work.
- Leave ordinary and unscreened incremental providers unchanged. A resident
  no-delta warm solve cannot enter refinement and retains its one-full-SCF,
  zero-final-full route. Every new solve resets the sentinel, including warm
  replay, external seeds and changed geometry.
- Preserve strict canonical finalization after any actual delta, including
  full refinement. Reusing those final full Focks would require a separate
  qualified determinant/provenance policy; this repair does not remove them.
- Keep incremental J/K off by default and keep the separate KS policy intact.

## Rejected alternatives

- Widening energy tolerance or declaring success from density/physical gates
  would evade the requested scientific endpoint rather than repair it.
- Periodically refreshing the anchor alone still compares different screened
  energy operators; the phase must remain full-only to finish energy closure.
- Waiting for 100 failed delta iterations and then another ordinary solve
  wastes avoidable work and complicates complete retry accounting. The observed
  failure already has a stationary physical tail, so retire deltas there.
- Sharing a writable mode flag across matrix CTAs would mix classification and
  state mutation in one launch. Reusing the read-only update state during
  preparation keeps scalar ownership unambiguous.

## Evidence

Host tests execute the real controller bodies with fixed, rolling and sticky
full-only states at intervals 0/1/3/8 for both spins, preserving inactive items,
screened reconstruction/error depth, reset admission and work counts. The
focused host/controller/resource suite passes 103 tests.

Native convergence coverage exercises the actual multi-CTA preparation,
finalization and RHF/UHF acceptance kernels: energy-only rejection, an
accidentally passing delta energy, no-delta warm neighbors, nonstationary,
coarse, nonfinite and inactive items, both density-retention modes, two fresh
full energies, sticky interval-zero behavior and bitwise full-provider output.
Live independent-reference and GPU/sanitizer results are retained with this
change's source-matched validation artifacts.

### Qualified source and acceptance

The final combined host suite passes **288 tests**, including controller,
resource, KS-policy, SCF structure and CUDA ownership coverage. Source identity
is `c7a15685cd6709d078e3de239c6190261384cc38b9526563af222f1977fac108`;
library SHA256 is
`a91856f07320a0eb7c1350591a32a819715960f6be758af7c907c26663fbfeb4`.
The source snapshot is based on master
`4e28477e96d257528a60bcb68c00529f22585bc1`. It is a Release CUDA 12.9.1,
sm120, scalar-CPU-linalg, generic/non-AOT diagnostic build, not a production
profile or default-promotion qualification.

The final review host selection passes 328 tests (including extra ownership
pairing/publication checks). The library and qualified native targets rebuild
with no work and the identical library hash. A broader all-target build is
blocked by unchanged `tests/native/test_ks_resident_weights.cpp` calling the
absent `PreparedBatch::ks_transport_diagnostic`; this repair does not claim
all-target build acceptance or modify that unrelated baseline mismatch.

Finite Slurm n1/RTX 5090 jobs, with assigned device visibility preserved:

- **8810:** native RHF/UHF intervals 0/1/8, resident/external/moved/forced
  finalization, actual convergence-controller kernels, KS physical-state and
  fresh-solve checks, and live independent CPU PySCF water/OH energy/force
  comparisons pass.
- **8812:** convergence/controller memcheck and synccheck, plus complete
  incremental HF memcheck, all report zero errors. This does **not** discharge
  the previously retained full-HF synccheck failure in the unchanged initial
  density-bounds kernel. Complete HF synccheck acceptance remains withheld.
- **8811:** all 30 large-fixture endpoints pass, including six priming points
  and 24 formal points. The exact retained 384-AO/48-atom RHF/def2-SVP moved
  fixture now converges in **16 iterations**, consistently across priming and
  four independent formal owners: **4 full + 12 delta SCF builds, then 2 final
  full builds**. There are no retries, discarded iterations, or initial-guess
  fallbacks. This supersedes the earlier unfixed-tail status, not the historical
  job-997 rejection itself.
- **8814:** all 60 small-fixture endpoints pass, including 12 priming and
  48 formal points. Preflight failures are retained separately: n4 job 999
  was cancelled while pending; an initial n1 runner lacked the copied
  environment file; job 8813 lacked `base-sha.txt` before any samples.

All 90 endpoint samples, not only formal timings, pass independent CPU PySCF
gates of 1e-8 Eh and 1e-7 Eh/Bohr. Maximum absolute errors are
**1.319e-11 Eh / 9.330e-11 Eh/Bohr**. Requested SCF energy/density tolerances
remain 1e-12/1e-9, screening remains 1e-12, and the iteration budget remains
100. Every endpoint uses one target attempt with zero discarded target work.
Every fixed8 warm sample preserves **1 full + 0 delta + 0 final full**.

### Complete endpoint timings and limitations

Four formal repeats per arm, each with an independent owner and its own
cold-derived warm density, after a complete unmeasured priming sequence per
arm. Cold includes construction/preparation/initialization/SCF/finalization
and host energy/forces, excluding common global bootstrap; teardown is
separate. The 384-AO order is balanced AB/BA/AB/BA. The retained small harness
projects its three-arm permutations to **AB/BA/BA/BA**, not a balanced order;
its timing observations are correspondingly weaker evidence.

| AOs | Cold OFF to fixed8 | Warm OFF to fixed8 | Moved OFF to fixed8 |
| ---: | --- | --- | --- |
| 96 | 1.427140 to 1.382179 s (-3.15%) | 0.185145 to 0.183801 s (-0.73%) | 1.132513 to 1.078658 s (-4.76%) |
| 192 | 13.264286 to 12.628320 s (-4.79%) | 1.917659 to 1.945118 s (+1.43%) | 11.241270 to 9.695104 s (-13.75%) |
| 384 | 47.292056 to 44.416665 s (-6.08%) | 5.604425 to 5.601752 s (-0.05%) | 41.892395 to 32.360700 s (-22.75%) |

These are same-binary OFF/fixed8 observations after the correctness repair,
not a claim that full-energy refinement itself is a speed optimization or
that every workload improves. Warm is baseline parity, including a small
192-AO regression. Bounded-streaming quartet work at 192/384 AOs is unknown,
not zero. No larger UHF-cluster speedup or general SCF convergence guarantee
is established; ordinary full refinement can still honestly exhaust its bound.

The [compact evidence](../../../../benchmarks/results/incremental-hf-full-energy-refinement-20261015/README.md)
retains all scalar endpoint gates, work and timings, full independent reference
arrays, reference/source/library hashes, frozen harnesses, validation logs and
per-returned-force-array hashes. Repository evidence headroom cannot retain
every returned force vector: these and the complete raw packet remain only in
ignored local `.artifacts/delta-anchor/full-energy-refinement/`, not a permanent
remote archive guarantee. No evidence budget is raised, no old evidence is
deleted, and no external archive is published.

## References

- `src/scf/cuda/scf_incremental_policy.cuh`
- `src/scf/cuda/scf_convergence_policy.cuh`
- `src/scf/cuda/scf_matrix_kernels.cu`
- `src/scf/cuda_rhf.cpp`
- `tests/native/test_cuda_force_convergence.cpp`
- [KS full-energy refinement](2026-10-07-incremental-ks-full-energy-refinement.md)
- [Screened fixed full anchors](2026-10-15-incremental-fixed-full-anchor.md)
