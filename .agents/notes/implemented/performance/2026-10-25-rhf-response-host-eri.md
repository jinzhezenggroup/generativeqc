# Decision: Bound RAM-backed exact RHF frame actions by complete response capacity

Status: implemented
Date: 2026-10-25

## Problem and decision

The eager exact orbital/Z owner repeatedly contracts the same geometry-only
four-center source. Complete GPU residency previously refused when its full
inventory exceeded the allowance. Preserve its existing relaxed, unscreened,
at-least-64-AO automatic eligibility and use the shared strict FP64 host provider
for omitted buckets after fitting whole GPU buckets. Explicit positive GPU
ceilings continue to qualify smaller validation frames. Complete GPU residency
keeps priority and requires no RAM staging. This extends the full-range provider
from PR #2762; it does not change integral equations or source precision.

Admission observes three independent bounds: available job RAM, complete response
numeric allowance, and free device storage after reserving the mandatory numeric
inventory plus 256 MiB. Reserving the entire mandatory inventory on the device is
conservative because some components live on the host. Source setup may therefore
refuse reuse even when a more exact liveness model could admit it.

The default host ceiling is min(32 GiB, one quarter of machine and enclosing
cgroup headroom). Two pinned and two device tiles are charged together with the
values and host owner. Their configured size is only a ceiling: reserve at most
half of either host or combined available capacity for staging. Otherwise small
complete budgets can spend their entire allowance on staging and cache no ERIs.
GPU bucket metadata is reserved before fitting can introduce a partial mask.

## Invariants and boundaries

- The source owns exact geometry, basis, radial operator, stream and device.
  No global dimension-only cache or borrowed pageable pointer is introduced.
- GPU values retain priority. The current provider needs an admitted GPU source;
  no GPU bucket fitting means bounded Direct, not an unqualified host-only path.
- Independent final scalar residual and nuclear derivatives regenerate their
  physical sources. Positive-screening response actions do not use this RAM tier.
- Optional host allocation refusal preserves GPU values. A later required OOM
  unwinds the entire owner before one exact retry without optional values.
  Pending non-allocation CUDA faults and numerical failures propagate.
- Report complete host+pinned+device capacity and actual replay/H2D separately.
  The conservative discarded-attempt capacity includes host preparation. Existing
  endpoint reports suppress successful-attempt work fields after a retry.
- This eager owner does not authorize RHF-SCF graph staging, public prepared CPKS,
  density-dependent projected-factor reuse, or temporary disk for full-range ERIs.
  Earlier negative full-range KS disk evidence precludes borrowing the positive
  LR disk conclusion for these response actions.

## Validation

The same-source Slurm suite passes 164 tests on an RTX PRO 6000, including
signed nonzero Z, Cartesian and spherical d/f sources, moved geometries,
independent two-step energy finite differences, small-frame refusal, a naturally
eligible 70-AO frame with limited GPU capacity, and a full-GPU control. Executable
host resource doubles additionally exercise the production admission and retry
controller for host OOM/refusal, numerical failure, pending/asynchronous CUDA
execution faults, and a later mandatory allocation failure. Existing GPU retry
and cleanup checks remain intact. Six RAM/refusal cases pass CUDA memcheck with
zero errors. These checks establish correctness and ownership; complete endpoint
performance is qualified separately with the consuming DF-CCSD(T) force runner.
Two complete water CCSD/CCSD(T) cases explicitly require host replay and pass
independent total energies plus two-step directional force finite differences.

## Revisit when

Retain same-source complete endpoints and semantic work counts when changing the
size gate, memory reserve, bucket cost order or tile sizing. Test full GPU, partial
GPU, host refusal and changed geometry controls. A public CPKS or graph integration
requires its own source-lifetime and scheduling validation. Disk requires measured
reuse and I/O profitability on that consumer, not just insufficient RAM.
