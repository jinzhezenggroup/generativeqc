# Decision: bound large Direct cold work and complete force capacity

Status: implemented
Date: 2026-10-25

## Problem and decision

The 96-atom r2SCAN-3c energy-plus-force endpoint spent time in repeated full J
builds, divergent public-AO Schwarz preparation, small XC batches, and a force
tile constrained by the incumbent host allowance. Four independent bounded
changes address that work without changing the scientific endpoint:

- All-electron Direct forces at 96 or more atoms request 512 MiB host and 1 GiB
  additional device capacity. With both allowances the complete dry planner
  prefers 1024 points, retaining its full capacity search. Fleet estimates use
  the same per-item ECP exclusions and identity-bound limits.
- Public Schwarz preparation groups AO lanes by angular momentum, primitive
  count and sparse component count. Every directed (ij|ij) retains its original
  evaluator and accumulation order; no mirrored shortcut is introduced.
  Preparation borrows B*N int32 slots in the not-yet-initialized density arena,
  drains the owning stream before releasing host staging, and charges the host
  permutation. Active public ledgers and host allocation refusal keep ordinary
  ordering. GENERATIVEQC_DIRECT_ORDERED_SCHWARZ=0 is the rollback.
- Pure-J PBE/r2SCAN RKS cold solves, including the electronic part of r2SCAN-3c,
  admit the existing bounded incremental proposal policy at 512–1536 AOs and
  48–128 atoms. Default FP64, through-d all-electron basis, grid, stopping,
  exact-provider and resource gates remain mandatory. Warm/imported/moved
  seeds, DF, mixed, range/nonlocal and explicit selectors retain their previous
  policy. Proposal limits and the final full-density energy pair are unchanged;
  neither an approximate Fock nor a cached force certifies acceptance.
- Large meta-GGA layouts use up to 64 point tiles under 1 GiB optional capacity.
  Explicit budgets, active resource ledgers, and allocation failures retain
  their existing fallback. No additional grid points or precision changes enter.

The existing automatic hybrid size domain and refresh policy are unchanged.
Pure-J uses the existing bounded proposals, with its ordinary fixed-anchor
controller retained when proposals are disabled. Complete endpoint receipts
now expose anchor/delta/closure work alongside total SCF/Fock counts.

## Evidence and limits

The force-only change was measured in three same-card rotated fresh-process
rounds: cold about 36.86 to 34.87 seconds and warm about 15.4 to 14.07 seconds.
Original/displaced/fresh-displaced independent energy/force gates pass.
The combined candidate passed 24 complete E/F checks at 96 and 120 atoms across
Direct r2SCAN-3c, DF r2SCAN-3c and DF r2SCAN, as well as 16 default/rollback,
seed-lifetime and independent-oracle GPU regressions. Public Schwarz tests
cover Cartesian/spherical s/p/d/f bases, batches, exact capacity and nonfinite
failure. The new DF omissions and Schwarz preparation pass CUDA memcheck.

The initial combined clean Direct96 observation was 33.366 s cold and 13.94 s
warm versus GPU4PySCF 32.383 / 14.81 s. It therefore did not establish cold
parity. Final README publication uses a separate full #2786 default-only
campaign, not these exploratory observations.

## Rejected and unfinished experiments

Force tile 2048 and XC tiles 1024–4096 regressed. Smaller force page widths,
class-major dddd scheduling and exact-zero Becke seed elision did not help the
complete endpoint. Resident AO bitmasks gave no net improvement for this Direct
case; a conservative envelope helped cold slightly but regressed warm. The
existing DF producer crossover is not extrapolated to Direct.

One-electron thread mapping, alternative Schwarz block widths and a looser
provisional pure-J cutoff remained diagnostic trials. They were not promoted;
remaining trials were stopped when the user requested the README refresh.
Preserve the unchanged final-state gates and reconsider only with complete
cold/warm/moved evidence, failure/resource coverage and a clear net benefit.
