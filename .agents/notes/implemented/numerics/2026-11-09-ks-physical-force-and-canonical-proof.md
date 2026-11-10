# KS force stationarity and pure-UKS canonical publication

Status: implemented
Date: 2026-11-09

## Problem

Issue #2840 retains a Cartesian PBE UKS water-dimer cation whose ordinary SCF
gates pass before four unextrapolated closure corrections diverge. A distinct
spherical observation fails canonical density reconstruction at
`1.4803757414361129e-8` against `1e-8`. A small physical commutator alone does
not certify the unshifted canonical density, and the two observations must not
be conflated. In the matched current-source baseline, both Cartesian consumers
terminate at iteration 35 and the spherical E+F consumer fails at iteration 85.
The spherical E-only baseline succeeds at iteration 107; it must not be counted
as another baseline failure.

Issue #2915 separately retains original-default, original-grid water96 PBE
DF-RKS moved/fresh force differences near `4.794e-7 Eh/Bohr`, above the unchanged
`1e-7` gate. The small-grid/tight-SCF ownership fixture is not evidence about
that physical task or about causality in historical PR #2910.

The prepared KS options deliberately disable direct force evaluation during
SCF. Consequently, `options.compute_forces` alone cannot distinguish a public
energy-only request from an imminent native force consumer. Additional memory
bytes cannot encode that distinction either: retained force storage can make
the incremental allocation zero.

## Decision

- Bind the actual derivative-consumer flag before the solve, separately from
  the existing additional-allocation hint. Clear it on energy-only replays.
- Keep the full-range force/reference physical residual ceiling at
  `min(1e-9, density_tolerance)`. Preserve public energy/density controls and the
  independently qualified nonlocal orbital-gradient and mixed budget policies.
- Use the same allowance in iteration, resident weights and detached export.
  Do not accidentally tighten the independent canonical density gate from
  `1e-8` to `1e-9` by passing the physical ceiling as a blanket validator
  override.
- Extend the existing actual-F[D] canonical proof from exchange-bearing UKS to
  all otherwise eligible UKS. Neither exact exchange nor its absence makes an
  unextrapolated fixed-point update safe. A successful complete proof permits
  publication without that update; rejection continues DIIS without publishing
  a token or warm seed. ECP, mixed precision, range/nonlocal and resource guards
  remain unchanged.

## Rejected alternatives

Increasing iteration limits does not address a deliberately four-step closure
after ordinary convergence. Accepting only the physical residual would waive
the independent spherical reconstruction gate. Relaxing the force gate,
changing the grid, or silently tightening the caller's SCF controls changes the
original water96 task rather than repairing its consumer contract. Applying
this full-range decision indiscriminately to WB97M-V would overwrite its
separately qualified orbital-gradient semantics.

## Evidence and boundaries

The opt-in CUDA regressions preserve the retained large-s/p/d geometry,
Cartesian/spherical representation, grid, Q/M, energy/density/screening controls
and iteration limit. They test E-only/E+F, masks 0/127, cold/warm/moved states,
canonical density and independent PySCF E/D/Fock/full-moving-grid forces.
Python scientific force fallbacks are forbidden in those native endpoints.

On Slurm job 16985 (node1, RTX 5090), all eight candidate controls passed:
40 final states, including 20 force states. Maximum independent force error was
`1.2819691391774413e-8 Eh/Bohr`; maximum canonical reconstruction error was
`9.26770282738687e-9`, below the unchanged `1e-8` gate. The measured native
source identity is
`254e1db0bcf1401bca2e350a56132334cdd9d58f2a909f309aa9902cf32c0d75`.
This was a Release/sm120 `CUDA_FAST_COMPILE=ON` diagnostic build, not a
performance or production-promotion qualification. The selected host tests
also passed (190 tests).

The water96 test retains the original seed import rejection, atomic rollback,
frozen replay and displacement sequence. Its original-grid/default-SCF arm is
separate from both the tighter-SCF-only arm and the existing small-grid
ownership fixture. Its receipt writer retains full force vectors, physical
histories, loaded-library hashes, source identities and scheduler visibility
when an endpoint completes. No original-grid water96 force endpoint completed
in this campaign: job 16986 timed out its default baseline arm after 300 seconds
during cold SCF, and the tighter baseline arm was cancelled before completion.
Corrected candidate arms were not executed to an endpoint. The observed
approximately 40-second RI-J / 43-second complete-Fock iterations explain why
the bounded attempt was retired; they are not a baseline/candidate performance
comparison or a numerical failure at the force gate. Issue #2915 remains
unqualified, including historical causality and tighter-SCF-only isolation.

New measurements do not relabel the historical #2287 failures or the #2910
binaries. A current-source repair does not establish whether #2910 caused its
historical force discrepancy. Device qualification is scoped to the actual
recorded devices; it does not transfer to an unmeasured A100 or PRO6000.

The existing exact native `--df-range-exchange` control separately checks the
original DF-primary/Direct-LR RKS/UKS composition and independent CPU E/state.
That control alone is not an independent force receipt for every RSH consumer.

The reviewed publication in
`benchmarks/results/ks-original-state-gates-20261109/` retains source/library
identities, unchanged gates, all UKS final diagnostics and full force vectors,
all four baseline observations, and compact water96 timeout/progress provenance.
Complete per-iteration histories remain in the ignored original receipts with
byte identities/history hashes retained in the publication. Its overall decision
is inconclusive; it does not close either issue.

## Revisit when

Revisit the guards only with source-matched complete endpoints for the proposed
new precision, ECP, nonlocal or resource domain. Preserve every independent
publication gate and bounded rejection path.

## Review correction: early-probe AO maximum

The restricted early-probe path still admitted an AO maximum below `1e-8`
after force/reference consumers stopped passing a blanket stationarity override
to the final validator. Thus RMS `1e-10` and AO maximum `5e-9` could bypass the
new force ceiling even though ordinary convergence rejected them. The early
physical admission now uses the same force/reference ceiling, while leaving
the separate canonical-density tolerance and the energy-only early path intact.
An executable host policy regression covers that counterexample, the exact
boundary, tighter density controls and nonfinite residuals.

The retained RTX 5090 measurements above precede this narrow admission fix;
no new GPU, original-grid water96 or independent RSH force qualification is
claimed. The retained Libxc catalog is refreshed only to report the live source
identities; all historical measurements, existing stale status and zero current
energy/force qualification remain unchanged.
