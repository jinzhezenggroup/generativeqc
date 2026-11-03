# Decision: separate budgeted DF96 admission from Direct-HF's small route

Status: implemented
Date: 2026-11-03

## Problem

The prior DF96 probe stopped in the Direct-HF inventory's <=16-AO eigensolver
domain before reaching the DF owner. DF does not allocate that Direct-HF plan:
it owns ordinary AO and metric eigensolvers, generated integral sources,
SCF/DIIS state and bounded analytic-response staging. Requiring the unrelated
owner first prevents auditing the actual large DF execution route.

This supersedes only the DF admission boundary recorded in
`2026-11-03-resident-dft96-qualification.md`; that note's historical refusals and
DFT evidence remain unchanged. No composed KS/DF admission follows.

## Decision

- Dispatch fitted requests to their authoritative native DF inventory before
  querying any Direct-HF layout. Direct-HF retains its existing limits.
- Extend the DF owner only to explicitly budgeted RHF sources with <=768
  orbital and <=768 auxiliary AOs. Large UHF, unbudgeted large resident requests
  and larger dimensions remain explicitly unsupported. Small requests keep
  their existing domain. Native refusal does not permit compatibility fallback.
- Reuse the existing overflow-checked tile/source/DIIS leaves and whole-fleet
  lifetime composition. Do not delete the allocation gate, increase an actual
  allocation limit, weaken a metric cutoff or substitute an iteration estimate
  for measured Fock work.
- Ordinary AO and metric provider queries are checked against their existing
  `1 MiB + 128*n*n` per-workspace allowance before allocating. The inventory
  includes serialized AO state, retained metric/SCF factors, detached singleton
  retry and separate force staging. Explicit packed storage still has to fit
  its full retained factors; dimensional eligibility is not budget feasibility.
- Preserve conservative host envelopes, including their historical quartet-size
  metadata overestimate. This change does not qualify a practical finite host
  budget or claim host/RSS reduction. Tightening that independent owner requires
  a separate complete allocation audit, not deleting terms to obtain admission.

## Qualification protocol

`benchmarks/df96_autotune.py` compares complete energy/analytic-force replays
between isolated resident owners, after independently converging and freezing
their own native densities. Five alternating-order pairs, both allocation
ledgers, exact native/source/library identity, full auxiliary rank, measured
complete Fock counts, strict independent E/F and a fresh final holdout are
required. Faster samples cannot override a failed gate. The benchmark does not
install a profile or execution recipe; explicit user overrides remain explicit.

The independent CPU fitted-HF oracle uses the immutable Gaussian input tables
with libcint integrals, full Fock rebuilds, `1e-16` screening and analytic
auxiliary-basis response. Its recomputed orbital-gradient norm is
`2.2137677305011807e-11`. The 768-dimensional metric is independently full rank:
minimum eigenvalue `0.09304817858971107`, cutoff `2.3994330289515964e-7`.
No oracle density, orbitals or integrals enter the native timed path.

## Rejected alternatives

- Removing the small Direct-HF limit globally would admit an unaudited exact
  quartet route, not repair DF ownership.
- Merely deleting the DF size guard would also admit unbudgeted resident and
  large UHF routes, with no corresponding endpoint qualification.
- The historical 512-MiB hint is insufficient for the large phase planners once
  fixed source/metric/DIIS state and the half-sized force allowance are charged.
  Keep that failure explicit rather than silently increasing a user's hint.
- Installing a resident benchmark's winner into a general local-profile bundle
  would conflate qualification protocols and override unrelated workloads.

## Evidence

Native resource tests cover 768-AO dense/generated/packed admission, separate
energy/force bounds, ordinary provider reservation and unsupported-domain
rejections. CPU tests exercise the real scalar-only native request without
constructing the unrelated Direct-HF inventory, plus the benchmark's strict
DF gates and alternating protocol. Real-device evidence is retained separately
from the earlier DFT experiment; its scientific and source identities must not
be combined with that campaign.

## Measured outcome

See `benchmarks/results/df96-local-autotune-20261103/README.md` for source-bound
receipts. At the fixed 4-GiB hint, seven complete five-pair comparisons retain
70 endpoint observations. None wins the >=2% promotion gate; default complete
warm E/F is approximately 5.23 s. All completed comparisons pass strict E/F,
rank, actual four-Fock census, both resource-owner and identity gates. Keep
defaults rather than installing an unproven policy.

Explicit packed storage is infeasible under the half-sized force allowance.
Serial derivative startup/warmup is retired without a timed pair, not labelled
a numerical failure or qualified timing. The old resident helper masks dead
worker errors with a pipe-close exception; preserve that historical log, fix
cleanup without changing native scientific code, and verify the corrected
refusal separately. Both executed and maintained controller variants are retained.

## Remaining consumers

The Python-free native CLI has its own earlier <=128-auxiliary-AO and
Direct-HF-small-layout preflight in `src/cli/resources.cpp`. It still refuses
large fitted runs even though its downstream DF fleet helper shares the C++
owner. This change qualifies Calculator/the Python tuner only; do not treat
the common scalar owner's larger domain as native-CLI endpoint evidence or
delete those independent frontend guards without qualifying that consumer.

## Revisit when

Independent endpoint and allocation evidence exists for larger auxiliary
bases, finite practical host bounds, large UHF with complete native Fock
counters, or a safely composable KS/DF owner.
