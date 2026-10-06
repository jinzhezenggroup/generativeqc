# Pre-AO native CSR force portfolio qualification

This bundle qualifies the new **force-domain default dispatch** on one frozen
merged candidate source/binary. `public/` contains five interleaved AB/BA
campaigns for 3, 24 and 48 atoms, independent references and separate intrusive
profiles. The 120 measured complete E+F calls and 24 intrusive calls remain
separate; excluded prewarms are not timing samples. All actual Fock histories
and AO/jet/projection/gather/transfer counters are retained.

## Control and candidate

- `current-default` retains only the already enabled sampled profile, including
  its original continuous dense point×AO² crossover and 16 MiB map allowance.
  It runs on the **same merged candidate source and binary**, not a separately
  built unmodified master binary. This isolates the default-dispatch change.
- `auto` leaves the public portfolio unchanged: native conservative pre-AO CSR
  extends previously dense work below the existing crossover; the existing
  sampled route remains selected at or above it. Native occupancy above 0.8
  declines the whole inventory, never truncates AO labels or changes cutoff.
- Native SCF remains sampled in both cases. Its `point_ao_visits` is a
  one-traversal inventory; `xc_evaluations` is solve-local. Do not multiply or
  relabel those fields as whole-solve executed work without their proper scope.
- GPU products, SM-version windows, atom counts, AO counts and grid-count
  whitelists do not select this route. Admission uses general CUDA/semantic
  capability, resources, occupancy and continuous work cost.

The accepted objective is the user's stable warm/moved-warm improvement, with
all other phases retained and no material regression. The shared robust gate
is improvement greater than `max(2%, 2*(relative_MAD_A + relative_MAD_B))`.
Unchanged sampled and dense-occupancy paths need no regression, not a fabricated
new warm gain. At least one sparse extension must qualify.

At 24 atoms, warm complete E+F decreases from **3.970949 to 3.814492 s (3.94%)**;
moved-warm decreases from **4.009009 to 3.852154 s (3.91%)**. Both exceed the
2% robust-noise gate. Cold is 43.144010 → 42.920474 s and moved is
24.388917 → 24.194730 s: neither is a significant gain. AO visits/jet values
decrease 24.68%, projection FMA pairs 38.51%. Native label H2D, Python AO-label
lookups and discovery AO-jet values are zero; warm replay has no discovery
transfers. The final `qualification.json.gz` contains all per-phase medians,
robust spreads, reference errors, guard outcomes and per-round work reductions.

Only **n1 / Slurm main / RTX 5090** is measured. Hardware names are provenance,
not default eligibility or evidence of timing on other GPUs. Core SHA-256:
`cbc3fae7362c2417d6b2117429541a54d501e113c3116fa4279ed59186061a78`;
native scientific source identity:
`25e88a8a007bc1609c2bcdca5a4680182ca41ec9369abdeaba0e4b6b5de778a0`.

## Numerical and resource scope

The publication decision accepts independent numerical E+F gates, not a generic
all-phase performance promotion. `evidence.json.gz` therefore deliberately keeps
`performance.status = not-run`; the separately verified warm portfolio
eligibility in `qualification.json.gz` is not relabelled as significant cold
improvement. The same prebuilt core/AOT binary is used on both sides; uncached
deployment compilation delta is not measured. Numeric memory bounds and optional
map allowances are retained in each force census, not claimed as observed global
allocator peaks. GPU tests, profiler, memcheck and racecheck all use finite Slurm
allocations and preserve scheduler device visibility. `support.json.gz` retains
26 CUDA cases, zero-error memcheck/racecheck, explicit native analytic gates,
10 public analytic force cases, 970 passing host tests and compilation receipts.

## Historical negative comparisons

`mechanism/` and `mechanism-qualification.json.gz` retain the older frozen
48/96-atom A/B/C experiment at base
`a85459c13d16741ef5fab37cca6a64fc4a5bca15`. It is **not merged production/default
qualification**. Its sampled experiment uses 64 MiB, unlike the 16 MiB current
incumbent. Native improves warm over dense by 15.86% / 33.53%, but is **2.95% /
4.80% slower than sampled**. Both losses exceed their 2% robust-noise floors;
they are retained, not discarded. This is why the new portfolio does not replace
the high-work sampled default. Old native source/core identities and different
Slurm allocations are not relabelled as the merged candidate.

## Offline verification and source reconstruction

From the repository root with its Python source available:

```bash
PYTHONPATH=python:. python -m benchmarks.verify_preao_portfolio \
  benchmarks/results/preao-force-portfolio-20261006/public --atoms 3 24 48 \
  --output .artifacts/preao/reproduced-portfolio.json
PYTHONPATH=python:. python -m benchmarks.verify_preao_force \
  benchmarks/results/preao-force-portfolio-20261006/mechanism --atoms 48 96 \
  --output .artifacts/preao/reproduced-mechanism.json
```

Raw JSON and source patches use deterministic byte-preserving gzip, not rounded
or reduced records. Qualification summaries omit only duplicated raw/profile
payloads: every complete original remains in `public/` or `mechanism/`, and the
offline verifier regenerates its full output. Summary schemas explicitly identify
that distinction. `measured-dirty.patch.gz` applies to the full measured
revision in `publication.json`; `measured-full.patch.gz` reconstructs the same
snapshot from base `196126d88`. `mechanism-dirty.patch.gz` applies only to the
older base above. Decompress with `gzip -dc` before `git apply`. Exact base/full
revisions and uncompressed patch digests are in `support.json.gz`.
The publishing checkout additionally fixes host-only test fixtures and current
documentation; those changes do not alter measured scientific/endpoint sources.
Failed/partial/unpaired attempts stay in ignored working artifacts and are not
counted as successful samples.

P0-A and the force-owner portion of P0-B are implemented. Separate native KS/XC
inventory sharing, P0-D projected-panel aliasing and P0-E sparse provider
scheduling remain deferred. This does not close all of #1893 or revive the
stopped owner-grouping/point-parallel geometry experiments.
