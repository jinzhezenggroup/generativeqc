# Resident pre-AO force domains

The ordinary CUDA force route can build a conservative AO domain before AO
evaluation. Compiler-owned Gaussian region envelopes produce an ascending,
unique device CSR inventory. Each tile lends the existing `AoGridBlockLayout`
to the indexed AO, local density and projection consumers; AO labels do not
round-trip through Python or upload again on warm replay.

The automatic portfolio in `python/generativeqc/_force_active_ao.py` targets
CUDA, all-electron, non-density-fitted RKS/UKS, deriv=1 or deriv=2,
fixed 256-point tiles, and ordinary force budgets of at least 512 MiB device /
256 MiB host. Neither GPU products nor atom/AO/grid counts are dispatch whitelists.
Below the existing sampled route's dense point×AO² crossover of 173,946,175,488,
the native profile extends work that previously remained dense. At or above
that crossover the existing sampled default remains selected: a conservative
box domain must not replace a faster qualified route merely because it improves
over dense. This is a continuous work-based producer choice, not a problem-size
or product whitelist. The native profile uses cutoff `1e-16`
and an optional 64 MiB map allowance. Native device identity protects ownership;
it does not impose a product whitelist or an extra CuPy dependency.

Unmatched force compositions/derivative orders, unavailable resident grids,
unsupported optional map artifacts and insufficient map budget retain the dense
route, as does a typed optional native map-allocation failure. Other native
errors are not relabelled as allocation misses. A geometry with mean selected
AO fraction greater than 0.8 also evaluates
the whole dense domain; occupancy is a scheduling guard, not an additional
scientific cutoff. Retained declined-map storage still counts toward the ledger.
Invalid/stale scientific bindings and device execution errors remain errors.

Geometry/basis/grid identity, point order, device and exact derivative capability
bind map lifetime. Density updates invalidate task leases but reuse the geometry
inventory; center updates revoke it. Compatible same-grid force/nonlocal leases
can share it. The separate native KS/XC owner still has its own AO inventory;
deriv=1 and deriv=2 maps are not silently interchanged.

For synchronized complete endpoint qualification, run
`python -m benchmarks.qualify_preao_force --domain-producer auto native ...`
through finite Slurm GPU allocations. Compare with `current-default`
on identical inputs/binaries using five interleaved campaigns. `auto` verifies
the real capability/cost-based portfolio. `current-default` retains only the
pre-existing sampled profile with its original crossover and 16 MiB allowance.
Use `--intrusive` only
in a separate campaign. Retain all cold/warm/moved/moved-warm E+F, actual Fock
histories, independent reference gates and every public force work census.
The baseline changes the force AO domain only: native SCF keeps its existing
sampled domain in both variants. Its `point_ao_visits` is one-traversal
inventory, not the whole-solve execution count; `xc_evaluations` is solve-local.

Keep filenames `reference-ATOMS.json`, `ATOMS-ROUND-MODE.json` and
`profile-ATOMS-MODE.json`, where `MODE` is `current-default` or `auto`, with
rounds 0–4 alternating forward/reversed mode order. Excluded prewarms do not
become timing samples. Run all five
campaigns for one size on the same Slurm allocation and retain source/library
identities, scheduler and device receipts. The host-only verifier checks every
scientific reference gate, indexed lease count, AO/jet/projection/gather census,
same-allocation provenance, warm replay transfers and separate intrusive profile:

```bash
python -m benchmarks.verify_preao_portfolio RECEIPT_DIRECTORY --atoms 3 24 48 \
  --output .artifacts/preao/qualification.json
```

The portfolio verifier records the user-authorized warm acceptance objective;
it does not alter the shared robust-noise calculation or scientific tolerances.
The report keeps all-phase comparisons and guarded warm eligibility. An unchanged
sampled route must preserve its public policy, cutoff, cache and executed work.
A declined dense occupancy domain must avoid material regression
in every phase; a selected sparse domain must improve warm and moved-warm while
avoiding material cold/moved regression. Hardware names remain measurement
provenance, not eligibility whitelists or claims of measurements on other GPUs.

Explicit `dense`, `sampled-jets` and `pre-ao-envelope-native-csr` campaigns remain
useful mechanism experiments. The older `benchmarks.verify_preao_force` checks
their five-round A/B/C receipts, but an all-native comparison against dense does
not qualify a portfolio whose incumbent high-work default is already sampled.

The qualified objective is stable warm and moved-warm E+F improvement above the
shared robust-noise gate. Cold/moved results remain visible; a warm qualification
does not imply statistically significant cold improvement. Rationale and
evidence are in the [pre-AO native CSR decision](../../.agents/notes/implemented/performance/2026-10-06-pre-ao-native-csr.md).
