# Resident pre-AO force domains

The explicit stationary CUDA force route can build a conservative AO domain
before AO evaluation. Compiler-owned Gaussian region envelopes produce an
ascending, unique device CSR inventory. Each tile lends the existing `AoGridBlockLayout`
to the indexed AO, local density and projection consumers; AO labels do not
round-trip through Python or upload again on warm replay. Current public force
calls use the native prepared controller when its item capability is admitted.
That controller has its own capability/work/resource-based exact-jet map
admission; it does not consult the Python active-AO profile registry. The
explicit/compatibility selector and the native prepared controller are distinct
execution routes, not evidence that one route's policy was used by the other.

## Native prepared forces

`src/dft/cuda_stationary_capacity.hpp` admits optional exact-jet bitmasks for
supported four-/ten-jet force bindings at dense point×AO² work of at least
`2**35`, with at most 16 MiB of optional map storage. This is independent of
functional name, spin, ordinary/composite composition and Direct/DF Fock
selection. Public method admission is still determined by the compiled
scientific binding; map capability cannot widen it. Unsupported jet orders,
ECPs, insufficient capacity, optional allocation failure and mean occupancy
above 0.8 retain the full dense force domain at the same `1e-16` cutoff.

The geometry-owned native grid consumer discovers every required derivative
jet, retains masks across density-only replay, and rejects stale point spans,
geometry epochs, derivative orders and binding identities. A moved geometry
replaces the parent quadrature and controller. A shared XC/nonlocal quadrature
may consume the same indexed panel; a distinct nonlocal quadrature retains its
independent dense traversal. The separate FP32 grid route does not consume an
FP64 map. Optional maps never shrink mandatory dense AO or stationary capacity;
the cached tier charges device masks/staging and host offsets.

For same-endpoint dense ablations, set
`GENERATIVEQC_CUDA_FORCE_ACTIVE_AO=0` before the native force controller is
created. This only declines optional selection; it cannot force admission or
change the cutoff, precision, scientific model, or native force capability.
Existing controllers retain their prepared schedule. The capacity query remains
a conservative upper bound even when this debug control declines the map.

With `GENERATIVEQC_DF_TRACE`, each `stationary_force` receipt reports executed
`grid_active_point_ao`, `grid_dense_point_ao`, `grid_ao_jet_values`,
`grid_projection_fma_pairs`, `grid_discovery_ao_jet_values` and discovery passes.
`indexed_grid` identifies actual indexed execution, not merely a retained map
that the FP32 route does not use. Discovery belongs to the first force call;
warm replay has zero rediscovery. Trace stage durations are diagnostic, not
complete endpoint timings.

`tests/python/test_native_cross_functional_force_ao_cuda.py` exercises public
native LDA/PBE/PBE0/r2SCAN/WB97M-V RKS/UKS force calls above the unmodified
production work crossover, including fitted and Direct controls. It independently
reconverges both geometries, forbids Python/CPU force orchestration, and compares
automatic, same-capacity dense and minimum-capacity native endpoints. Its
cold/warm/moved/moved-warm samples retain numerical errors, physical residuals,
SCF/Fock counts, complete E/F timing and actual AO work. Run with
`GENERATIVEQC_DFT_CUDA_TEST=1` in a finite Slurm allocation and a matching native
build/reference environment. These tests establish native execution and
numerical/resource behavior, not universal profitability or a new crossover.
The source/build-bound [retained qualification](../../benchmarks/results/native-force-ao-20261117/README.md)
provides raw independent arrays, semantic work and complete endpoint replay.

For fresh-process comparisons without importing an oracle, use
`python -m benchmarks.native_force_ao_endpoint --receipt ... --method ...
--mode auto --output ...` and the same-binary `--mode dense` arm. Independent
references must already be retained in the receipt. Interleave matched fresh
processes on one finite Slurm allocation, including cold and moved convergence,
and compare SCF/Fock branches as well as actual force AO work. A private minimum
allowance may first retire an automatic force tile; re-query the bounded
capacity before checking its exact minimum-minus-one rejection.

## Explicit/compatibility forces

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

Ordinary and composite explicit stationary forces use the same map binding and
cache owner. The binding includes the grid object and basis/geometry generations,
scientific identities, derivative order, tile layout, producer, cutoff, occupancy
guard and admitted cache allowance. A replacement releases the old cache before
allocating another one. Composite semilocal and nonlocal consumers share a map
only on the same quadrature; independent nonlocal grids retain their own dense
AO traversal. Composite automatic selection remains dense until a production
profile has complete endpoint qualification. Neither a test-only profile nor
the historical ordinary-route qualification promotes native-controller maps.

The opt-in numerical matrix in
`tests/python/test_cross_functional_force_ao_cuda.py` compares dense and selected
explicit stationary Direct LDA/PBE/PBE0/r2SCAN/WB97M-V RKS/UKS energy and forces
with independently converged GPU4PySCF, including moving-grid response. It exercises first-call,
warm, moved, moved-warm and zero-cache endpoints, records actual selected/dense
AO work in JUnit properties, and includes separate XC/nonlocal grids. It calls
the retained stationary consumer after each native energy solve, without
overriding native capability or mislabelling dense public forces as selected
work. A separate public native-force control forbids Python force orchestration
and checks the same independent reference. The matrix qualifies consumer
correctness and lifecycle behavior under test-only admission. Its small-fixture
timings do not establish a production crossover or a fresh-process cold gain.
Run these tests with `GENERATIVEQC_DFT_CUDA_TEST=1`, a matching native/AOT build,
and the reference dependencies inside a finite Slurm GPU allocation.

Ordinary density-fitted forces use the separate native AO-jet bitmask producer
at dense point×AO² work of at least `2**35`. Its discovery cost does not share
the sampled producer's crossover. It checks every requested AO derivative at
the unchanged `1e-16` cutoff, including second derivatives for forces, and
retains every grid point and Becke response. Explicit force tiles and budget-auto
tiles share the same admission, 64 MiB optional map allowance, 0.8 mean active
fraction guard and complete-owner byte limits. Smaller work, missing resident
capabilities and optional capacity/occupancy refusals retain dense execution.
The fixed tile request still goes through complete capacity planning; specifying
`KsOptions.force_tile_points` does not disable the fitted AO map. Map identity
includes the actual tile layout, so changing the tile rebuilds its inventory.
The [crossover decision](../../.agents/notes/implemented/performance/2026-10-25-df-force-bitmask-crossover.md)
records endpoint qualification and the reason for the separate threshold.

For synchronized complete endpoint qualification, run
`python -m benchmarks.qualify_preao_force --domain-producer auto native ...`
through finite Slurm GPU allocations. Compare with `current-default`
on identical inputs/binaries using five interleaved campaigns. `auto` verifies
the real capability/cost-based portfolio. `current-default` retains only the
pre-existing sampled profile with its original crossover and 16 MiB allowance.
After compiler-input changes, build the needed stationary AOT `*_manifest`
targets, such as `generativeqc_stationary_pbe0_rks_spd_manifest`, as well as the
core library. These targets build their corresponding libraries and record the
current contract/binary identities. Library-only targets can leave an old
manifest; do not bypass the fail-closed artifact loader to qualify such a build.
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
evidence are in the [pre-AO portfolio decision](../../.agents/notes/implemented/performance/2026-10-06-pre-ao-portfolio-default.md).
