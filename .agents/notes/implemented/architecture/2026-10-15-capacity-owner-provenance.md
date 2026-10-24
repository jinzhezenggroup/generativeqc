# Decision: record capacity owner bytes as provenance, test their behavior

Status: implemented (local repair candidate; no new device qualification)
Date: 2026-10-15

## Problem

At PR #2515 head `efe0db1b435374c263f7c133cda0a05e5315b2af`, the
capacity suite rejected source drift in three complete Python owners. The
snapshot-initializer and public-force errors masked a third endpoint-owner
mismatch. These were whole-body acceptance pins, so harmless edits and the
reviewed native stationarity-allowance transport failed identically to unsafe
changes. Merely accepting three new hashes would repeat the problem and violate
the repository rule against permanent generated-source/function-body CI pins.

The historical pinned identities were:

| Owner | Former acceptance SHA-256 |
| --- | --- |
| `NativeKsSnapshot.__init__` | `985187cc57507421e9c7990c022e50b32b9a0d6d0a39a0507ae9e2df3d7e6573` |
| `PreparedBatch._public_dft_cuda_force` | `60885551705fbb5410a450037c08bbf9e3e9674502b9b77ff94d78c001cb6e28` |
| `_complete_rks_cuda_gradient_diagnostic` | `dd4e8026d26989e122b28b46098deb5925421ccaf11a7d26142670272f60dbb8` |

They identify historical reviewed source; they are not numerical acceptance
criteria for later source. Original failing CI evidence and historical numerical
results remain evidence for their original revisions, never relabeled as a pass.

## Decision and boundaries

Remove only those three equality gates and continue reporting their current
source hashes. Keep missing/ambiguous-owner errors, input and manifest integrity,
import-versus-disk consistency, reference fingerprints, and all unrelated
resource/planner/native ABI qualification checks. This is a bounded migration;
other legacy source-spelling guards remain migration debt, not an endorsement of
new pins. A report remains a static preflight, not proof of numerical correctness
or complete GPU qualification.

Before retiring their 31 existing source-mutation cases, execute those exact
semantic perturbations against behavioral tests. Sixteen already fail existing
controls. Add full-owner calls for the fifteen gaps: primitive rank, complete
pair work, DF/Combined separation, final component publication/reduction,
concurrent device accounting, selected-grid propagation, public AO policy and
composite admission. Use manufactured values and inert backend owners to observe
actual production orchestration through publication. Do not assert production
AST shape, private variable spelling, or a historical function-body hash.
Semantic work counts and required ABI/resource ownership remain valid assertions.

The snapshot transport tests execute the real constructor against a small native
ABI double, including legacy absence of the optional allowance query, domain and
backend admission, finite bounded allowances, immutable storage and explicit
failure cleanup. Retain the partially constructed object while checking failure
cleanup so its destructor cannot mask a missing exception cleanup. Force tests
check native provenance and explicit opt-in, strict response defaults, unchanged
orbital/weighted-density algebra, and validation before device work.

## Preserved historical audit rationale

The removed endpoint comment had accumulated the following substantive reasons
for earlier audits. They remain requirements even though an owner hash no longer
stands in for their behavior:

- Full-range Combined returns three channels; Separate/DF returns four. Keep
  strict shape/finite checks, bounded v1 fallback, complete source coverage and
  reduction, and prohibit DF silently taking Combined direct-integral semantics.
- Native host staging must fit its concurrent reserve; device capacity subtracts
  already-live storage and reports the native peak. DF resident-one-electron
  metadata distinguishes host fallback from device execution, without extending
  admission or asserting DF-response qualification.
- AOT uses the same coverage gate and admitted build-graph provenance without
  generating reduction/weight IR. Lazy source reuse changes preparation and
  telemetry, not arithmetic, native requirements, work windows or reserves.
- CSR/exact-bitmask AO maps preserve geometry allowances and reserve both offset
  mirrors plus compact scratch. The original prepared lease, force-memory
  receipts and allocation ordering preserve capacity equations. Read-only grid
  admission follows requested capability; do not restore writable indexed
  staging in the no-map census.
- Optional precision candidates require fitted/composite admission, selected
  policy forwarding, attempt/work evidence, and unchanged finite/shape,
  complete-grid, pending-window and allocation bounds. Public source owners must
  close on success and failure; force sign and immutable publication remain
  externally observable contracts.
- Snapshot identity remains MethodIR/KSOptions based. Density-only automatic
  snapshots require exact domain 5; other automatic/curated domain, token,
  coefficient/spin and exact-grid requirements are not relaxed.

## Historical strict-FP64 census versus optional budget stationarity

The capacity census selects `fp64_energy_forces` rows in `qualify_capacity.py`.
At the original audited head (before `199ce3ca2d7dde0f7dc3044001cf86d3fad27158`),
`src/dft/cuda_ks.cpp` enables `budget_stationarity` only when
`budget_fock` is admitted: AUTO precision, canonical all-electron PBE0 model and
full-range exact terms, default grid/device-fused schedule, no fitted/range or
nonlocal substitution, and loosened SCF/screening controls. FP64 does not enter
that branch. Native snapshot export supplies legacy `1e-8` when no allowance was
published; the optional C API checks the current token before returning it.

Python accepts only finite positive allowances at most `1e-4`. The changed public
force call opts in, and the endpoint opts in only for a native allowance greater
than `1e-8`; the response contract remains strict by default. Physical residual
and density reconstruction may consume the authenticated force allowance.
Orbital eigen-residual, orthonormality and weighted-density checks remain strict.
This source/behavior audit found no defect in those three transport/wiring deltas.
It does not establish the new Fast candidate's numerical error or GPU throughput.

## Evidence and full legacy-mutant mapping

All 31 rows below were executed separately against restored production sources.
“Existing” identifies a pre-existing executable control; “added” identifies a
full-call control added for a previously surviving mutant. Each listed test
failed under its mutation and passed with the original owner. Cases are numbered
for this audit, not an assertion of internal call counts.

The current-master composition replay corrected row 9's control attribution.
The earlier fitted-response probe's failure was an unrelated `NameError` in
its extracted scaffold, not evidence of reserve enforcement; that failure is
not counted as a semantic control. The dedicated concurrent-reserve test
rejects the mutation when actual staging is 33 bytes against a 32-byte reserve,
and both cases pass with the original comparison restored. The historical
logs remain retained; the table records the independently verified control.

The replay at head `199ce3ca2d7dde0f7dc3044001cf86d3fad27158` also invalidated
row 5's old extracted-loop attribution: that scaffold omitted `grid_points`, so
its `NameError` did not prove bounded-chunk enforcement. The full-owner
publication fixture now enforces its configured one-pending-tile window. The
same full-grid mutation queues a second tile and fails that explicit resource
assertion across all ten endpoint combinations. This corrected control replaces
the invalid attribution; both historical failures remain retained as negatives.

Test-file aliases (all under `tests/python/`):

- **A**: `test_stationary_capacity_publication.py`
- **B**: `test_stationary_large_grid_work.py`
- **C**: `test_stationary_aot_no_compiler.py`
- **D**: `test_stationary_df_provider_failure.py`
- **E**: `test_stationary_grid_schedule.py`
- **F**: `test_public_stationary_policy_forwarding.py`

| # | Semantic perturbation (old → mutant) | Executable control | Coverage |
| --- | --- | --- | --- |
| 1 | `("coulomb", 4, "four_center_eri")` → `("coulomb", 2, "four_center_eri")` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 2 | `for begin in range(chunk_begin, chunk_end, tile_points):` → `for begin in range(chunk_begin, chunk_end, 2 * tile_points):` | B: `test_actual_geometry_loop_preserves_offsets_and_stops_at_failed_window` | existing |
| 3 | `int(source.metadata[6])` → `0` | C: `test_public_aot_force_does_not_probe_nvcc` | existing |
| 4 | `if requires_native_integrals and not native_complete_integrals:` → `if False:` | D: `test_enlarged_direct_domain_cannot_select_ao_task_fallback` | existing |
| 5 | `for chunk_begin, chunk_end in grid_work.chunks():` → `for chunk_begin, chunk_end in [(0, grid_points)]:` | A: `test_complete_stationary_composition_and_work_accounting` (configured pending-tile bound) | corrected full-owner control |
| 6 | `sources.drain_geometry()` → `pass` | B: `test_actual_geometry_loop_preserves_offsets_and_stops_at_failed_window` | existing |
| 7 | `or work["xc_points"] != grid_work.grid_points` → `(removed)` | B: `test_publication_checks_points_even_for_zero_atom_pair_work` | existing |
| 8 | `work["grid_pair_visits"] != pair_visits` → `work["grid_pair_visits"] > pair_visits` | A: `test_complete_stationary_endpoint_rejects_missing_or_extra_pair_work` | added |
| 9 | `> native_integral_host_reserve` → `> max_host_bytes` | D: `test_native_host_staging_must_fit_its_concurrent_reserve` | existing |
| 10 | `(3 if combined_requested else 4, na, 3)` → `(2 if combined_requested else 4, na, 3)` | D: `test_default_complete_direct_owner_combines_two_electron_sources` | existing |
| 11 | `(3 if combined_requested else 4, na, 3)` → `(3 if combined_requested else 3, na, 3)` | D: `test_complete_fitted_response_and_explicit_exact_fallback_remain_distinct` | existing |
| 12 | `or not np.isfinite(native_integral_components).all()` → `or False` | D: `test_fitted_provider_rejects_invalid_publication` | existing |
| 13 | `not use_fitted_integrals` → `True` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 14 | `{"combined_two_electron": True}` → `{"combined_two_electron": False}` | D: `test_combined_unavailable_retries_complete_separate_owner` | existing |
| 15 | `if combined_requested and native_integral is None:` → `if combined_requested:` | D: `test_default_complete_direct_owner_combines_two_electron_sources` | existing |
| 16 | `combined_requested = False` → `combined_requested = True` | D: `test_combined_unavailable_retries_complete_separate_owner` | existing |
| 17 | `components.pop("coulomb", None)` → `(removed)` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 18 | `combined_two_electron=native_combined_integrals,` → `combined_two_electron=False,` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 19 | `"two_electron" if native_combined_integrals else "coulomb"` → `"coulomb"` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 20 | `native_integral_budget = max_device_bytes - peak` → `native_integral_budget = max_device_bytes` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 21 | `int(native_integral_resources.get("one_electron_host_peak_bytes", 0))` → `int(native_integral_resources.get("one_electron_host_peak_bytes", 0)) // 2` | D: `test_native_host_staging_must_fit_its_concurrent_reserve` | existing |
| 22 | `+ int(native_integral_resources.get("one_electron_device_peak_bytes", 0)),` → `+ int(native_integral_resources.get("one_electron_device_peak_bytes", 0)) // 2,` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 23 | `sum(value.host_bytes for value in layout.tensor_plans.values())` → `0` | E: `test_tight_ecp_candidate_includes_prepared_tensor_host_storage` | existing |
| 24 | `grid_plan = layout.grid_plan` → `grid_plan = None` | A: `test_complete_stationary_composition_and_work_accounting` | added |
| 25 | `"resident_ao_producer": decision.producer` → `"resident_ao_producer": "sampled-jets"` | F: `test_public_force_preserves_admission_and_selected_policy` | added |
| 26 | `"resident_ao_max_active_fraction": decision.max_active_fraction` → `"resident_ao_max_active_fraction": 1.0` | F: `test_public_force_preserves_admission_and_selected_policy` | added |
| 27 | `active_ao_producer=decision.producer` → `active_ao_producer="sampled-jets"` | F: `test_public_force_preserves_admission_and_selected_policy` | added |
| 28 | `active_ao_max_active_fraction=decision.max_active_fraction` → `active_ao_max_active_fraction=1.0` | F: `test_public_force_preserves_admission_and_selected_policy` | added |
| 29 | `device_name=getattr(self, "_stationary_cuda_device_name", None)` → `device_name=None` | F: `test_public_force_preserves_admission_and_selected_policy` | added |
| 30 | `fitted_gradient_admission(calculator, state, basis)` → `True` | C: `test_public_aot_force_does_not_probe_nvcc` | existing |
| 31 | `composite_gradient_admission( calculator, state, basis )` → `True` | F: `test_public_force_preserves_admission_and_selected_policy` | added |

Additional stationarity negative controls reject eleven independent changes:
relaxed legacy default, relaxed allowance upper bound, ignored native query,
missing constructor cleanup, missing public or endpoint opt-in, skipped endpoint
validation, admitted non-native allowance, ignored density allowance, relaxed
weighted-density algebra, and removed force cap. Benign initializer-local
renaming and AST formatting of the three owners pass the behavioral probes.
Moving independent domain/backend rejection before the allowance query also
passes all 52 snapshot transport cases; lifetime assertions allow that reordering.
Full endpoint/public-owner formatting also passes all 16 new composition/policy
cases. The capacity report accepts harmless owner comments while recording their
changed hashes; unrelated nested-owner gates still constrain broader refactors.

Local validation at this head:

- `tests/python/test_dft_mp_v1_capacity.py`: 360 passed with verified ccache,
  no skips. Original targeted reproduction: 1 failed, 4 passed at the stale
  snapshot pin; no original failure or historical numerical evidence was deleted.
- Executable provider, work, scheduling, AOT, snapshot transport, publication,
  public-policy and selected stationarity-contract tests: 281 passed.
- All 31 legacy semantic mutants fail (16 existing controls, 15 added), plus
  all 11 stationarity mutants; benign controls pass.
- A broader stationarity-file attempt needed the unavailable native library for
  29 oracle cases; those failures were not counted as passing. No native rebuild
  or fresh GPU qualification was performed for this migration. Independent
  numerical/real-device gates remain required for the precision feature.

Composition update for author refinement `c59f1c1c0eecde86d389e9b9e52c6305939f23a9`
with master `e88dee9d2f242f76bb8f712aaff2cd0cab103e41`: the three audited
Python owners and their semantic mutation coverage are unchanged. The separately
compiled final-closure test now executes 160 combinations of budget admission,
authenticated stationarity, density thresholds (including adjacent values around
`1e-4`), incremental mode, spin and ECP. Nine semantic gate mutations compile
and fail at runtime; benign gate reordering passes. This updates test coverage
for the author's changed stopping policy without changing numerical production
code or treating historical GPU measurements as qualification of a new source.

Reproduce the host suites (with normal development dependencies and a verified
compiler-cache launcher on PATH or selected through `CCACHE`):

```sh
python -m pytest tests/python/test_dft_mp_v1_capacity.py -q
python -m pytest \
  tests/python/test_stationary_df_provider_failure.py \
  tests/python/test_stationary_large_grid_work.py \
  tests/python/test_stationary_grid_schedule.py \
  tests/python/test_stationary_aot_no_compiler.py \
  tests/python/test_ks_snapshot_stationarity_transport.py \
  tests/python/test_stationary_capacity_publication.py \
  tests/python/test_public_stationary_policy_forwarding.py \
  tests/python/test_dft_stationary_gradient.py \
  -k 'not test_dft_stationary_gradient or force_budget or budget_contract or cuda_force_entry' -q
```

## Integration with shared-module Fast policy at `199ce3ca2`

The historical canonical-PBE0 admission description above is not the current
Fast domain. The author subsequently selected precision through shared Direct
capabilities and unified force-contract construction through `for_native_force`.
This repair composes that author head with master
`cb8781fb1bdcfb9166ef07cfed463c0482dea752` without changing production numerical
or GPU implementation. The new author test for inherited native force allowance
is preserved. Inert test doubles inherit the real force-policy factory and
replace only their state/device boundary; they still assert strict selection
when no native allowance exists.

That author change also invalidated a fourth historical acceptance pin, the
`Calculator.cuda_hybrid_force` assignment. Its former SHA-256 was
`18f4f010596672eb47b8d085e28b8a26373c41178ac1c6a5ff4fa705ef2f3944`.
Keep the current assignment digest in report provenance while removing equality
as an acceptance condition. Existing executable route tests retain all five
safety conditions: CUDA device, no ECP, options present, device-fused XC schedule,
and eligible MethodIR. They cover strict/AUTO requests across curated/generated
hybrids and reject incompatible source compositions. A benign predicate
reordering changes its reported digest while preserving those executed decisions.
Each of the five retained guards is independently weakened in a local negative
control; none is removed from production. Source-shape checks for locating an
unambiguous owner and unrelated integrity/ABI/resource checks remain intact.

The benchmark runner now reserves a new output using exclusive creation before
scientific work. Every update uses that same open descriptor, so existing files,
dangling symlinks and competing writers cannot be overwritten by reopening the
path. The report takes its native-library digest from metadata for the actual
loaded Calculator library rather than a possibly absent or different environment
path. Each endpoint attempt's mode, repeat and phase are saved before execution;
returned data, oracle-gate failure and endpoint exceptions remain attributable to
that attempt. Protocol failures are retained after reservation too. Host-only
mocks and negative controls verify these evidence-retention behaviors; they do
not run CUDA or independently validate the author's GPU measurements.

Current-head host evidence: the selected integration suite passes 1,120 tests,
with 49 ECP cases skipped because native prerequisites are unavailable. Its
focused stationary-gradient selection deselects 82 other cases; a separate
host-complete stationary-gradient selection passes 78 and excludes only its 29
native-library oracle cases. These selections overlap and are not added as
independent test counts. All 31 original semantic mutants are rejected using
the corrected row 5 and row 9 controls; their restored shared suite passes 67.
The five retained hybrid-guard mutants and five benchmark-retention mutants are
rejected; restored hybrid and benchmark controls pass 54 and 16 respectively.
Ruff, formatting and whitespace checks pass. These are host/test-integration
results, not full CI or new numerical/device qualification.

## Rejected alternatives and consequences

Refreshing hashes alone would hide the distinction between semantic damage and
benign source edits. Dropping all qualification checks would also lose genuine
provenance and resource safety. Removing only the pins without first executing
the mutations would miss fifteen real behavioral coverage gaps. None of those
alternatives is the chosen repair.

The new probes deliberately do not replace independent scientific or GPU
oracles: manufactured integral channels verify composition and accounting, not
integral accuracy. Future endpoint changes should preserve or update these
observable contracts, rather than restore whole-owner spelling gates. Revisit
when the native ABI, admission domain, publication contract or capacity equations
change, and continue migrating other legacy pins only with equivalent evidence.

## References

- PR #2515: https://github.com/jinzhezenggroup/generativeqc/pull/2515
- Source-guard migration context: #2296
- Repository `AGENTS.md`, `docs/maintainer/performance_engineering.md`
- `tools/dft_mp_v1/qualify_capacity.py`
- `python/generativeqc/_ks_snapshot.py`, `_dft_gradient.py`, `_stationary_cuda.py`
- `src/dft/cuda_ks.cpp`, `src/api/c_api_ks_snapshot.cpp`
