# DF-CCSD(T) analytic-gradient composition

The native components target the correlation-only DF Hamiltonian: conventional
all-electron RHF supplies the reference Fock/orbitals, while the correlation
two-electron interaction is density fitted. The
`methods::detail::run_df_ccsdt_native` owner composes the complete energy/force
endpoint and is also the public `df-rccsd(t)` / `df-ccsd(t)` Calculator force
owner. Public force publication adds no alternate response equations or
scientific fallback.

## Correlation response

`solve_lambda_parameter_response_cuda` differentiates the retained RCCSD core
and every auxiliary slice of the virtual residual. It returns retained
Fock/integral block cotangents and Q-major `df_bov/df_bvv` cotangents. The latter
cover the virtual residual only. Fresh primal replay and an independently
expanded Lambda action retain the ordinary CC residual acceptance gates.

Staged Lambda already batches its auxiliary primal, transpose and factor
actions under its own complete budget. The residual owner reuses that Q-axis
transform and the same generated ordered accumulation consumer. This does not
change Lambda's program equations, Q order, scalar fallback or independent
expanded audit; the residual and response owners select their tile capacities
independently.

`cc::triples::pullback_df_cuda` supplies all fixed-canonical-input (T)
cotangents. Its T1/T2 sources drive the corrected-Lambda solve. The full
`fock_response_df_cuda` supplies same-space Fock matrices, including internal
occupied/virtual degeneracies. These matrices **replace** the epsilon-diagonal
sources; adding both would count denominator response twice.

The fixed-canonical API requests all nine cotangents by default. Its explicit
`include_gap_response=false` demand omits the epsilon outputs, seed VJP,
reduction, scatter and associated arena. Omitted vectors are empty, not fake
zero derivatives. All nine primal inputs remain validated, uploaded and
charged, and the original energy/denominator audits remain active. Only a
consumer supplying the replacement full-Fock response may omit these sources.

For requested epsilon outputs, `parallel_gap_reduction=true` selects a generic
TensorIR linear-reduction region for virtual dimensions at least four. It
streams pointwise producers, shares aliased scalar outputs, composes vector
marginals without intermediate vectors, and uses fixed FP64 warp trees with
at most 256 scalar partials. Runtime tile counts determine the actual partial
reservation. The original source-major serial order remains the default and
explicit comparison path; no floating-point atomics, fast math, FP32 or new CC
equation is introduced. The graph and schedule have a compiler-visible identity.

`DFGapReductionDiagnostic` reports demand, schedule identity, actual launch count,
reserved workspace, cumulative intermediate elements and logical value
reads/writes and summands. These count generated work, not DRAM transactions,
total instructions or measured peak VRAM. Native reduction tests independently
compare signed seeds with `math.fsum`/NumPy, preserve all requested cotangents,
and refuse publication after nonfinite arithmetic, including partial-drain
overflow. Complete force finite differences must also qualify demand changes.

`pullback_df_factors_cuda` combines retained Gram-block and virtual-factor
cotangents. Compressed Bov includes both ov/vo sectors, so full symmetric
embedding assigns half to each. `pullback_df_source_cuda` then reverses the
original retained molecular source:

```text
B[p,q,Q] = sum_P A_MO[p,q,P] W[P,Q]
A_MO[p,q,P] = sum_mn C[m,p] C[n,q] A[m,n,P]
W = M^(-1/2)
```

The original metric/source owner and immutable frame must remain alive and the
nonzero source identity must match. The common fixed-rank spectral VJP retains
retained/discarded subspace motion and rejects unresolved cutoff crossings.
Raw A cotangents address full, unit-weight `[mu,nu,P]` entries, with no triangular
doubling. All callbacks are provisional until the entire reverse call succeeds.

`scf::CudaDfNuclearSink` consumes raw A rows and the metric cotangent directly on
the producer stream. It visits all nuclear centers without materializing a
coordinate-indexed derivative tensor. Setup is persistent; consume performs no
allocation, copy or synchronization. The producer owns its stream and must
outlive the sink. Call `finish()` only after successful producer completion.
The caller must bind the exact original orbital/auxiliary geometry and basis.

## Conventional-reference response

`hf::rhf_frame_response_cuda` accepts the complete full-MO Fock cotangent and
AO frame cotangent. Compiler-owned matrix maps derive the closed-shell density,
Fock projection, unrestricted frame reverse map, symmetric metric/Pulay
transport and orbital tangent through the shared TensorIR AD machinery.

The physical action remains `G(D)=J(D)-K(D)/2` from the unscreened exact CUDA
provider. It is used for both forward tangent and reverse density response;
substituting DF J/K here changes the method. A single provider stream owns the
matrix maps and resident signed density actions. Optional BLAS executes packed
matrix products with sticky finite audits; the same-arena scalar CUDA schedule
remains available.

GMRES applies the orbital operator on demand. Neither a full MO ERI nor a dense
`(occupied*virtual)^2` Hessian is constructed. The final Z residual is recomputed
with the scalar CUDA lowering, followed by full frame stationarity after the
Z seed is subtracted. Same-space stationarity never divides same-space gaps.
An occupied/virtual gap and residual checks qualify the local solve; they do
**not** certify global RHF stability or the minimum Hessian eigenvalue.

The internal `orbital_screening_tolerance` defaults to zero. A positive value
requests a provisional GMRES solve with a fixed geometry-only Schwarz mask over
complete canonical ERI permutation orbits. This path bypasses density-dependent
shell screening, preserving a fixed self-adjoint action for signed densities.
The immutable source is still prepared unscreened. A provisional solution must
pass the original zero-screening Z residual gate (`1e-10`); otherwise exact GMRES
refines it and the independent audit runs again. Missing optional canonical
storage retains the exact solve. Final reference, stationarity and nuclear
sources always use the original unscreened Hamiltonian. Positive thresholds
are experimental solver controls, not promoted force approximations.

The reference nuclear branch contracts AO hcore and Pulay weights with existing
CUDA derivative providers. Its two-electron source `P:G'(D)` defaults to
symmetric polarization `[E2'(D+P)-E2'(D-P)]/2`, with `E2(D)=D:G(D)/2`. This
retains the admitted shell consumer and its primitive/component reuse in two
bounded passes. The identity requires the same fixed unscreened linear source
for both operands. Set `symmetric_polarization=false` to retain the original
three-pass identity `E2'(D+P)-E2'(D)-E2'(P)` for matched validation.

The separate `bilinear_derivative` opt-in uses compiler-owned direct weights,
Cartesian projection, symmetry-unique angular buckets and translation
reconstruction. It can replace generic or bounded through-f consumers when
canonical storage is admitted; specialized SPD leases retain their existing
consumer. This experimental route requires a measured crossover: a single
canonical AO pass can lose reuse present in a shell consumer. Neither route
materializes a derivative ERI tensor, and both retain explicit capacity
fallbacks. All routes are independent of orbital-response dimension.
The result is an
**electronic gradient**: the final method must add the correlation-source and
nuclear-repulsion gradients, then negate once to publish forces.

## Complete native owner and qualification

The complete owner starts from normalized orbital/auxiliary geometry, computes
a fresh exact CUDA RHF reference and native DF-CCSD amplitudes, then composes the
triples, corrected-Lambda, factor/source and reference branches above. Forces
retain the original source/frame across the CC solve. Energy-only execution
releases that state and evaluates the same Hamiltonian without response.

Each phase charges all other live owners to its admission. The nuclear sink
uses the same immutable geometry/basis arguments as the source producer, and
the source identity must match. After successful source reverse and sink drain,
the owner releases completed CC/amplitude/factor/source buffers before the
exact-reference Z/Pulay phase. Nuclear repulsion is added once through the shared
ionic-gradient assembly, followed by a single sign conversion. Errors publish
no partial energy/force result. An optional CCSD-only mode omits triples for
separate closure validation.

The internal interfaces admit complete numeric payloads before execution;
outer callers must charge all other live owners. Matrix response reports zero
explicit Hessian elements, J/K actions, derivative passes, generated contraction
summands, BLAS calls and owner-managed transfers, including derivative operand
uploads. Provider-internal setup and execution transfers are excluded; these
counters are not a complete endpoint traffic ledger.

Response wall times separately report setup, reference audit, weight assembly,
Z solve, independent residual audit, one-electron and two-electron derivatives.
Optional `profile_jk` observes the ordinary value dispatch without selecting a
different physical execution path. It synchronizes each J/K call and records
wall/device time and, when all selected channels have canonical census coverage,
canonical integral counts. Compare matched selectors and hardware. J/K times are
subsets of the phase times; quartet visits, contracted ERI values and three-axis
derivative jets are distinct
work units and are not FLOPs. Census/timing flags and action counts distinguish
unmeasured fields from measured zeros. Provider-internal transfers remain outside
the owner-managed traffic counters.

Relevant validation modules include `test_df_cc_lambda.py`,
`test_df_source_metric_response.py`, `test_df_nuclear_sink.py`,
`test_rhf_frame_response.py`, `test_rhf_frame_response_codegen.py` and
`test_rhf_frame_response_cuda.py`. Real-GPU tests require finite Slurm allocation.
The RHF module checks independent forces and nonzero-Z molecular energy finite
differences with both scalar and BLAS schedules.

`test_df_complete_force.py` compares complete native CCSD/CCSD(T) energies and
forces with independent libcint/PySCF correlation-only DF Hamiltonians. It
includes nonzero triples, two-step energy directions, every water coordinate,
native energy/force consistency, auxiliary g in both representations, fixed-rank
duplicate-auxiliary metrics, translation and failure publication. PySCF and its
tiny dense ERIs are test oracles only. Set `GENERATIVEQC_DF_COMPLETE_FORCE_TEST=1`
and `GENERATIVEQC_DF_COMPLETE_FORCE_PROBE` for the native validation seam.

Hundreds-AO complete-force qualification is retained separately from the public
wrapper: the 230-AO / 488-auxiliary ethane endpoint has completed on merged
production sources with bounded-memory, response-residual, stationarity and
independent finite-difference checks. Later response optimizations may change
wall time but not the public Hamiltonian or force acceptance contract. See the
[composition decision](../../.agents/notes/implemented/architecture/2026-10-04-complete-native-df-ccsdt-forces.md)
and the retained `rhf-response-accelerators-1901` / `cc-packed-diis-1902`
evidence.

## Native benchmark controls

The `benchmarks/df_ccsdt_force_endpoint.cpp` executable accepts the following
positional arguments (brackets denote optional trailing controls):

```text
df-force-endpoint INPUT OUTPUT_JSON REDUCTION_0_OR_1 [MATRIX_0_OR_1 [FORCES_0_OR_1 [LAMBDA_MATRIX_0_OR_1 [Q_BATCH_LIMIT [DIIS_HISTORY [CCSD_Q_BATCH_LIMIT [ORBITAL_SCHWARZ [PROFILE_JK_0_OR_1 [NUCLEAR_0_LEGACY_1_CANONICAL_2_SYMMETRIC [DERIVED_DENOMINATORS_0_OR_1 [Z_TRUE_RESIDUAL_INTERVAL [Z_DF_PRECONDITIONER_0_OR_1 [Z_RECYCLE_REPEAT_0_OR_1 [PACKED_DIIS_0_OR_1 [RESIDENT_JK_MAXIMUM_BYTES_OR_AUTO [PARALLEL_GAP_0_OR_1 [REQUEST_GAP_0_OR_1 [REFERENCE_TOLERANCE]]]]]]]]]]]]]]]]]]
```

`MATRIX`, `FORCES` and `LAMBDA_MATRIX` default to one, `Q_BATCH_LIMIT` to eight,
`CCSD_Q_BATCH_LIMIT` to eight, and `DIIS_HISTORY` to six. `DIIS_HISTORY` retains argument position eight and
accepts zero (disabled) or integers two through twenty. A decimal or scientific
notation token in this position is rejected; it is never guessed to be a
screening threshold. `CCSD_Q_BATCH_LIMIT` retains position nine. Both batch
limits require complete unsigned integer tokens. Response controls follow at
positions ten through twelve: `ORBITAL_SCHWARZ` defaults to
zero and requires a complete finite nonnegative number, `PROFILE_JK` to zero and `NUCLEAR` to two (symmetric polarization).
`NUCLEAR=0` selects the legacy three-pass identity and `NUCLEAR=1` explicitly
opts into the experimental canonical bilinear derivative.
`DERIVED_DENOMINATORS` follows all existing controls at argument thirteen and
defaults to one. Zero retains the explicit CUDA denominator representation.
It accepts only the complete token `0` or `1`; it does not change the meaning
of the screening token at argument ten, profiling at eleven, or the nuclear
response schedule at twelve.

Arguments fourteen through seventeen select the true-residual interval, DF Z
preconditioner, repeated recycling endpoint and packed DIIS. Argument eighteen
accepts a resident exact J/K byte limit or `auto` to retain its ordinary policy.
Arguments nineteen and twenty select parallel gap reduction and requested
epsilon cotangents. Both accept only `0` or `1`; omitted controls currently
retain serial reduction and all cotangents. Requesting these diagonal sources
is a matched-work diagnostic control for the complete force owner: its
full-Fock response replaces them and never adds them to molecular forces.
Automatic demand promotion still requires complete cold-force acceptance;
isolated reduction accuracy or a smaller arena does not qualify that change.
Keep automatic J/K selection and all earlier selectors matched when comparing
serial/all-output, parallel/all-output and demand-pruned endpoints.

Argument twenty-one optionally sets both RHF energy and density tolerances to a
finite positive value no larger than `1e-12`. Without it, the benchmark retains
its original `1e-12` energy and `1e-11` density tolerances. This benchmark-only
control cannot loosen either criterion or change CC, Lambda, Z, stationarity,
or paired-force acceptance gates. JSON records both requested tolerances and
the original RHF's final energy change, density RMS and iteration count; these
diagnostics do not rebuild or replay the reference. Iteration work is null if
a discarded endpoint attempt prevents complete work accounting. Match this
argument across schedules when investigating cold-reference variability.

For example, an exact force endpoint with the default six-vector DIIS history
and explicit symmetric response is:

```sh
./df-force-endpoint molecule.input force.json 1 1 1 1 8 6 8 0 0 2
```

Historical response benchmark receipts retain the CLI for their recorded source
revision. When adapting such a command to the current executable, insert the
DIIS history and CCSD Q batch limit before the orbital screening threshold.
Do not rewrite retained receipt commands or imply that they used this layout.

## Same-primal response diagnostic

`methods::detail::diagnose_df_ccsdt_gap_schedules` invokes the native cold
RHF/DF-CCSD owner once and compares serial/all-output, parallel/all-output,
omitted-output and repeated serial complete force compositions. Each composition
uses an independently owned host copy through the existing force implementation;
the physical reference and original DF source/metric/frame remain shared. A
bit-pattern census guards the nine explicit triples inputs before each response
and checks that the retained original is unchanged afterward. No replacement
metric, CPU oracle or second CC equation is used.

The diagnostic requires an explicit positive numeric budget and disallows Z
recycling. All fixed force outputs are charged during the common cold solve.
Admission precedes cloning and includes the original host copy and one working
copy, with shared source/reference ownership charged once. The original host
buffers remain reserved in every response phase; the original DF source and any
still-live exact-reference source remain charged beside the final independent
orbital provider. A failure publishes no partial comparison. Common native
owners are released before the total comparison timer is stopped.

`benchmarks/df_gap_same_primal_endpoint.cpp` accepts `INPUT OUTPUT_JSON` with the
same normalized geometry/basis/budget input as the cold endpoint benchmark.
It uses the retained cold benchmark's reference, DIIS, response and automatic
J/K controls. JSON separates the single common native call, clone work and each
response composition. These are diagnostic timings, **not independent cold
endpoint timings**, and cannot replace cold acceptance or qualify an endpoint
speedup. Independent force finite differences, unchanged residual/stationarity
gates and the strict paired force gate still apply.

Each comparison also records ordered bit-pattern identities and element counts
for 35 existing host payload boundaries: seven requested triples cotangents,
full-Fock response, corrected Lambda, composed parameter/factor sources, DF
nuclear gradient and coefficient source, orbital response/weights, and final
forces. This fixed scalar metadata adds no retained numeric intermediate or
GPU transfer; empty payloads have explicit zero element counts. Ordinary cold
calls do not collect these fingerprints. Fingerprinting time is reported
separately, along with logical host value reads, and remains included in
diagnostic response/total times. Identity
differences localize the first observed divergence but are not numerical
acceptance gates, proof of causation, or a replacement for force tolerances.

The optional `--physical-replay` benchmark mode runs one native primal,
triples response and Lambda solve, then four source/nuclear responses on the
same factor seed and four orbital/nuclear responses on one final `bar_f/bar_c`
seed. The original physical reference is shared and its host identity is
guarded; DF gradients are not inputs to the orbital solve. This diagnostic
requires an explicit positive budget and refuses recycling/DF preconditioning.
Small gradient outputs are reserved before the cold solve; preceding orbital
matrices are retired before the next response owner is admitted.

Unlike the host-boundary-only comparison, physical replay explicitly copies
device three-center and metric weights through one admitted row/metric buffer.
It reports the buffer capacity, logical values, transfer bytes and synchronized
census time. Host weight copies are discarded after hashing; no complete
three-center weight tensor is retained. Both the observer overhead and repeated
physical work remain included in diagnostic timings. The observer can change
launch timing, so a passing replay does not supersede an uninstrumented failure.
Weight hashes distinguish source-weight variability from later contraction
variability only within the observed run; unchanged force/residual gates and
independent reference qualification still apply. Ordinary cold and same-primal
schedule calls do not enable physical replay or its added transfers.
