# Decision: extend the active-AO default without regressing its sampled incumbent

Status: implemented; merged-source warm qualification passed
Date: 2026-10-06

## Problem

The [native CSR experiment](2026-10-06-pre-ao-native-csr.md) froze its dense
public baseline at `a85459c13d16741ef5fab37cca6a64fc4a5bca15`. While those
measurements ran, #2007 enabled `ordinary-direct-active-ao-cost-v3` on master.
At merged baseline `196126d88`, work above the existing continuous dense
point×AO² crossover of 173,946,175,488 therefore already selects sampled jets.
Comparing only with dense would overstate the value of replacing that default.

## Decision

Preserve the incumbent profile, cutoff `1e-16`, 16 MiB cache and continuous
crossover unchanged. Extend previously dense work below that crossover with
the native conservative pre-AO CSR profile, using a 64 MiB optional allowance.
The native path declines the whole inventory above mean AO occupancy 0.8.
Resource/capability misses retain dense execution; no labels are truncated.

GPU product names, SM-version windows and molecule/atom/AO/grid-size whitelists
are not admission criteria. Generic CUDA capability, derivative order, ordinary
all-electron non-DF RKS/UKS semantics, resident-grid availability, tile policy,
resource headroom, occupancy and continuous contraction cost are the guards.
Measured GPU names remain provenance, not claims of qualification on other GPUs.

The final comparison is five interleaved `current-default`/`auto` campaigns on
one Slurm allocation per workload, with all four complete E+F phases, independent
references, actual force work/Fock histories and separate intrusive profiles.
`current-default` keeps only the existing sampled profile; it does not impose
a dense override or enlarge that profile's 16 MiB allowance. `auto` uses the
public production registry without a producer override.

Native sparse extensions must reduce actual AO/jet/projection work and clear
both warm robust-noise gates without material regression in other phases.
Dense occupancy fallback and unchanged sampled dispatch must avoid material
regression, but do not need to manufacture a new warm gain. At least one sparse
extension must qualify before the portfolio verifier reports promotion eligible.
This honors the user's explicit authorization that stable warm gains suffice.

## Rejected alternatives

Do not replace sampled with native CSR merely because native beats dense.
Frozen 48-atom evidence recorded native warm 9.474753 s versus experimental
sampled warm 9.203358 s: native was about 2.95% slower. That experiment used a
64 MiB sampled allowance and predates the merged baseline, so it is mechanism
evidence rather than an exact comparison against #2007's production default.
The new portfolio comparison uses the actual 16 MiB incumbent.

Do not restore a GPU-product or atom-count guard to avoid qualification work.
Do not resume stable-owner grouping or point-parallel `geometry_point_setup`;
their negative evidence remains authoritative for this task.

## Evidence and provenance

Merged foundation revision: `336f4cf41`, based on `196126d88`, with explicit
dirty source reconstruction retained for the qualification harness.
Core SHA-256: `cbc3fae7362c2417d6b2117429541a54d501e113c3116fa4279ed59186061a78`.
The final stationary AOT manifests/packages are rebuilt against merged compiler
identity with ccache; prior frozen binaries are not relabelled as merged ones.
The [reviewed publication](../../../../benchmarks/results/preao-force-portfolio-20261006/README.md)
retains exact raw inputs/campaigns/profiles, source reconstruction patches,
numeric acceptance and separately verified warm default eligibility. Its control
keeps the incumbent profile on the same merged candidate source/binary; this
isolates dispatch rather than claiming a separately built unmodified-master
comparison.

### Final merged-source measurements

All five AB/BA campaigns per size run on n1 Slurm allocation 6217. Complete E+F
medians in seconds (current-default → public auto):

| Atoms | Cold | Warm | Moved | Moved-warm | Actual route |
| --- | --- | --- | --- | --- | --- |
| 3 | 69.836138 → 69.602569 | 0.273536 → 0.273797 | 1.930219 → 1.928621 | 0.269595 → 0.269247 | native inventory declines to dense identity |
| 24 | 43.144010 → 42.920474 | 3.970949 → 3.814492 | 24.388917 → 24.194730 | 4.009009 → 3.852154 | native selected |
| 48 | 98.047321 → 97.810097 | 8.949306 → 8.950215 | 51.472653 → 51.278235 | 8.959008 → 8.939420 | sampled preserved |

At 24 atoms, warm improves 3.9400% and moved-warm 3.9126%, both above the shared
2% robust-noise floor. Cold/moved gains are within noise; no significant cold
improvement is claimed. Force AO visits/jet values fall 24.6817%; projection
FMA pairs fall 38.5145%. Native discovery performs zero AO-jet evaluation,
zero Python AO-label lookups and zero per-tile label H2D; warm replay also has
zero discovery transfers. All five rounds' work counts, actual Fock histories,
reference errors and separate intrusive profiles are retained.

The 3-atom occupancy fallback and unchanged 48-atom sampled route avoid material
regression in all four phases. The latter preserves cutoff, cache allowance,
public policy and actual work exactly. The portfolio verifier reports default
promotion eligible, with maximum energy error 8.41283e-12 Eh and force error
2.69484e-11 Eh/Bohr across the final benchmark references. Measured scientific
source identity is `25e88a8a007bc1609c2bcdca5a4680182ca41ec9369abdeaba0e4b6b5de778a0`;
all recorded scientific/endpoint source hashes match the final checkout.

Additional merged gates: 26 CUDA cases, memcheck zero errors/racecheck zero
hazards, eight explicit native independent analytic cases and eight public
cases on 6217, plus ten public analytic force cases on 6218. Host validation:
970 passed/306 skipped, 473 compiler modules/zero dependency errors, Ruff and
clang-format checks, ccache-backed core/AOT builds and a no-work rebuild. Host
fixture corrections and documentation updates after freezing do not change
measured scientific/endpoint sources.

The publication accepts numerical gates and records guarded warm eligibility
separately. Generic all-phase performance status stays `not-run`; uncached
deployment compilation delta and observed global allocator peaks are not
invented from compilation success or numeric memory bounds.

### Completed historical mechanism experiment

The original frozen 48/96-atom A/B/C has five complete campaigns and separate
profiles, also retained in the publication with its own source reconstruction.
48-atom pairs use 6178; all 96-atom pairs use 6191, never mixing its earlier
unpaired 6178 attempts. Native warm improves over dense by 15.8625% / 33.5251%,
but regresses versus experimental 64 MiB sampled by 2.9489% / 4.8031% (both
exceed the 2% robust-noise floors). These are negative producer-comparison
receipts, not merged default qualification. They reinforce sampled preservation
instead of justifying an all-native replacement.

An initial staging allocation, 6216, passed 26 device tests, memcheck and
racecheck, but its later source-manifest check rejected two verifier/test files
changed during staging. It supplied no endpoint samples and is not qualification
evidence. The failed tree and receipts remain intact. A new frozen tree and
allocation 6217 perform the final campaign; scientific sources are not updated
in place while it runs.

Host validation includes the existing sampled map-owner tests: their fake owner
now models the added native constructor and producer cache binding, and the
AST gate requires selected feature leases to go through their bound map owner.
No real GPU is accessed by these host-only cache tests.

## Post-integration source-scoped revalidation

The initial publication above remains frozen evidence from the first merged
source; its binary is not relabelled as the later head. After merging master
`20087a073` in `aa79a3453` and retaining the reviewed indexed identity-map,
installed-header and detached-jet-layout repairs, the new campaign measures clean
revision `1886ca21637a244c544927d128bafaa9b28ab295` on n1 Slurm allocation 6244.
Core SHA-256 is
`1e56e3afee967f91b3a0188bb283753b442bb36fd6f13d3888707a194aa64044`;
scientific source identity is
`f0f179895e00b9a499eb75d2182fbb2cbb5ed9303d023963f57d31386c4bfa1e`.

Five interleaved AB/BA complete E+F campaigns at 24 atoms pass the same guarded
warm objective. Current-default versus public-auto medians, in seconds:

| Phase | Current-default | Auto | Improvement |
| --- | --- | --- | --- |
| Cold | 54.828060 | 54.762690 | 0.1192%, within noise |
| Warm | 3.999093 | 3.839689 | 3.9860%, above 2% robust-noise floor |
| Moved | 24.396644 | 24.181707 | 0.8810%, within noise |
| Moved-warm | 4.000244 | 3.840637 | 3.9899%, above 2% robust-noise floor |

Actual AO visits/jet values again fall 24.6817%, projection FMA pairs 38.5145%,
with zero native Python AO-label lookups, per-tile label H2D or discovery AO jets.
The 24-atom independent gates have maximum energy error 5.57066e-12 Eh and force
error 2.44519e-11 Eh/Bohr. The control remains incumbent-only dispatch on the same
rebuilt candidate source/binary, not a separately built unmodified master.
The new 3/48 coverage consists only of single numerical/work smoke pairs, not
new five-round no-regression timing qualification. The original full five-round
3/48 results retain their earlier source scope.

The additional [integration publication](../../../../benchmarks/results/preao-force-integration-20261006/README.md)
preserves all 19 original JSON byte streams, hashes, complete histories/counters,
independent references and separate intrusive profiles in one cross-record XZ
bundle. Compact summaries omit only duplicated payloads, not samples or precision.
Full source checks, scheduler/device, build/host/ccache and GPU gate receipts are
retained; numerical acceptance and warm eligibility remain distinct from generic
all-phase performance promotion. There is still no GPU-model/shape whitelist or
claim of timings on hardware other than the measured n1 RTX 5090.

The first integration attempt, 6240, passed 26 CUDA cases and zero-error
memcheck/racecheck but stopped at public analytic tests: library-only CMake
targets left stale stationary AOT manifests. The loader correctly rejected the
contract mismatch. No endpoints from that attempt are credited. Build the needed
`generativeqc_stationary_*_manifest` targets after compiler-input changes; they
also build the corresponding libraries. The rerun verifies all 14 used AOT
contract/plan/binary identities before dispatch, without rewriting identities or
bypassing the loader. Its 26 CUDA cases, two memcheck/two racecheck cases and ten
public analytic-force cases pass. Final-source host validation records 940 passes
and 305 explicit skips, with 20 additional packaging passes; the 35 focused
repair tests are a subset, not extra unique passes. The compiler audit checks
478 modules with zero dependency errors; scoped format checks and the ccache
build/no-work rebuild pass.

The subsequent `5d11d025` fast-forward changes only host tests and the capacity
qualification tool. All 16 measured production campaigns' scientific file hashes
still match; its six changed test files add a separately recorded 441 host passes
and one NVCC-unavailable skip, not a new GPU or performance campaign.

## Boundaries and follow-up

P0-A and the force-owner portion of P0-B are implemented. Separate native KS/XC
inventory sharing remains unimplemented. P0-D restricted-spin projected-panel
aliasing and P0-E sparse provider scheduling remain deferred; this change does
not claim all P0-A–E or close #1893.
