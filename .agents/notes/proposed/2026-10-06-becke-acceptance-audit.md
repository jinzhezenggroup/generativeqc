# Audit: complete generated Becke derivative acceptance

Status: proposed; final paired observations and completion review pending
Date: 2026-10-06
Related: #1894; draft PR #1996

## Scope and requirements

Audit the actual issue and its 2026-10-05 profiling boundary, not only a green
test count. The goal is canonical normalized-product AD recognition, generated
CUDA lowering, independent fixed-grid/physical forces, complete work/provenance
and direct comparison with the retained phased route. A negative experiment is
not a promotion; optional execution and the generic/phased fallback remain.

| Requirement | Current evidence and boundary |
| --- | --- |
| Scientific expression → AD → whole-graph operation → generated CUDA | `grid_partition_ir.py`, `becke_partition.py` and `grid_native.py` authenticate both reachable primal/JVP roots of the actual active-prefix family 1–128. Scalar companion identities alone are not the production witness. Foreign/changed roots, wrong bounds and adjusted partition graphs are rejected. Native owns execution, not another scientific formula. |
| Independent fixed-grid derivative values | Current host audit: 687 passed. It includes exact versus mixed-prefix graphs, changed metadata/roots, scalar/coefficient consumers, zero/saturation/nonfinite semantics, translation/permutation, geometry rebinding and 60-digit independent direct-product Decimal differences at actual 48/96/128 atoms. Iterations 1/3/5 remain separately checked. |
| RKS/UKS and changed geometry | Six method-label/spin gates pass, with alias cases explicitly not new functionals. PBE0 gradients use the returned native state plus reconverged FD/moved-fresh checks. Two converged nonlocal RKS/UKS cases have independent GPU4PySCF cold/moved vectors, warm replay and all FD steps; both actual source owners select. Failed isolated-H controls remain excluded SCF preconditions. |
| Deterministic scatter, bounded memory and invalidation | Ordered ascending-neighbor gather uses no floating atomics. Planner and native admission preserve concurrent reservations, cached geometry and actual kernel caps; composite planning charges both owners. Native tests exercise tails, external sources, changed/restored centers and transactional invalid-point publication/recovery. |
| Actual admitted atom boundary | New n1 job 6203 verifies the 128-atom native cap: 48 selected synthetic tests, zero skips, and memcheck/racecheck/initcheck/synccheck each pass. This adds backend-boundary coverage, not molecular cases. All 585 scientific/compiler inputs match the previously compiled standalone owner; its exact binary is reused. |
| Pair-production/visits, bytes, launches and phases | Seventeen dense-domain counters and seven separate intrusive intervals are retained. Physical clean/profiler observations are kept separate. Logical panel bytes are not hardware transactions; branch-dependent log/sqrt counts and privileged hardware profiling remain unavailable, not fabricated. |
| Direct phased comparison and losing routes | Current #1830 route including later log scheduling and candidate share frozen science/core inputs. Existing indexed/dense losing publications remain unchanged, primitive default stays off. Unchanged old log traversal, xyz-cache, broad-zero and losing fusion are not reopened. |
| Complete 48/96 PBE0 endpoints | Primary job 6173 and separate baseline profiles 6184/6193 retain cold/five warm/moved/five moved-warm, independent vectors, all actual solver histories, owner preparation, seven diagnostic phases and all raw samples. Grouped arms cannot establish paired significance. No normalization of different cold/moved work. |
| All compatible DFT consumers, no method-name specialization | Matching binds canonical math and dimensions, never a functional/molecule name. Ordinary and composite owners share the policy/primitive; label-independent PBE0, both spins and nonlocal source ownership are exercised. Unsupported graphs retain their existing bounded ownership; this does not invent a heteronuclear-adjusted implementation. |

The current actual graph identity is
`389ba8af8fc7ce291546a4b52e6ceb7c4c45b0f9d2f7d674a2147e12ed97dd9e`,
and operation identity is
`54a2495eba36e0cf9ae296828afcaf58bf0dee24844bcf08a02cee59c977bf32`.
The 635,655-node construction/recognition cost is host compiler work, not a
kernel speedup. Keep source/graph/emission/native evidence distinct.

## New reviewed boundary evidence

`benchmarks/results/becke-native-cap-20261006/publication.json` retains 96 route
observations from the 48 cap tests and all four sanitizer receipts. Recorded
generic-native maximum absolute/scaled errors are zero. That comparator is not
an independent physical oracle, and a scalar statistic's RMS is not a force
vector RMS. The numerical decision remains inconclusive for full acceptance.
The unchanged binary is
`cf82d6ecd5e973c38fd2f30f54fc04ebde00dd0fe5bd832ff558f7715faa6bfb`;
the earlier `becke-native-phases-20261006` publication retains its actual
generated-source/header/compiler/ccache provenance. No new compile duration or
individual owner cache hits are claimed.

The current host audit receipts are in
`.artifacts/issue1894/final-audit-20261006/`; compiler invocations use verified
ccache with before/after aggregate statistics. Source restoration must verify
the cap publication's scientific/test manifest and dirty patch, not just its
revision label. The new reproduction shell is syntax-checked, not separately
rerun as an additional campaign.

## Remaining paired comparison

The primary grouped endpoints show no established material bottleneck reduction.
Do not call them statistically interleaved. Supplemental job 6204, finite
`main/gpu:5090:1` on n1, is collecting five genuine balanced A/B pairs for each
original-warm and moved-warm endpoint at 48 and 96. Its result is not yet a
completion or promotion proof. Retain all measured calls, actual work and
independent errors; do not discard unfavorable samples.

Two simultaneous large SCF owners may exceed the card's memory. Instead keep
one public native prepared SCF batch and freeze its engine-local post-cold or
post-move density with `set_warm_start_updates(False)`. This contract is checked
in the public API, DFT method and CUDA KS owner. No reference seed is injected.
Close the **entire private gradient executor**, clear its batch reference, set
the next opt-in policy and perform an untimed complete E+F prime. This follows
the batch's existing whole-executor replacement pattern and legally creates a
new native configuration before topology; never mutate a configured native
owner or bypass its once-before-topology guard.

Then time the complete public E+F replay with profiling off. All priming,
construction and setup work is retained separately; it is neither subtracted
from nor normalized into a cold/moved performance claim. The supplemental
shared-state warm experiment does not replace the complete primary protocol.
Native retained dm0 has no independently recorded byte digest; report its
identity at the inspected frozen-snapshot contract/single-batch scope, not as a
measured checksum of opaque native storage.

Feasibility job 6202 finishes with zero exit and validates four timed calls,
four separate primes and both geometries. Every replay actually performs one
iteration/Fock build, uses its warm start without fallback, selects the expected
off/on route, and passes independent vector gates. One pair per geometry is
feasibility only, not significance. Earlier 6200/6201 harness attempts fail on
metadata/private compatibility-view names; retain their frozen inputs and
terminal failures, not successful paired qualifications. They do not justify
changing scientific code, source guards or tolerances.

## Completion boundary

Inspect the actual terminal paired output and PR/CI state, validate/publish the
full raw evidence, then repeat the issue's requirement-by-requirement audit.
Until that evidence is present, keep the goal active, PR draft and primitive
default off. Even an admissible timing assessment is not default-promotion
authority without accuracy, compilation and memory gates. Keep #1892 Direct and
#1893 AO/XC ownership separate from this Becke result. No merge or release is
authorized by this audit.
