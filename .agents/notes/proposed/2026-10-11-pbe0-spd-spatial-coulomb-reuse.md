# Decision: reuse primitive-local order-six spatial Coulomb states

Status: proposed; independent GPU qualification and positive two-pair pilots, promotion pending
Date: 2026-10-11

The cleaned implementation and scoped automatic policy supersede this prototype
in the [implemented decision](../implemented/performance/2026-10-11-pbe0-spatial-force-reuse.md).
The frozen prototype, pilots and intermediate decisions below remain historical.

## Problem

The fused all-center order-six pair gradient already shares the primitive and
radial workspace. Its 64 pair-subset combinations still evaluate one base and
three raised spatial Coulomb states each: 256 scalar recurrence calls per
primitive. Many requested states are identical. More center fusion would not
remove this work because all centers are already accumulated together.

The [narrow-low scheduling prototype](../rejected/2026-10-11-pbe0-spd-narrow-low-force.md)
did not establish an endpoint improvement. This is a separate algebraic-work
experiment, not a relabeling of its rejected timing evidence.

## Decision

The compiler emits an optional primitive-local cache containing the Cartesian
state box and its three single-response faces. It evaluates each retained state
with the existing scalar recurrence and leaves the gradient accumulation order
unchanged. For total angular order six, this needs 22 to 54 scalar evaluations
and an enclosing allocation of at most 64 doubles. Multi-face corners are never
initialized or read. Radial moment validation still precedes cache preparation.

The first frozen prototype captures the explicit `spatial-reuse` force selector at preparation
and forwards it only into the existing narrow order-six drain. All spin/output
layouts, class/prefix/topology guards, screening and bounded page semantics remain.
The unset/default path remains the incumbent; `narrow-low` remains a separately
reachable uncached control. No retained quartet inventory, new traversal or
unaccounted device allocation is introduced.

## Invariants

- The compiler owns the mathematical lowering; native SCF owns dispatch policy.
- Scalar Coulomb algebra, radial moments and term accumulation are unchanged.
- Full-precision derivative acceptance requires an independent oracle.
- The 48/96-atom original/moved complete endpoints, density and semantic-work
  gates remain unchanged; retain failures and slow samples without pooling.
- Both endpoint arms explicitly enable resident final validation from #2220;
  its benefit must not be attributed to the force-state cache.
- Only qualified endpoint improvement can justify default promotion or a PR.

## Evidence

Local receipts are retained under `.artifacts/ks-weight-stage/spd-spatial-v1/`
in the original checkout. The initial focused run had a stale codegen string
expectation; its failed receipt is retained. After repairing that expectation,
all 12 focused tests pass. The formatted scoped suite passes 52 tests with
eight native/GPU-dependent cases skipped; those skips are not GPU evidence.
Compiler and shared-SCF structure checks report zero dependency errors.

The emitted cache executes against independent state inventories for every
axis partition at orders four through six. Tests verify capacities, exact call
counts, untouched multi-face corners and the uncached default. Instrumented real
scalar recurrence reduces the order-six primitive from 256 evaluations to
22/54 with identical gradient components. Both modes retain independent
Hermite-value finite-difference checks for full/LR gradients and invalid omega.

At the first source freeze, no linked device resource or complete endpoint
observation existed. The following follow-up receipts do not overwrite that
frozen source or the earlier rejected prototype.

### Source-matched GPU qualification and pilots

Production patch SHA256:
`f4679c68054a1d0250518e3751cedd21059716ccdef04e46bb86fc7a62c36cb6`.
Qualification patch SHA256:
`c9bb5b71135a793860001eda005fa598791186357eb3ba1eb9a2a2b608cfd268`.
Native library SHA256:
`61bf64c4d8483c51fa4cdafdbd2925ec5a5936e5a2441ad78e555005846c8c20`.
Frozen remote source: `candidate-pbe0-spd-spatial-v1` under
`/data/jzzeng/qc-next-hotspot-20261009-125a4e33f` on n1.

The cached CPU build 7312 succeeds. GPU job 7314 passes all 64 independent
arbitrary-density derivative calls and five independent FCI complete-force
cases, with zero skips. The preparation-captured selector survives later
environment mutation. Linked symbols authenticate all eight spin/output/reuse
variants. Qualification-only observer build 7313 succeeds; GPU admission 7317
passes 20 complete calls across both sizes and geometries. Each candidate call
uses the cached low, order-seven and order-eight consumers once each, with no
generic-low, uncached narrow-low, rejected-domain or wrong-width launch.
Instrumented admission is not performance evidence.

Static final-linked cubin inspection reports REG255, SHARED4173 and LOCAL0
for all low variants. Cached STACK fields increase by 576 to 608 bytes over
the uncached narrow-low control; LOCAL0 does not prove spill-free execution,
and these reports are not a runtime private-memory peak or a speed claim.

Clean GPU jobs 7318/7319 use no launch observer, one native image and one CUDA
context with two independently converged resident owners. Each population has
18 independently gated complete E/F calls and eight timed samples, including
original and moved geometry. All numerical, immutable-owner and available
semantic-work/resource gates pass. Capacity counters are not executed screened
quartet counts.

| Atoms | Geometry | Baseline median (s) | Cached median (s) | Raw reduction |
| --- | --- | ---: | ---: | ---: |
| 48 | warm | 4.841818370 | 4.618702231 | 4.608% |
| 48 | moved-warm | 4.874624925 | 4.595880434 | 5.718% |
| 96 | warm | 14.096563727 | 13.373456761 | 5.130% |
| 96 | moved-warm | 14.215549545 | 13.400297558 | 5.735% |

Maximum independent E/F errors are 7.048584e-12 Eh / 2.465006e-11 Eh/Bohr
at 48 atoms and 9.549694e-12 / 3.760614e-11 at 96 atoms. Maximum cross-arm
density drift is 6.251666e-13 and 2.165434e-12 respectively, below 2e-11.
These successful populations neither explain nor erase the earlier narrow-low
96-atom density failure. Different cohorts are not pooled.

Both pilots are positive but have only two interleaved pairs per geometry.
They do not authorize a performance PR or default promotion. Clean the
committable patch, then qualify five pairs and scoped safety on that source.
All raw, failed-command, source/artifact and analysis receipts remain in the
new ignored artifact/evidence directories. The first numerical submission
used an unsupported srun option and created no job; its failure is retained.
Slurm accounting is disabled here, so completion is authenticated by actual
srun driver exits, job-local exit/termination traps and terminal queue state,
not an invented accounting result.

Master was rechecked at `637466c62`; the new CC-triples change does not touch
these force/compiler paths and does not trigger blanket retesting.

## Shipping cleanup and automatic-policy candidate

The new `spd-spatial-shipping-v1` cohort starts from master `637466c62`;
it must not borrow qualification from the frozen prototype above. The cleaned
native adapter removes the unmerged class-page, indexed-domain and narrow-low
selectors. Only four spin/output variants of a cache-only low kernel remain;
the original mixed-f domains and generic/order-seven/eight consumers are kept.

The prepared force owner captures unset/empty as `Automatic`, explicit `0` or
unknown values as incumbent, and `1`/`spatial-reuse` as a guarded request.
Only authenticated DFT callbacks can supply automatic eligibility: full-precision
exact-direct PBE0/RKS, at least 384 public AOs, no fitted, range-separated,
nonlocal, D4 or ECP work. The composition matcher uses the restricted native
exchange coefficient `-0.125`, not the positive physical hybrid fraction.
Both callbacks validate their final-density token before forwarding eligibility.
Generic/HF/RSH consumers default the hint to false. Environment edits after
preparation cannot change the captured preference.

The initial cleanup host run failed because source-extracted fixtures retained
an old aggregate layout, CUDA status inventory, removed indexed classifier and
forwarding string. Failed receipts remain separate. Corrected focused probes
pass 528 cases, including executable production policy and prepared-preference
checks; the final formatted adjacent suite passes 561. At that host freeze no
cleaned-source GPU result, complete-endpoint result, default promotion or PR
was claimed. Qualification must compare explicit `0` against a genuinely
unset candidate, preserve both sizes/geometries, and use five interleaved pairs.

The cleaned production patch is
`7cfa7717e6682b5ccc95fa8d494d4a0e052b4258f1b33ba37341477618765f44`;
qualification patch is
`d8a05faeb7a258cc9ad3a33478c7d8be14dcba3c458bc800bfefd5c697752cf4`.
Cached build 7321 succeeds with native SHA256
`47209e21c0fb92038475c0f7705f3540d0ea06bc77c0f3afab4201bfd3b26a49`.
GPU job 7323 passes 64 independent arbitrary-density derivative calls and all
five independent FCI complete-force cases, with zero skips. The final linked
image contains exactly the four intended spin/output cache-only variants.
They report REG255, SHARED4173, LOCAL0 and STACK16824--25512 bytes. These are
static final-linked resource observations, not runtime memory-peak, spill-free
or speed evidence. Admission, sanitizer and five-pair endpoint qualification
remain pending; these numerical results alone do not authorize promotion.

## Consequences and promotion gates

Dynamic private indexing can spill the additional 64-double storage. Reduced
work therefore does not imply lower runtime. Preserve linked resource evidence
and qualify independent source derivatives, FCI forces and actual route admission
before clean two-pair pilots at both sizes/geometries. A positive pilot alone is
not sufficient: promotion needs five interleaved pairs and the unchanged gate
`max(2%, 2*(relative_MAD_baseline + relative_MAD_candidate))`, complete E/F gates
of 1e-8 Eh / 1e-7 Eh/Bohr, and cross-arm density drift no greater than 2e-11.
Do not relax a gate or repeatedly reseed to obtain a favorable result.

### Shipping safety and output-layout coverage

Full racecheck job 7325 timed out at its finite 15-minute limit, with scheduler
and driver exit 143. Its completed all-kernel memcheck passes all 64 source-major
oracle calls with zero errors. The full racecheck is not qualified or repeated.

Filtered racecheck job 7330 keeps all 64 source-major oracle calls and checks only
the new low kernel. Its tool reports zero hazards, and a qualification-only
observer records `[0, 24, 0, 24]` actual cached spin/output launches. The wrapper
fails its deliberately stronger four-variant assertion: this exposed that the
original arbitrary-density probe covers separate publication only. That failed
receipt is retained, not relabeled as a successful whole-wrapper qualification.

The additional combined-layout oracle calls the same native library with
`separate_sources=false` and independently contracts CPU ERIs using both terms.
It retains all spin, opposing-density, coefficient-mask, geometry, Cartesian/
spherical and exact-zero gates. Cached CPU build 7332 builds only that test
adapter; no production source or native library is changed or rebuilt.
GPU job 7334 passes all 64 additional combined calls under full memcheck and
filtered racecheck. Both tools report zero errors/hazards. Its observer records
`[24, 0, 24, 0]`, with zero wrong widths or unrecognized variants. Together the
two disjoint layout probes non-vacuously check all four cached consumers;
filtering avoids unmodified producer/other-order shared-memory work, not a
scientific or performance gate. Both observer images remain outside clean timing.

Dependency jobs 7327/7328 and performance preflight 7331 start no timed work
because their safety prerequisites fail. Their driver failures remain retained;
there are no samples to discard or pool. The new five-pair cohort uses fresh
`performance-v3-48/96` directories. Job 7339 completes 48 atoms: warm/moved
reductions are 6.0436%/4.3140%, both passing the unchanged robust gate, and all
30 complete numerical/work/checkpoint checks pass. Maximum E/F errors are
6.366463e-12 / 2.474398e-11; density drift is 6.978307e-13. Job 7342 is still
measuring 96 atoms. No stage PR or default promotion is authorized by the
completed 48-atom cohort alone.

### Final shipping qualification

Job 7342 subsequently completes all 30 numerical calls and 20 timed samples at
96 atoms. Warm/moved-warm medians are 14.104704417/13.352007795 and
14.248137016/13.543867871 seconds for baseline/candidate: reductions of
5.3365%/4.9429%, passing gates of 2%/3.1329%. Maximum independent E/F errors
are 1.136868e-11 Eh / 3.752010e-11 Eh/Bohr; maximum cross-arm density drift
is 1.840028e-12. Immutable checkpoints and available semantic work match.
Both completed sizes use the same frozen native image, with genuinely unset
candidate selection and explicit baseline `0`; #2220 remains `1` in both arms.

All four five-pair scoped assessments now pass without deleting or pooling any
sample. This changes the shipping decision, not the earlier two-pair pilots or
rejected scheduler evidence. The new publication lives at
`benchmarks/results/pbe0-spd-spatial-reuse-20261011/`; its numerical envelope
and scoped performance decision remain distinct from unmeasured broad formal
performance/production qualification. No cold-SCF, larger-system or other-GPU
timing claim follows from these observations.
