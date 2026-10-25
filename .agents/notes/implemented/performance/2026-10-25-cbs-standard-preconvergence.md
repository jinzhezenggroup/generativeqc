# Independent CBS preconvergence under strict final controls

Status: implemented
Date: 2026-10-25
Base: `7b3dc64f9` (master)

Follow-up: [routine Standard](../numerics/2026-10-25-cbs-routine-standard.md)
supersedes the profile naming below: these strict stopping settings are now
Tight. This note and its linked source-matched measurements preserve the
earlier experiment; they are not new-default benchmark results.

## Decision

Separate the execution request from the correlated convergence allowance.
`CCSDTCBSCalculator(preconvergence="auto"|"fp64"|"mixed")` transports an
independent, private size-tagged selector. Zero budget fields retain strict
CCSD, Lambda and response controls. Legacy descriptors retain their previous
schedule; the frozen public C descriptor is unchanged. Prepared owners decode
the extension before copying that public prefix. Unsupported families reject
explicit requests, and batch owners cannot silently lose the extension.

Standard automatic fitted CUDA E+F execution requests bounded mixed warmup
at occupied/virtual dimensions of at least 9/48. Energy-only execution uses
the existing CCSD automatic domain, 9/128. Fast retains its prior mixed
request when relaxed convergence is admitted. Explicit `mixed` bypasses the
shape selection but never resource/provider admission or physical checks.
`policy="fp64"` retains the original strict FP64 grouped schedule. The HF/DFT
`PrecisionCalculator` alias is unchanged.

The separate execution capability lets automatic calls preserve the legacy
schedule on older libraries, with an observable compatibility reason. Explicit
requests fail if that capability is absent. Auto admission and executed
arithmetic come from copied native counters; refusal cannot be labelled mixed.

## Evidence and rejected alternatives

The [audit](../../proposed/2026-10-25-fast-standard-policy-separation.md) found
that HF/DFT warm endpoints frequently execute identical strict work, whereas
CBS coupled mixed warmup with relaxed stopping. The new four-cell experiment
crosses strict/relaxed convergence and FP64/automatic warmup in one binary.
The [retained measurements](../../../../benchmarks/results/cbs-standard-preconvergence-20261025/README.md)
include raw complete endpoints, all component energies/forces, work counters,
source reconstruction and an offline verifier.

| Full DF-CBS E+F | Strict FP64 (s) | New Standard (s) | Time reduction |
| --- | ---: | ---: | ---: |
| C1 | 3.280 | 3.298 | No demonstrated gain; auto keeps FP64 |
| C2 | 9.373 | 9.072 | 3.2% |
| C4 | 86.066 | 77.210 | 10.3% |
| C6 | 635.207 | 560.766 | 11.7% |

C1/C2/C4 use medians of two alternating same-card calls per arm; C6 has one
call per arm. Every measured endpoint starts a fresh calculator and returns
host E+F. Process/CUDA startup is separately retained, not compared with a
subsequent initialized process. Basis snapshots are outside the timed endpoint
and supply no SCF oracle seed. These are scoped observations, not a universal
or statistically certified speedup. Numeric capacity is not measured peak VRAM.

The first exploratory C1 FP64 call included about 1.6 s of process startup;
using it to claim a mixed speedup would be misleading. Subsequent C1 calls
showed no useful gain. Reject unconditional Standard mixed selection. Lambda
amortizes the extra strict epoch earlier than energy-only CCSD: C4 strict
CCSD iterations change 20 to 21, but Lambda actions change from 23 FP64 to
12 mixed plus 13 FP64, and Lambda time falls from about 40.5 to 32.4 s.
This motivates distinct force and energy admission domains rather than
loosening Standard's residuals. C6 energy-only calls also improve by about 7%.

Fast retains an additional measured advantage on C2/C4/C6 after the shared
optimization. For example C4 changes from new Standard 77.2 s to Fast 67.2 s.
There is no reason to manufacture a larger energy error to distinguish modes.
The final selected stopping policy, not the warmup dtype, owns the accuracy
concession.

## Scientific and resource boundaries

- Standard CBS CCSD energy/residual limits remain `1e-12`/`1e-10`; Lambda,
  reference and orbital-response controls are unchanged.
- The CCSD handoff still clears mixed DIIS history, and Lambda starts a fresh
  FP64 GMRES epoch. Independent expanded residual replays, transpose audits,
  triples and analytic response remain FP64. Finite warmup exhaustion does
  not publish convergence.
- Same-binary strict controls use `1e-9 Eh` energy and `1e-8 Eh/bohr` full-force
  parity gates. Fresh independent C2 PySCF component/composite energies and
  two directional derivatives use `1e-8 Eh` and `3e-6 Eh/bohr` gates. Water
  algebra tests cover all Cartesian force components, repeated and moved
  geometries, both HF reference models and energy-only calls. Their compact
  STO-3G/def2-SVP pair is not a physical CBS accuracy claim.
- 47 native/Python integration tests pass. Five existing CCSD/Lambda native
  probe contracts also pass, including actual optional-provider refusal and
  finite warmup exhaustion. Successful C2 1/2 GiB endpoints retain strict
  parity; smaller complete-budget failures remain in the receipt. They are
  not reported as successful FP64 fallback calculations.
- A clean master library with the new Python automatic caller preserves the
  old strict schedule. Its C2 energy matches the explicit candidate FP64
  control exactly, with force differences below `2e-14 Eh/bohr`.

## Build and publication lessons

All device execution used finite Slurm allocations on n1 RTX 5090 cards;
assigned visibility was preserved. The local n3 host is not a permitted GPU
worker, but remote n1/n4/n5 were accessible. This supersedes the audit's local
availability limitation. The native build used verified sccache 0.16.0 and
generated launcher commands were checked; cache statistics are retained.

The first incremental candidate missed changed files because a preserved
overlay mtime preceded objects built from the clean baseline. Capability
tests rejected that library before any timings were accepted. Extracting
the overlay with current mtimes rebuilt the affected owners; subsequent
libraries also require an execution-capability smoke test. Keep immutable
library copies and source patches for each candidate.

PySCF's default 4 GB process memory ceiling can fail after a CUDA/native
library has increased process RSS, even for H2. The combined oracle tests
ran in a 24 GiB Slurm allocation with `PYSCF_MAX_MEMORY=12000`; no numerical
or oracle convergence gates were weakened.

The HF/DFT homepage figures now show the entire Standard/reference cohort,
including accuracy failures, using the existing immutable measurements.
Fast remains in their detailed tables. Reproduce that view with
`benchmarks.readme_defaults_report`, selecting `standard.json.xz` and
`gpu4pyscf.json.xz` from `df-dft48-work-20261025`, its original oracle/reuse
manifest and `--include-unqualified`. Historical CBS curves remain accessible
under an explicitly dated fold; they must not imply current Standard execution.

## Revisit when

Broader basis, charge, strongly correlated or small occupied-space measurements
justify different automatic domains. Tune stopping budgets separately, with
independent observable gates. Do not merge final-force narrowing or relaxed
response into Standard on the strength of these warmup measurements.
