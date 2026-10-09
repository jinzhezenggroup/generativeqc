# Decision: share CPU HF d/f preparation in mixed high-l bases

Status: implemented
Date: 2026-10-09

## Problem

The CPU value-only geometry-reuse owner admitted only an entirely s/p/d basis.
Adding one f shell therefore moved every quartet back to the AO-first traversal,
including low-l components that still used the generated scalar DAG. Those
components unnecessarily rebuilt geometry, Boys values and Coulomb auxiliaries
for every primitive/component tuple. Spherical fallback values and physical-atom
derivatives additionally transformed all eight equivalent tensor orientations.

## Decision

Make generated value eligibility per shell quartet. Its existing fixed dddd
component buffer, scalar DAG and primitive-first traversal remain unchanged.
For f-containing value quartets, share the existing Hermite/Coulomb preparation
across components of each primitive tuple. Both schedules call the existing
`fill_hermite` and `fill_coulomb`; no recurrence is duplicated or replaced. The
scalar/derivative primitive remains unchanged, including its local workspace
lifetimes, monolithic Jet contraction and delayed radial prefactor. Only the
shared value owner returns workspaces in a prepared aggregate, then consumes
their FP64 values without empty derivative Jets. This value-only contraction
uses the original association/reduction order; it does not implement another
Hermite or Coulomb recurrence.
Independent axis ceilings cover each component's powers; the radial order stays
the sum of the four shell angular labels, not the sum of those axis ceilings.
The invocation-local component vector reserves at most 10,000 records, one
ffff block. It is released before spherical transformation. The public resource
inventory's conservative zero-derivative Jet recurrence allowance exceeds this
buffer plus the retained recurrence box; no resident/cache lease is introduced.

The higher-l/derivative AO pass skips exactly the value orbits owned by the two
shared producers. No new f/g recurrence or generated class inventory is added.
All-s/p/d spherical values retain their shell-local producer. Quartets containing
g+ and all derivatives retain their scalar traversal and prior mathematics.

Permit only symmetry-qualified integral producers to request canonical eightfold
spherical projection. This includes full/range values and physical-atom nuclear
derivatives, not derivatives of an individual ordered shell-center slot.
The generic `transform_integrals` adapter retains its ordered behavior even for
nonsymmetric supplied tensors. Each representative keeps the existing coefficient
association and reduction order; its scattered permutations can differ from
separately reduced outputs in the last floating-point bits.

This extends the mixed-basis admission of the
[shared-geometry decision](2026-10-01-cpu-eri-shell-geometry-reuse.md), not its
scientific ownership, fixed scratch or precision contract. CUDA dispatch is not
changed, including the deliberately unpromoted bounded through-f value policy.

## Work and invariants

For loaded water/def2-TZVP, the source-derived census is 18,145 canonical shell
quartets and 2,692,535 primitive-component evaluations. Of those, 1,438,385 belong
to s/p/d quartets: their common geometry evaluations fall to 139,616. The
1,254,150 f primitive-component evaluations are unchanged, but their Hermite/
Coulomb preparations fall to 17,324. Thus all primitive-component work remains
2,692,535 while common preparation is shared separately by its generated and
retained owners. These are
exact source-loop counts, not hardware/FLOP measurements or a speedup claim.

Normalization, contraction, eightfold component ownership, Cartesian/spherical
ordering, nuclear-force conventions, FP64 and scientific tolerances are unchanged.
Rank-four output storage and conservative Cartesian resource admission remain
unchanged. Generic AO enumeration still visits the fallback domain; this change
does not claim a new high-l algorithm or work-bounded scalable dense HF endpoint.

## Rejected alternatives

- Merely enabling low-l reuse in the mixed basis is insufficient: the initial
  paired water/TZVP pilot regressed from approximately 16--17 s to 18--19 s,
  despite reduced geometry counts and identical 14-iteration trajectories.
  The unpromoted pilot receipts are `paired-water-tzvp-*` in local evidence.
  Static preparation counts cannot justify production selection by themselves.
- Factoring the scalar/derivative path through out-of-line aggregate preparation
  and contraction raised the diagnostic complete HeH+ force endpoint from an
  earlier frozen baseline of about 15.5 to 19.4 s with identical eight-iteration
  branches. Inlining alone recovered only
  part of that regression (16.4--16.7 s). Restoring local workspace lifetimes
  and delayed prefactor while retaining the inline Jet contraction still
  regressed a fresh ABBA comparison: baseline medians 15.20--15.54 s,
  candidate 16.39--16.87 s. Preserve the entire original scalar/derivative
  primitive rather than assuming helper factoring is neutral. A prepared
  aggregate and value-only contraction are only required by the shared schedule.
- Function-entry alignment was also tried and discarded; it did not recover the
  force endpoint. No alignment or always-inline attributes remain. Function
  sizes and normalized instructions alone do not qualify complete endpoints.
  Moreover, comparisons against an older whole-HEAD binary confound other
  concurrent changes. Final qualification isolates this integral owner against
  the same surrounding objects rather than treating diagnostic pilot timings
  as promotion evidence.
- Emitting every f component would enlarge compilation inventory without being
  necessary to share the already-supported retained recurrence preparation.
- Globally symmetrizing the generic projection would corrupt caller-supplied
  nonsymmetric tensors. Symmetry admission must come from a known producer.
- Removing the high-l recurrence or relaxing force thresholds would conflate a
  scheduling optimization with unsupported mathematical/precision changes.
- Enabling bounded CUDA f values repeats an already rejected performance policy
  and is unrelated to this CPU bottleneck.

## Evidence

Final frozen-source qualification passes all 67 CPU native tests and 119 Python
endpoint, census, storage, projection, resource and range tests, with 14 explicitly
GPU-only skips. Earlier generated-component gates additionally include independent
Libcint checks. Added native gates cover contracted mixed s/p/d/f values in both
shell orders, complete ordered-versus-canonical tensors, physical-atom f
derivatives and arbitrary nonsymmetric adapter input. Independent public
PySCF 2.14.0 gates cover RHF/UHF, Cartesian/spherical, value-only and complete
energy/force calls at original and moved geometries without CPU/GPU substitution.

Endpoint timing qualification uses `benchmarks/cpu_hf_high_l.py`, verified ccache,
Release CPU libraries, one-thread oracle/runtime settings, fresh calculators,
unchanged equations and original/moved geometry. The benchmark retains source
work counts, loaded angular momenta, library/driver hashes, CPU affinity, thread
settings, actual executed backend, every time, SCF iterations and independent
energy/force errors. Energy-only measurements are not force-endpoint claims.
Local raw evidence is under `.artifacts/hf-df-optimization/`. Shared-f
qualification additionally includes a
complete distinct-center ffff block against the scalar recurrence; prior pilot
tests/timings do not substitute for final-source gates. A moved formaldehyde
PySCF preparation exhausted ordinary SCF; the harness now permits 100 iterations
and Newton refinement without changing oracle or endpoint tolerances, recording
that refinement separately outside native timing.

Final timings use AMD EPYC 7K62, CPU 6 affinity, GCC 11.4 Release, verified
ccache 4.5.1, CUDA disabled and one thread. Every sample reports
`executed_backend=cpu_reference`. The control is the HEAD integral owner linked
with exactly the candidate's other objects; its compile/link commands and cache
statistics are retained in `context-control-*`. This fixes the surrounding build
context without reverting parallel work. The retained scalar/derivative primitive
body is byte-for-byte identical to HEAD. Frozen libraries are
`context-baseline.so` and `localized-candidate.so`; `context-final-*` retains the
final numerical gates. Interleaved B/C/C/B endpoint receipts give:

| Complete endpoint, spherical | Control median | Candidate median | Samples per library |
| --- | ---: | ---: | ---: |
| Water/def2-TZVP, energy | 17.2878 s | 0.4340 s | 4, original and moved |
| Contracted HeH+ s/p/d/f, energy plus forces | 16.3390 s | 16.3083 s | 8, original and moved |

Water has 48 Cartesian/43 spherical AOs and one actual f shell. Its approximately
39.8x energy-endpoint speedup preserves all 14 SCF iterations and independent
errors below 1e-12 Eh. The force endpoint preserves eight iterations and errors
below 2e-10 Eh/Bohr; the overlapping timing ranges and 0.2% median difference
support non-regression, not a force-speedup claim. H2 remains approximately
0.7 ms and water/def2-SVP approximately 20--21 ms in both libraries.

The larger candidate formaldehyde/def2-TZVP endpoint has 84 Cartesian/74 spherical
AOs and takes 4.2868/4.3767 s at original/moved geometry with 16/17 iterations.
Its 25,188,300 primitive-component evaluations remain unchanged: 12,144,035
generated components share 884,267 preparations and 13,044,265 f components
share 141,931 retained preparations. Oracle energy errors stay below 1e-12 Eh.
This is larger-size complete-endpoint evidence, not a larger-case speedup claim;
no matched formaldehyde baseline was measured. The JSON receipts and
`qualification-summary.json` retain individual samples and provenance.

## Revisit when

Larger high-l endpoints show the retained contraction itself dominates. Extending
compiler-owned shared high-l work or the shared schedule to derivatives requires
its own source/resource,
independent numerical and complete-endpoint qualification, rather than expanding
this fixed low-l owner blindly. GPU d/f work needs separate Slurm qualification.
