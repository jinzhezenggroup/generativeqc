# Decision: qualify OMol25-level DFT with the README HF endpoint protocol

Status: implemented
Date: 2026-10-01

## Problem

The README HF comparison used def2-SVP; an OMol25-level comparison must use the
actual ωB97M-V/def2-TZVPD model, retain diffuse functions, and include VV10 and
analytic forces. Relabeling the earlier WB97M-V/def2-SVP benchmark would change
neither its model nor its incomplete evidence. Different integration grids can
also make a speed comparison fail independent force acceptance.

## Decision

Add a separate direct-only benchmark on the exact nested HF water geometries.
Both engines consume one canonical offline H/O basis and the existing moving
WB97M-V quadrature adapter. Keep HF's five frozen-density repeats, original and
moved endpoints, independent all-call energy/force gates, and unnormalized wall
times. Run the two engines sequentially in separate processes through Slurm.
Native WB97M-V density-fitted forces are outside the current contract, so this
does not create or imply a DF implementation.

The actual def2-TZVPD oxygen basis contains an f shell. At the recorded master,
the composed WB97M-V CUDA force owner qualifies only s/p/d, and calculator
capabilities for all six sizes expose energy only. Preserve this result as an
explicit native-unavailable comparison rather than dropping the f shell or
pretending an energy-only call satisfies the force contract. Native capability
preflight must run before preparation/SCF, and admitted future execution must
explicitly request both properties.
The 1856-AO endpoint additionally exceeds the present 1024-AO stationary-force
shape bound; qualifying f shells alone would not admit the entire size matrix.

PR #1637 (`6795097b003a30f9f5e9ab8c34e0705dddbf2427`) removes the internal
stationary geometry f-shell restriction, but leaves the public calculator's
named/local basis capability predicate unchanged. Rebuilding and probing all
six sizes still reports energy only; explicit local def2-TZVPD and named
def2-TZVP force requests are rejected before SCF. Its new opt-in def2-TZVP CUDA
test independently fails at the capability assertion. The benchmark now trusts
advertised force capability instead of duplicating a static SPD gate, so a
correctly qualified future f-shell candidate can proceed without a benchmark
patch. No production source is patched just to bypass this rejection.

Retain every point's process outcome and journal its active phase before the
potentially expensive endpoint. A finite whole-point timeout is not an observed
warm latency or a lower bound on it. Publish only complete, independently
qualified native points, preserving reference-only evidence separately. Medians
include all five repeats even when SCF iteration branches differ.

## Rejected alternatives

- Borrowing def2-SVP timings or substituting def2-TZVP loses OMol25's diffuse
  orbital basis.
- Borrowing reference energies, densities or derivative tensors in the native
  endpoint would measure an oracle-assisted path rather than native execution.
- Extrapolating larger sizes, treating timeouts as endpoint bounds, or choosing
  matching-iteration repeats would hide missing/inaccurate scientific work.
- CuPy 14.2 environment probes used GPU4PySCF's fallback tensor engine. Exclude
  those probes from the retained comparison and pin CuPy 13.6.0/cuTENSOR 2.2.0;
  record the actually selected tensor engine, not merely the installed package.
- An initial default-property native execution took 602.84 seconds, converged,
  and returned success, but had no forces. Its full-endpoint gate rejected it.
  Defaults follow capabilities, so status/convergence alone do not establish
  endpoint completeness. Capability preflight avoids repeating this expensive
  energy-only diagnosis at each size.

## Invariants

The plot is a comparison of the OMol25 functional/basis on HF water clusters,
not an OMol25 dataset-throughput, ORCA-grid or method-convergence claim. Keep
explicit shared quadrature, self-consistent VV10, full moving-grid response,
canonical basis identity, source/binary hashes, semantic work, all-call gates,
and incomplete-point labels. Do not modify production numerical/resource
defaults to improve the figure.

## Evidence and revisit conditions

The executable contract is in `benchmarks/readme_omol25.py`; the reducer
`tools/render_omol25_benchmarks.py` rechecks every raw endpoint before retaining
scalar samples. CPU-only tests cover lost diffuse shells, AO counts, geometry
movement, every-repeat acceptance, duplicate/missing samples, raw-reference
hashes, timeout exclusion and variable-iteration medians.

See `benchmarks/results/omol25-wb97mv-20261001/README.md` and its per-size JSON
for measured results and reproducible Slurm commands. Revisit when a qualified
WB97M-V DF force implementation or a common converged production quadrature is
available; new settings require fresh independent evidence, not relabeling this
historical snapshot.

The public-admission diagnosis above records the unmodified PR. The subsequent
[through-f endpoint correction](../compatibility/2026-10-01-wb97mv-through-f-public-forces.md)
preserves the two downstream defects and the locally patched candidate's
qualification; it does not retroactively qualify the original PR probes.

The initial reference-only curve also used GPU4PySCF's default 1e-10 VV10
density mask, rather than native MolecularV1's 1e-8 mask. The
[matched-domain correction](../numerics/2026-10-01-matched-vv10-comparator-domain.md)
supersedes that reference timing snapshot; the revised benchmark records both
density and weight screening and requires fresh reference evidence.

## Appended HF host-output timing scope

The first matched-domain matrix passes all numerical gates on the complete
three-atom native point, but inspection finds a clock-scope mismatch: HF's
reference timer includes `cp.asnumpy(-gradient.kernel())`, whereas the first
OMol25 reference timer stops after kernel synchronization and exports forces
afterwards. Native already returns host force arrays inside its public call.
The force array is small, but host return can also synchronize pending work;
an HF-equivalent complete-endpoint claim must include it rather than infer its
latency from byte size or subtract an estimated transfer cost.

Schema v3 records `force_return = host_array_in_timed_endpoint`, converts the
reference force array before stopping the timer, and rejects old timing schemas
in the reducer. A source-structure regression protects this ordering; missing
host-export protocol and old-schema fixtures are rejected independently. Both
engines rerun in fresh processes; no old reference timing or paired native point
is transplanted into the new matrix. The pre-correction matrix remains ignored
under `.artifacts/omol25-pre-host-export-correction/` as historical diagnosis.
This clock correction changes no production source, numerical acceptance gate,
basis/grid/domain or native-force qualification outcome.
