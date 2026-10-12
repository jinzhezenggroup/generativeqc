# Automatic Libxc capability claims

The bulk Libxc importer has a machine-readable qualification layer in
`generativeqc_compiler.xc.libxc_bulk_capabilities`.

It separates **representation** from optional **runtime qualification**.
User-facing automatic semilocal admission is intentionally a different policy:
supported imported registrations are default-allow. The negative exception hook
is intentionally empty today. Shared boundary classes such as zero-spin channels,
zero gradients, density tails, and work-domain regularization are generic runtime
semantics and must not be encoded as per-functional blacklist entries. Only a
reproducible defect unique to one functional, with no generic capability/domain
rule available, belongs in that hook. Every imported registration starts with two
intrinsic claims:

- `graph-imported`: the pinned Libxc Maple owner lowers to the canonical Graph;
- `pointwise-validated`: polarized and unpolarized energy, `vxc`, and packed
  `fxc` agree with independent PySCF 2.14.0 / Libxc 7.0.0 fixtures on
  `libxc-bulk-interior/v1`.

C and CUDA source emitters are also available, but source emission is not
compilation or runtime evidence.

## Physical boundary predicates

Before differentiation, the shared physical-coordinate adapter compares imported
`tau_W/tau <= 1` branches as `sigma_total <= 8*(rho_total*tau_total)`.
This avoids fractional-density round trips selecting the saturated branch at an
exact physical rank-one state. Branch values and their raw Libxc derivatives
remain unchanged, including empty/minority-spin work coordinates. If the
physical density-times-tau product is at or below the binary64 normal minimum,
the original normalized comparison is retained to avoid a new underflow branch.
Other comparisons, worker floors, and acceptance tolerances are unchanged.

The same canonical Graph feeds CPU and CUDA source emission. The exact-input
full-coefficient and inward positive-semidefinite density-direction regressions
are in `tests/python/test_libxc_physical_kinetic_predicate.py`; the rationale and
rejected whole-ratio rewrite are recorded in the
[physical kinetic predicate decision](../../.agents/notes/implemented/numerics/2026-10-11-physical-kinetic-predicates.md).

## Qualification evidence is not an admission whitelist

Higher stages are optional regression/qualification claims. They are computed
from identity-bound evidence envelopes and a fail-closed dependency DAG, but
they do **not** decide whether an otherwise structurally supported automatic
semilocal functional is user-accessible. The public resolver checks imported
Graph support, required ingredients, backend support, and the explicit negative
exception map instead. That map is currently empty; it is not a catalog of
qualification failures. New representable registrations therefore do not need a
positive per-functional admission record.

The evidence DAG remains useful for CI diagnostics and for proving stronger
claims:

```text
graph-imported
  -> pointwise-validated
       |-> compiled-cpu -----------|
       |-> compiled-cuda -> gpu-runtime
       |-> production-domain -----|-> molecular-scf
                                      |-> forces
                                      |-> response
                                      `-> public-method
```

`molecular-scf` requires `production-domain` plus either a qualified CPU
binary or a qualified CUDA runtime. `forces` and `response` are independent
branches. A public energy method therefore does not incorrectly require
response support.

Each non-intrinsic pass must use
`generativeqc.libxc-bulk-stage-evidence.v1`, bind to the exact capability identity,
name its stage, and point to a non-empty retained evidence reference. The subject
binds parameter bindings, pinned upstream/source digests, and the compiler source
inventory under stable logical paths. A source or binding change invalidates old
evidence; bulk queries hash the source inventory once and do not lower all graphs. Missing,
failed, malformed, cross-functional, or out-of-order evidence never promotes a
stage.

A `production-domain` pass has one additional hard requirement. It must attach
the exact `generativeqc.libxc-production-domain-profile.v3` qualification payload
computed from the registration family and required ingredients. The current
`semilocal-boundary-matrix/v3` profile requires both spin layouts and
first-order energy/vxc coverage. Density, gradient, tau, and control cases apply
to both layouts, while alpha/beta zero-spin and full-polarization cases apply
only to the polarized layout. The exact cases-by-spin matrix is part of the
profile identity, so dropping or moving a case, changing the matrix version, or
changing an ingredient invalidates the admission proof. Registrations requiring
ingredients outside the current generic `rho/sigma/tau` domain (for example,
Laplacian-dependent meta-GGAs) carry an explicit structural blocker and do not
appear as `production-domain` ready.

This first-order scope is deliberate. The intrinsic `pointwise-validated`
claim still includes packed fxc on the audited interior domain, but exact
vacuum/full-spin endpoints need not possess a finite full feature Hessian.
Production energy, SCF, public-method, and stationary first-gradient admission
therefore do not infer response capability. Endpoint fxc/CPKS qualification
remains an independent `response` stage with its own evidence.

## Query

```python
from generativeqc_compiler.xc.libxc_bulk_capabilities import (
    STAGE_EVIDENCE_SCHEMA,
    claimable_functionals,
    functional_capability,
)

base = functional_capability("GGA_X_PBE_SOL")
compiled_cpu = {
    "schema": STAGE_EVIDENCE_SCHEMA,
    "subject_identity": base.identity,
    "stage": "compiled-cpu",
    "status": "pass",
    "reason": None,
    "evidence": "ci://xc/GGA_X_PBE_SOL/compiled-cpu.json",
}
qualified = functional_capability(
    base.name,
    evidence={"compiled-cpu": compiled_cpu},
)

print(qualified.qualified_stages)
print(qualified.ready_stages)
```

With no attached higher-stage evidence, the current 221-registration inventory
remains exactly `pointwise-validated`. Queries for known higher stages return
an empty inventory rather than manufacturing a claim; unknown stage names still
fail explicitly.

`ready_stages` is the machine-readable qualification frontier. CI producers
can use it to decide which evidence jobs are meaningful next, while the registry
itself remains a pure, deterministic admission evaluator.

## Compiled CPU point evidence

`tools/qualify_libxc_compiled_cpu.py` produces the concrete
`compiled-cpu` evidence used by the automatic semilocal path. It builds the
polarized first-order Graph selected by the shared `production_candidate_domain`
recipe and binds it to the native `SemilocalPointProgram` ABI. Density-only LDA
uses `libxc-7.0/work-density-only-v1` (native domain version 5); other supported
ingredients retain `libxc-bulk-production-candidate/v2` (version 3). The wrapper
compiles the recipe's total-density screening before executing the Graph, then
the tool compiles and executes one C++ translation unit.

A pass binds all of the following into
`generativeqc.libxc-compiled-cpu-result/v2`:

- exact functional capability identity;
- exact point binding and pinned density threshold: binding v4 for legacy
  domains, v5 with a separately identified restricted companion for density work;
- point-expression and emitted-artifact identities;
- compiler executable hash and version;
- generated translation-unit hash;
- compiled executable hash; and
- deterministic **interior and vacuum** native smoke cases plus their
  expected/observed output vectors.

The vacuum smoke requires the compiled wrapper to return the same exact zero
`SemilocalPointValue` as the v2 array-Graph candidate when total density is
screened. This prevents a receipt from claiming the v2 domain while executing a
v1 binary that lacks the outer Libxc density boundary.

The smoke comparison is an **artifact execution** check against the exact bound
Graph. It is not the independent scientific oracle: boundary correctness remains
owned by the production-domain Libxc campaign. Conversely, successful
production-domain evidence is not proof that a C++ artifact compiled or ran.

Compilation failure, unavailable compiler, native metadata mismatch, nonfinite
output, or numerical mismatch remains explicit fail/not-run stage evidence.
Passing `compiled-cpu` alone grants neither production-domain, molecular-SCF,
force, response, nor public-method capability.

The generic bulk KS resolver does not trust the stage label alone. Before it can
construct the qualification candidate it revalidates the exact compiled-CPU
qualification and retains that receipt's point-binding identity and compiled
result identity in `BulkKsResolution`. This keeps MethodIR/KS composition tied
to the v2/version-3 executable that the evidence producer actually compiled.

`generativeqc_compiler.xc.molecular_scf_evidence` defines the next promotion
boundary. One `molecular-scf` pass requires both CPU RKS/UKS layouts and
cold, warm-replay, and changed-geometry rows for each. Passing rows bind fixture,
geometry, independent-reference and exact KS-resolution identities, require
convergence plus independent energy and physical-residual gates, and reject a
warm replay that changes geometry or a changed-geometry row that reuses the cold
geometry. Version-2 receipts additionally require a complete native value-provider
work census in every passing row: positive observed Coulomb applications and zero
exchange applications for these pure semilocal endpoints. Counts come from the
executed providers, not iterations or the requested MethodIR. Unknown work stays
unavailable; failed receipts retain incorrect or unavailable work without promoting
it. Legacy version-1 receipts do not satisfy the current public-admission producer.
This narrow census does not certify precision, preparation, or derivative work.
Only an all-pass dual-spin receipt emits CPU energy endpoint coverage;
forces, response and public admission remain separate stages.

## Bulk inventory queries

```python
evidence = {
    "GGA_X_PBE_SOL": {
        "compiled-cpu": compiled_cpu,
    }
}

assert claimable_functionals("compiled-cpu", evidence) == ("GGA_X_PBE_SOL",)
assert claimable_functionals("compiled-cuda", evidence) == ()
```

The evidence inventory is keyed by Libxc registration. Unknown or blocked
registrations are rejected rather than silently ignored.

## Scope

This registry still does **not** generate or mutate the public method manifest.
A semilocal component may be runtime-qualified without being a complete DFT
method. Hybrid/range-separated/nonlocal/correction primitives must be qualified
at the MethodIR/provider layer before a method-level admission producer may
attach `public-method` evidence.

The existing bulk numerical test continues to cover every intrinsic pointwise
claim. The promotion tests additionally enforce that stages cannot jump their
prerequisites, CPU and CUDA runtime branches are alternatives for SCF execution,
and public admission never appears merely because source code exists.

## Boundary qualification

`generativeqc_compiler.xc.boundary` owns a deterministic physical probe suite for
zero-spin, near-zero-spin, zero-gradient and tail limits.  The probe
coordinates are compiler inputs only.  Independent expected values are
generated by `tools/generate_libxc_boundary_reference.py`, which calls the
Libxc 7.0.0 C API through PySCF and never imports a GenerativeQC Graph or generated
kernel.

Boundary admission is intentionally split into two facts:

1. the generated expression stays finite at the probe;
2. energy and physical-feature derivatives agree with the independent oracle
   over the exact versioned production-domain profile.

Only a retained evidence record covering the **entire required profile** may
satisfy the `production-domain` stage. A finite value by itself is not a
correctness claim. Oracle-nonfinite points remain unqualified instead of being
coerced into a pass.

The receipt additionally binds the exact first-order execution programs used by
the campaign for both spin layouts: executor kind, runtime domain, imported
source identity, expression identity, optimization mode, feature ABI, and the
complete energy/vxc output contract. This binding is content-addressed and
included in the receipt hash. A matrix cannot be retained as production-domain
evidence without naming the exact mathematical execution that produced its
candidate values. Backend compilation/runtime qualification remains a separate
stage; the current B1 campaign explicitly records the shared array-Graph
executor rather than pretending that interpreted evidence is a compiled-CPU or
CUDA result.

The unclipped candidate uses the explicit
`libxc-bulk-production-candidate/v2` runtime domain. It keeps the v1
zero-gradient expansion and additionally admits nonnegative rho/tau for
qualification. Exact total-density rows below the pinned registration's Libxc
`p_a_dens_threshold` are screened to zero before Graph evaluation, matching the
outer Libxc work-driver boundary instead of forcing vacuum through an interior
formula. Active empty-spin and zero-tau channels are **not** clipped or filled:
they enter the imported first-order Graph exactly, and the independent oracle
decides whether the functional's E/vxc is valid there. Negative rho/sigma/tau,
non-PSD polarized sigma Gram matrices, and nonfinite inputs remain rejected.

This expansion is still not a generic pass. Zero-gradient, empty-spin and
near-boundary rows must produce finite E/vxc and match the independent Libxc
oracle for that exact functional. Functionals with a true or unresolved endpoint
singularity remain blocked by their numerical matrix row.

Ordinary bulk runtime consumers continue to default to the original interior
domain; both qualification-candidate versions are explicit opt-ins and their
domain/threshold semantics are part of the retained execution identity.

Density-only LDA qualification now uses the separate, explicit
`libxc-7.0/work-density-only-v1` domain. After constructing the imported interior
derivative roots, it evaluates those roots at the pinned Libxc worker's floored
density coordinates. It does **not** differentiate the clipping map. Energy is
weighted by the original total density rather than the floored spin sum; the
complete minority-spin potential is retained. Total-density screening runs
before work evaluation in array, native point, and CUDA tile execution.
The projection owner, worker policy, spin normalization, domain, and requested
outputs enter the expression/binding identity. Stored work receipts are rebuilt
against the current owner and rejected when stale. Prepared programs retain the
owner identity captured at construction.

The production-domain, compiled-CPU, molecular-SCF, and sharded-catalog producers
share this structural recipe, not functional-name branches. Density-work CPU RKS
uses a distinct `RestrictedSemilocalPointProgram` callback on original total
density, gradient and tau; it does not fold total inputs into polarized spins.
The existing polarized descriptor retains its binary layout. Its RKS overload
rejects domain version 5, while legacy domains retain their existing spin-fold
fallback. Automatic CPU method dispatch selects the registered restricted
descriptor for density-only LDA and the polarized descriptor for UKS.
Public KS plans and diagnostic readers select the density-work domain by the
same ingredient-family policy and require exact native version 5; legacy
automatic semilocal diagnostics still require version 4. Neither reader accepts
the other's version merely because both belong to the automatic registry.
The live stationary CPU point bridge also selects this layout explicitly from
the verified functional spin. Its private v4 ABI reconstructs original total
features from equal RKS half-feature channels and duplicates U1 coefficients
for the existing half-channel pullback (including native `vtau/2`). The v2/v3
ABIs remain polarized, including equal-spin UKS inputs; equality never selects
RKS. Restricted calls reject unequal channels and fail explicitly when the
native library lacks v4 or its matching restricted descriptor. Legacy work
domains retain the same spin-fold continuation as their SCF. This point bridge
does not qualify a complete molecular force endpoint or the automatic CUDA RKS
provider. The static automatic CUDA provider independently binds the canonical
U1 density-work Graph for RKS and the P2 Graph for UKS. Both single and compact
batched launchers restore the original total density from paired transport
channels before evaluating U1 work floors; they duplicate its scalar potential
for the existing half-channel contraction. Equal-input UKS remains P2.
`GENERATIVEQC_CUDA_XC_EQUAL_SPIN=0` disables scalar sharing for the legacy
GGA/meta-GGA continuation, not the density-only U1 normalization. Density-only
descriptors report exact domain version 5; other ingredient families retain
version 4. The unchanged private native descriptor binds both explicit
mathematical/derivative layouts through a combined expression identity, and the
provider's version-2 source identity additionally binds both emitted bodies,
traversal and launcher bytes. Old P2-only source receipts do not establish this
new provider contract. These executable point products still grant neither
automatic CUDA KS/force admission nor installed AOT/backend promotion.
The compiled-CPU receipt includes restricted interior, vacuum and active
`[floor, 2*floor)` smoke cases in addition to polarized E/vxc. The molecular
producer verifies the actual native registry's restricted expression identity
against that companion before running the unchanged both-spin lifecycle matrix.
Ordinary geometry success cannot waive the total-density tail contract or
establish physical empty-spin/birth qualification. The catalog campaign
v2 records each registration's candidate domain and the shard's domain set.
Older unclipped LDA failures remain diagnostic artifacts, but an unclipped
receipt cannot provide new density-work compilation/admission evidence.
Compiler-source changes conservatively invalidate old capability/expression
identities even when old-domain emitted mathematics is unchanged. Unsupported work
ingredients fail closed; GGA/MGGA keep their existing unclipped recipe and gates.
Unpolarized total-density and polarized per-spin worker layouts remain distinct:
the native RKS consumer's folded polarized layout is not silently substituted for
the standalone unpolarized Graph. Work-coordinate Libxc parity is not physical
particle-birth, analytic-force, public backend, or catalog-wide qualification.

`generativeqc_compiler.xc.production_domain_cases` instantiates every numerical
rho/sigma/tau row in the exact v3 cases-by-spin matrix from finite physical
density, Cartesian-gradient, and kinetic-density coordinates.
`tools/qualify_libxc_production_domain.py` evaluates those rows through the
generic first-order bulk candidate and compares energy and vxc against the
independent PySCF 2.14.0 / Libxc 7.0.0 oracle. The tool always writes a complete
identity-bound receipt; numerical rows rejected by the versioned candidate
remain explicit failures rather than being silently clipped or skipped.

The two generic control rows are evaluated by the shared
`production_domain_controls` owner. `control/invalid-nonfinite` requires every
nonfinite rho/sigma/tau feature to be rejected before functional mathematics.
`control/lazy-inactive-branch` verifies value, first derivative, second
derivative, array interpretation, and C/CUDA lexical branch emission on a
singular inactive branch. These controls are compiler/runtime invariants rather
than functional numerical coordinates; the surrounding receipt still binds
their pass to the exact functional capability identity.

Running the campaign still does not imply that an imported registration is
production-qualified: unresolved numerical boundary rows remain explicit
failures. Use `--require-pass` only when the caller intends a fully qualified
matrix to be a hard gate.

### Catalog campaign

`tools/qualify_libxc_production_catalog.py` applies the same single-functional
campaign to a deterministic sorted inventory or shard. Use an artifact/scratch
directory rather than writing raw runs directly under `benchmarks/results/`:

```bash
python tools/qualify_libxc_production_catalog.py \
  --output .artifacts/libxc-domain/shard-0 \
  --evidence-prefix artifact://libxc-domain/shard-0 \
  --shard-count 4 --shard-index 0
```

Each eligible registration gets its own campaign JSON with exact receipt and
execution identities. Structurally unsupported registrations are summarized
without a fabricated receipt. `summary.json` reports pass/fail/not-run,
structural blockers, runner errors, family counts, and counts of blocked matrix
case IDs. Sharding is deterministic by sorted registration name, so shards are
disjoint and reconstruct the same selected inventory.

The catalog command is evidence collection, not publication or admission.
`--require-all-pass` turns any non-pass selected registration into a nonzero
command result; without it, negative results are retained for diagnosis and
later B2 aggregation.

The canonical `tests/data/xc/r2scan-tail-reference.json` fixture has
machine-readable status `pass` for the compiled CPU FP64 production entry point,
including its shared Libxc work-MGGA boundary wrapper. Its acceptance oracle uses
the original Libxc 7.0.0 Maple formulas evaluated in 113-bit arithmetic, because
the raw binary64 Libxc empty-spin derivatives are cancellation-sensitive. CI checks
the canonical zero/near-zero-minority tail points and their spin permutations.
`tests/data/xc/boundary/r2scan-zero-minority.json` remains a retained binary64
Libxc diagnostic, not the production acceptance target. Bare derivative roots omit
the wrapper and cannot establish the production entry point's status. This bounded
qualification does not promote the bulk inventory's production-domain or
public-method capabilities.
