# Native structured materialization audit

Run the standalone standard-library tool from the repository root:

```sh
python tools/audit_structured_materialization.py --path src/posthf/mp2_gradient.cpp
python tools/audit_structured_materialization.py --output native-support.json
python -m unittest tests.python.test_native_structured_materialization
```

The default scan inventories native files under `src/` and `include/`, excluding
vendored `src/xtb/native/`. Repeat `--path` for individual files or directories;
missing or outside-root inputs are errors, including recursively discovered
links. Sources are resolved before containment checks, vendored exclusions and
deduplication; overlapping inputs and same-root symlink aliases are scanned once.
The JSON includes byte SHA-256 identities for all scanned sources and all consumed
scanner modules and the compiler-owned shared materialization policy, Git commit/tree, and source/working-tree dirty state.
Line endings affect byte identities; compare receipts using their recorded bytes.
Source dirty state checks selected paths, resolved selection roots and canonical
source paths, including untracked sources. Ignored-file detection is limited to
files actually scanned. Resolved roots retain deletion evidence even when no
canonical file remains. Known changes yield `true`. Unverified symlink
topology or unavailable Git evidence yields `null` unless a change is already
known; `false` requires clean ordinary paths. `alias_topology_unverified` records
that topology boundary. Broken descendant links are errors.

## Supported certificates

The inventory sees two-argument `std::vector<double/float>` zero constructors
and `.assign(count, 0.0)` candidates. Only a fresh local vector with a closed
producer body can earn `exact-structured-write-support`. A fresh default vector
immediately followed by its first zero `assign` is also admitted. Aggregate
members and previously used storage remain unknown.

The certified subset has integral scalar parameters, immutable scalar setup,
homogeneous rank-2/3/4 symbolic allocation products, canonical row-major index
arithmetic, and canonical unit-stride `for` loops. Same-file, same-namespace,
unambiguous arithmetic helpers must have integral parameters and a single return
expression. Their bodies are expanded; helper names alone are never proofs.
Immutable extent aliases are resolved for full-domain equality, offset partitions
and symbolic growth, with their relationships retained as `extent_aliases`. For
example, `all = n` does not hide a full dense write, while a fixed three-address domain
has growth degree zero. Offsets proved equal to zero are canonicalized to `0`,
so a zero-offset alias cannot hide a complete dense domain. Scalar conversions
must preserve mathematical values; the scanner does not infer integer widths or
prove that precondition.
Writes have side-effect-free scalar arithmetic RHSs. Unknown calls, aliases,
other mutations, lambdas, unsupported setup, preprocessor control and unparsed
control flow fail closed. This lexical subset assumes ordinary C++ token meanings
and valid code; it cannot resolve rewriting macros supplied by included headers.

Supported address domains include occupied/virtual Cartesian blocks, repeated
indices (diagonals), and inclusive lower triangles. Offset virtual indices need
an explicit `virtuals = n - occupied` relation. Certificates state nonnegative,
in-range dimension, value-preserving scalar conversion and no-overflow
preconditions. They describe addresses that
the admitted loops can write, rather than guaranteed numerical nonzeros. Zero
values outside these domains follow from initialization and the absence of any
other admitted producer mutation. A certificate ends at the producer return.

| Domain | Dense elements | Written addresses | Growth degree |
| --- | --- | --- | --- |
| Occupied/virtual rank 4 | `N^4` | `O^2 (N-O)^2` | 4 when both families scale |
| Full rank 4 | `N^4` | `N^4` | 4; no storage recommendation |
| Diagonal rank 2 | `N^2` | `N` | 1 |
| Inclusive lower triangle | `N^2` | `N(N+1)/2` | 2; strict reduction only for `N > 1` |

For a single domain, the tool emits its symbolic dense/address ratio. A ratio
can be undefined at zero support and does not by itself prove a strict reduction
for all admissible dimensions. Multiple domains receive a union upper bound;
overlap and strict savings remain unproved, so no storage recommendation is made.
A full-domain write is classified `dense-write-domain`, even alongside other
writes. No numerical threshold, screening or approximation is used.

## Producer, consumer and representation boundaries

Every candidate carries its producer/allocation location, dense vector or member
ABI evidence, syntactic same-file caller locations, and explicit unresolved
consumer/structured-IR fields. The additive `materialization_diagnostic` uses
the same compiler-owned policy as Python source audits and TensorIR. A caller location does not establish overload
resolution, executed reachability, required dense layout or consumer support.
Before changing a representation, review the returned/escaped object's entire
consumer lifetime and its layout contract.

The common `tools/audit_native_work.py` entrypoint consumes this strict native
selection/provenance path, includes every native candidate (including unknowns)
in its `findings`, and exposes normalized `materialization_diagnostics` and the
MP2 `production_boundaries` census. Python sources are selected and hashed through
the same boundary before the existing AST analyzer runs. Every analyzer consumes the same captured source bytes; the common audit never
reopens a path after selection. Missing paths and outside-root aliases are errors.
Filesystem dirty/alias state records scan-time observations, not a promise about
subsequent filesystem changes. A policy file edited after its module was imported
fails the report rather than hashing new bytes while executing stale policy.
Source/scanner hashes include the common policy and every consumed analyzer.

`generativeqc_compiler.common.materialization` owns only evidence normalization
and recommendation/blocker policy. Existing native/Python analyzers still own
parsing and certificates. `analyze_complexity(program).materialization_diagnostics()`
explicitly reports TensorIR logical element counts, symbolic degree, retained
output layout, and missing exact-support evidence. It does not infer zero sectors
from arbitrary TensorIR shapes or factorized equations. The ordinary complexity
summary, equation serialization/hash, AD, and normal preparation are unchanged.

Normalized blockers distinguish unsupported producer writes, full-domain writes,
unresolved union cardinality, downstream consumer uncertainty, declared vector or
aggregate-member layout, retained TensorIR outputs, and missing structured IR.
Unknown consumers are never reported as proved dense-layout requirements. Only
an exact single-domain certificate permits a conservative block/diagonal or
packed-storage review recommendation; union upper bounds, unknown aggregate
writes, and shape-only IR do not. The adapter performs no automatic rewrite.

In current MP2 source, dense `initial_orbital_weights` and the corresponding
canonical dense APIs coexist with streamed/factorized owners. The scanner
reports the rank-four `result.two_electron.assign(fourth_power(n), 0.0)` as
**unknown**, not as a certified sparse tensor, because this is an aggregate
member with several exact sectors and later mutation. No automatic N^4
representation rewrite is authorized.

The full PR source-audit JSON now includes an additive
`production_boundaries` array. For
`src/posthf/mp2_gradient.cpp`, the `mp2-representation-boundary.v1`
census binds the actual source SHA-256 and checks eight specific, same-file
free-function and source-expression anchors: checked extent helpers, canonical
N^4 allocation and caller, streamed Fock-weight storage/caller, factorized
Lagrangian ownership, and the RI reverse consumer. `SOURCE_VISIBLE` means
**only** that those source paths coexist in the scanned revision. The shared
diagnostic retains the observed `fourth_power(n)` expression with unknown certified
growth degree: helper anchors alone are not a proof of their full arithmetic.
`INCOMPLETE` makes changed or missing anchors explicit. Neither status
proves the selected public endpoint, consumer ABI/lifetime, complete
scientific write support, or runtime allocation bytes. The linked #1574
implementation remains a separate performance decision.

CI publishes `native-structured-materialization.json` and
`native-complexity.json` beside the existing work audit and DF ratchet.
Existing exact rank-2 matrix-chain findings use the strict
`audit_native_complexity.py --fail-on-matrix-chain` rule; **other**
high-rank/materialization candidates are advisory. No numerical sparsification,
source filename exemption, or synthetic performance claim is introduced.

The tool reports source evidence only: no runtime allocated bytes, endpoint
speedup, scientific validation, ABI migration or zero-allocation guarantee.
