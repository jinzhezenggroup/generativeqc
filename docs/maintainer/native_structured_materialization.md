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
The JSON includes byte SHA-256 identities for all scanned sources and all three
consumed scanner modules, Git commit/tree, and source/working-tree dirty state.
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
consumer/structured-IR fields. A caller location does not establish overload
resolution, executed reachability, required dense layout or consumer support.
Before changing a representation, review the returned/escaped object's entire
consumer lifetime and its layout contract. This slice provides a standalone
CLI/API; common work-audit and TensorIR integration remain follow-up scope under
[#1631](https://github.com/jinzhezenggroup/generativeqc/issues/1631).

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
**only** that those source paths coexist in the scanned revision.
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
