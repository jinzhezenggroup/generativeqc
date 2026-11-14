# Decision: dependency-scoped lifetime for native Libxc evidence

Status: implemented
Date: 2026-11-14

## Problem

The native catalog used a whole frontend/compiler/native input hash and the
embedded native build identity as a single qualification-expiry switch. An
unrelated change could demote all measured backend/spin/product contexts. The
same aggregate provenance also appears recursively in scalar expression,
functional, MethodIR and execution-plan identities. Simply deleting the outer
staleness test would not fix that second invalidation path.

## Decision

Keep historical source, compiler, environment, ELF and receipt identities
immutable. Separately authenticate a measured per-file census against both the
campaign source hash/count and its embedded native input digest. Native inputs
include generator scripts and assets, not just C++/CUDA files. Compare additions,
removals and edits through explicit backend/product ownership boundaries;
unknown/shared inputs remain conservative dependencies.
In particular, `src/posthf/capacity.hpp` and `src/posthf/raw_source.hpp` are
included by SCF/integral consumers: the post-HF directory is not a safe blanket
exclusion. Only the separately owned MP2 numerical families are excluded there.

The original campaign at `fbf57c111c7c885efbc0cd1457e6a7732c9e3d95`
did not embed its census. Recover its 1,955 frontend/compiler inputs and 2,124
native inputs (2,130 distinct paths) from that Git tree, authenticate both
original digests, and retain the hash-pinned sidecar under `manifests/libxc/`.
Do not alter the measured campaign or pretend that recovery performed an
experiment. Fresh producers embed their census directly.
The old build manifest omitted the native stationary CUDA and Libxc CPU/CUDA
registry generator scripts. Retain their three hashes as explicitly supplemental
inputs recovered from the same declared Git revision, protected by the sidecar's
manifest digest rather than rewriting the old embedded identity. Add them to
the current native identity manifest so future build verification and cache
invalidation bind the actual generators. Missing generator dependencies fail
closed; the supplemental map cannot override an original input hash.

Per-registration point validation first attempts exact current identity
agreement. If aggregate inventory metadata changed, a separate proof can hash
the current emitted bytes with the immutable recorded mathematical metadata.
It must reproduce the original point-source digest, and every layout/domain/ABI
field except the aggregate expression provenance must match. This is a
cryptographic byte-equivalence check, not a relaxed numerical comparison.
Similarly, historical functional/MethodIR/plan digest preimages are recomputed
from exact current scientific payloads with only authenticated aggregate
capability/catalog metadata transported. No compiler object, generated artifact
identity, cache key or executed library is modified by these offline checks.

Source changes in general code generators still require scoped regression.
Only the registration catalog and the separately owned curated B3LYP policy
can use per-registration point equivalence instead of a whole-file expiry in
the initial dependency policy. Unknown future emitters are not assumed safe.

Merge independently ingested incremental campaigns without synthesizing new
endpoints from incomplete receipts. Preserve execution identities and exact
coverage from every batch, and count registrations rather than duplicate
measurements. Pending requests are resolved only by exact matching current
contexts, keeping superseded source deltas as historical metadata rather than
repeatedly requesting an already completed rerun. The representative acceptance
gate examines paired admitted
contexts, rather than vetoing the entire inventory because some other receipt
needs revalidation. Published-report comparison excludes only volatile
whole-tree provenance fields, never scientific or artifact identities.

## Rejected alternatives

- Erase or relabel historical GPU evidence: historical execution remains valid.
- Unconditionally restore all old E/F claims: Direct J/K, convergence and
  precision changes after the campaign can genuinely affect its endpoints.
- Ignore all XC compiler source changes: scalar, restricted-spin, spatial and
  provider generation have different contracts; a CPU point proof alone cannot
  qualify changed CUDA/force generators.
- Require fresh measurements of every imported functional at every commit:
  representative paired family/backend/spin coverage and incremental CI are
  sufficient; missing stronger public-method receipts are not runtime failure.
- Combine partial products from different receipts: exact paired context and
receipt/library identity remain required by the acceptance gate.

Partial checkpoints may contain only one complete registration or backend.
Once another batch supplies paired E/F receipts for both spin layouts of that
backend, its missing-receipt gap must not remain a global blocker. Preserve the
gap in per-campaign metadata and `superseded_blockers`; never suppress genuine
scientific failures or infer the absent spin from a single-spin measurement.

## Invariants

- Preserve all numerical gates, independent oracle inputs, compiled/loaded
  object checks, work counts, lifecycle and unsupported-domain boundaries.
- An inherited claim proves unchanged relevant dependencies within the original
  toolchain/environment scope, not execution of a new whole-tree binary.
- Without an authenticated measured census, changed-source reuse fails closed.
- Slurm GPU availability and new scientific execution are separate from host
  receipt replay; never report the latter as a fresh GPU experiment.

## Evidence

Executable mutation tests cover unrelated and documentation changes, CUDA-only
and force-only changes, shared SCF/J/K/ABI inputs, additions/removals, changed
per-registration Graph contracts, recursive provenance equivalence, incremental
merging and immutable historical receipts. The current scalar-domain adapter
also revalidates real retained numerical data without the historical test seam.
See `tests/python/test_libxc_dependency_qualification.py` and
`tests/python/test_libxc_native_catalog.py`.

Fresh scoped execution is retained under
`benchmarks/results/libxc-scoped-qualification-20261114/`. Jobs 17386 and 17388
measure one common LDA/GGA/tau-MGGA registration in both backends and spin
layouts: 12 contexts and 24 E/F products on ELF
`de2c01a15769c7942338ec9158e0fc0a517f5f447ae23c84475b0a6f5289121e`.
The first checkpoint's completed LDA is supplemented rather than rerun after
the finite job limit. Job 17388 additionally passes three resource/lifetime
pytest cases and the equivalent nonconvergence case on compiled MGGA_X_GVT4.
The manifest hash-pins that separate receipt to its measured ELF; archival
integrity does not make it numerical E/F promotion or a new-source experiment.
Full source content fingerprints disclose the uncommitted working-tree patch;
the recorded Git base is not falsely presented as containing that patch.

## Consequences and revisit conditions

The policy is intentionally conservative for unknown/shared dependencies.
Narrow a dependency only with an ownership argument and executable mutation
coverage; extend context dimensions before claiming DF/ECP, higher-l, mixed
precision, response or another execution environment. Current Direct J/K and
SCF changes remain explicit endpoint regression requirements outside the newly
measured representative contexts. A policy change alone is never evidence of
new scientific execution.

## Review follow-up: incomplete generator authentication

The original three-generator repair did not cover the full transitive native
source-generation closure. Review identified 42 additional generator inputs;
the current build manifest now includes them. The two measured working-tree
campaigns do not retain authenticated hashes for those omitted inputs, and the
available checkout is not proof of their contents at execution. Do not invent
historical hashes or silently patch the immutable campaign inventories.

The exact published summary at `7d7dc89b5cce4c71f617c96fdd4865bf8eb7968f`
is retained as a separate previous-summary artifact. Actual live ingestion
withdraws current compiled/endpoint inheritance with exact stage-demotion
acknowledgements, while retaining 11 independent production-domain proofs and
all historical 12-context/24-product measurements. The normal archival checker
passes; representative `--require-current` acceptance must fail. This records a
source-authentication gap, not a new numerical failure or changed runtime policy.
Tracking issue #1118 is not closed by this partial qualification. Recover an
authentic measured-source supplement or obtain appropriate new source-bound
qualification before restoring affected claims; preserve exact provenance and
all original numerical gates.

## References

- Tracking issue #1118 and integration #3054.
- `docs/developer/libxc_bulk_capabilities.md#native-endpoint-catalog`.
- `tools/generativeqc_validation/libxc_dependencies.py`.
- `manifests/libxc/native-qualification.json`.
