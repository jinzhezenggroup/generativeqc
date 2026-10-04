# Bounded exact-class force pages

`GENERATIVEQC_BOUNDED_FORCE_SCHEDULE=compact` selects an experimental full-range
J/K derivative schedule when the prepared `GeneratedExchangePlan` can admit its
optional storage. Other values retain the existing mixed bounded schedule. The
control is part of prepared/checkpoint policy identity; change it before preparing
the owner. This is not a default promotion or a qualified GPU speedup.

An admitted compact page takes precedence over the legacy
`GENERATIVEQC_BOUNDED_ANGULAR_FORCE=angular` diagnostic for full-range derivatives.
Without page admission, the existing schedule applies (mixed by default, or that
explicit angular diagnostic). Compact pages do not change the LR schedule. Leave
the angular diagnostic unset when qualifying the single-screen mixed/compact pair.

The compiler's `integral/direct_force_pages.py` uses the canonical shell catalog
and `ScheduleIR` to emit through-f class/order dispatch metadata and page budget
admission. The native owner composes existing bounded screening and generated
prefix/scatter facilities:

1. Partition the block-product domain into disjoint pages (at most 4096 products).
2. Screen each candidate once, retaining its raw pair indices and exact class.
3. Prefix the 55 class counts and scatter retained tasks into contiguous slices.
4. Launch fixed-class/order consumers for classes present in the prepared basis.

No class-specific full-domain scan, host count readback or overflow recovery pass
is required. A block product contributes at most 1024 candidates, so even a fully
admitted page fits. The indexed Schwarz domain and original triangular domain
use the same page protocol. Low orders 0–3 retain their existing weighted scalar
consumers; higher orders retain the qualified AO derivative contraction with a
fixed class/order entry. J and K remain separately observable, sharing recurrence
work. AO screening, spin conventions, spherical transforms and precision do not
change. This schedule does not claim to reduce exact-ERI asymptotic complexity.

For page capacity `Q`, scratch is `25 Q + 884` bytes: two 12-byte task arrays,
one-byte class tags, and class count/offset/write/head arrays. The maximum is
104,858,484 bytes, independent of the molecule's total quartet capacity. The
planner halves the page's block capacity until it fits the owner's remaining
budget; zero capacity or device allocation failure retains mixed execution.
Host bookkeeping OOM rejects owner preparation rather than pretending that the
optional page was admitted; non-allocation CUDA failures propagate. All admitted
bytes belong to the owner's allocation registry and accounting.

For `B` block products and page size `P`, shell traversal is once across `B`,
classification/scatter storage traffic is bounded by `1024 B` candidate slots,
and class launch count is at most `C ceil(B/P)` for `C` present classes. Empty
class launches and small-budget launch amplification are real costs, not zero.
The consumer still expands admitted shells into AO tiles. Class task counts are
not FLOP counts; contraction length, angular class and derivative algorithm matter.

## Qualification

Host tests execute the actual classifier/prefix/scatter in a serial capsule with
synthetic screening, and exercise generated budget admission and host launch
failure paths. They do not validate CUDA concurrency or derivative mathematics.
On an allocated GPU, compare the native
`generativeqc_cuda_fock_provider_tests --full-range-shell-sources-only` gate with
compact and mixed policies, both with and without the indexed domain. The native
gate uses an independent CPU ERI derivative oracle, separate J/K channels,
Cartesian/spherical bases, spin cases and nonzero screening.

Before promoting, retain per-class admitted work equality, constrained-budget
fallback coverage, sanitizer evidence, linked kernel registers/stack and whole
source-stage timing. Only then compare complete cold, warm and changed-geometry
energy/force endpoints on the same source and GPU under unchanged numerical
gates. Reduced stack alone is not speedup evidence. Never label another policy's
integrated endpoint measurements as measurements of this candidate.
