# Decision: share cached primitive-pair recurrence across Direct components

Status: implemented (qualification-only value candidate)
Date: 2026-10-06

## Problem

The generic high-order Cartesian source evaluator repeats four primitive loops,
Hermite preparation and Coulomb/Boys recurrence for every AO component. The
existing immutable `PrimitivePairData` cache already owns p, mu and P, but those
records did not enter this recurrence. A root-count specialization alone does
not change that work multiplication.

## Decision

Reuse one authoritative Hermite recurrence with supplied Gaussian geometry and
one authoritative contraction of a prepared Coulomb simplex. A generated
CTA-owned workspace borrows the existing ij/kl cache records. Each bra primitive
pair prepares its complete axis Hermite tables once; each bra/ket pair product
prepares the ket tables and Coulomb simplex once. All admitted AO components in
the exact packet consume that shared state, then reuse their contracted ERI in
the existing RHF/UHF J/K symmetry scatter.

The native HF high-order Fock launcher can select this schedule through
`GENERATIVEQC_DIRECT_PAIR_MATERIALIZED_VALUES=1`, frozen in its prepared batch.
The candidate is default-off pending complete qualification and endpoint
evidence. The DFT bounded Direct provider does not yet consume this schedule;
freezing the request in its packed batch does not prove execution. Mixed
precision, low orders, component-reachable/reassociated
experiments and missing pair storage retain their exact incumbent. There is no
new recurrence formula, global intermediate allocation or primitive/AO tensor.
The largest shared workspace is statically bounded below 48 KiB.

## Arithmetic and compatibility

Cached p/mu/P have the same preparation arithmetic as the raw evaluator. Keep
the original axis Gaussian bases, primitive traversal and individual
coefficient multiplication order. Do not divide the cache's combined weighted
coefficient to recover raw factors: underflow/zero can erase those inputs. A
unit-Gaussian normalization would reassociate the existing FP64 graph and needs
separate cancellation/overflow qualification; it is not required for reuse.

Inactive/tail lanes participate in shared publication and retirement barriers
without reading invalid AOs. No live component means no recurrence. Optional
test counters measure executed bra/ket/Coulomb preparation, component consumes
and publication; production supplies no counter or component-output allocation.

## Scope and remaining work

The current packet is the existing independently screened 256-AO domain. For
order-five classes all components fit in one packet; larger shell quartets may
have several packets and repeat recurrence between them. This is not claimed
as once-per-whole-shell-quartet execution for those classes. Coalescing their
admission masks, derivative consumers and complete 48/96-atom endpoint
qualification remain part of #1892's full goal.

## Evidence and revisit

Retained low/order-four/order-five host numerical controls protect the extracted
incumbent arithmetic. Real-device qualification compares orders 5--12 against
the retained raw component evaluator, and contracts an independently decoded
host permutation orbit for RHF/UHF, J-only and K-only matrices. Counters must
prove one Coulomb preparation per pair product per live packet. Full/tail,
screened, same-pair, coincident-center and inactive-system domains are gates.
Complete endpoint timing and semantic work must decide whether to promote this
candidate; isolated recurrence reduction is insufficient.

On n1, finite Slurm job 6141 qualified orders 5--12 and all three J/K channel
selections. Memcheck and initcheck each reported zero errors. The signed
two-primitive fixture executed 16 Coulomb preparations per live packet, versus
16 preparations per admitted component in the retained schedule. For example,
the order-five case published 162 components with 16 preparations; order twelve
published 10,000 components across 40 packets with 640 preparations. These are
work receipts, not endpoint speedup claims. An initial K-only host-orbit oracle
mistakenly applied the combined HF exchange factor; K-only publishes positive
K. Correcting that convention, while retaining the numerical gates, passed all
channels. Raw receipts are retained in `.artifacts/1892-pair-materialized/`.

## References

#1892; `direct_pair_cache_cuda.py`, `direct_pair_materialized_cuda.py`,
`direct_cartesian_contraction_cuda.py` and `direct_angular_fock.cu`.
