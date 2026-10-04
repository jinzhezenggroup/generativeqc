# Evidence: canonical AO-component preparation repeats across physical shells

Status: proposed consumer redesign; measured value-domain census, no new speedup
Date: 2026-10-04

## Observation and provenance

The user prioritizes orders 5--8 full/LR values and derivatives over VV10.
The earlier [endpoint diagnosis](2026-10-04-tzvpd-warm-integral-diagnosis.md)
assigns 89.49% of the profiled twelve-atom warm endpoint to **all** integral
families, and 76.04% of value-kernel duration to orders 5--8. Neither number
establishes the force-class distribution.

N1 RTX 5090 Slurm 5748 now measures the resident canonical value domain for
the same original twelve-atom geometry and complete H/O def2-TZVPD snapshot:
232 spherical AOs, 256 Cartesian source AOs, 96 shells, screening 1e-12.
The frozen #1842 source is `591a5395193f9874aa2c7aac9067e2a940e70e63`, identity
`5cac4ce02d4527c30728ead85ed089ab28bd6f28f3e019c4f564e22a794f6a3a`, library
`b05b345c564273e5ece11123b91b0af45cc9de7ebe38243c0792736c985183e4`.
The joint full/range candidate and alternative recurrences are disabled.

The diagnostic downloads the actual canonical pair order, screened row
prefixes, Schwarz bounds, AO-shell map and primitive offsets. A host traversal
enumerates those exact rows, retaining the final AO predicate. Actual resident
full J/K and LR K enqueues each independently report **407,065,289** candidate
and contracted radial evaluations, exactly matching the host count. An
identity density is sufficient for this route check because its value
screening is geometry-only. This is one source call per operator, not a count
of all calls in a cold/warm molecular endpoint or a force-domain census.

| Order | Admitted AO quartets | Primitive-loop products | Unique shell quartets | Shell preparation reuse bound | Existing 32-lane reuse bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 | 82,584,777 | 190,329,945 | 800,252 | 100.014 | 1.292 |
| 6 | 80,687,571 | 152,533,106 | 371,275 | 209.573 | 1.364 |
| 7 | 61,299,736 | 96,761,495 | 140,155 | 420.103 | 1.431 |
| 8 | 35,957,158 | 48,371,977 | 42,986 | 801.604 | 1.684 |

Across orders 5--8, current primitive-loop products total **487,996,523**.
Counting each of the 1,354,668 physical shell quartets' primitive products
once gives **2,921,528**; retaining the current contiguous 32-lane cohorts
instead gives **355,424,818**. Their ratios are **167.035** and **1.373**.
Across all orders the corresponding ratios are 27.151 and 1.288.

These are measured domain counts and derived *common-preparation* reuse
bounds, not FLOPs, saved total operations, achieved reuse or speedup. Every
AO-component contraction and source scatter remains necessary. Reachable
Coulomb closures can differ between components, so constructing a shared
union can add states. Counts also do not prove which exact shell class has
the greatest duration. Raw probe source, inputs, failed attempts, logs and
source/build hashes are retained in
[the offline-verifiable evidence](../../../benchmarks/results/wb97mv-canonical-work-20261004/README.md).

## Consequence for consumer design

`src/scf/cuda/direct_jk_kernels.cu::canonical_jk_kernel` already has angular
buckets and screened rows. The compiler's
`integral/direct_source_contraction_cuda.py::contracted_eri_cartesian_source_shell_class`
already selects shell-class templates, but each AO component separately
enters its four primitive loops and prepares geometry/Hermite/Coulomb data.
Adding another class name without changing this work is insufficient.

For shell quartet q, P_q primitive products and A_q admitted AO components,
let G/H denote common geometry/Hermite preparation, R_F/R_L range-specific
radial preparation and C_(q,a,F/L) the component contraction. A current
two-pass source has work of the form

    sum_q P_q [A_q (2G_q + 2H_q + R_F,q + R_L,q)
               + sum_a (C_(q,a,F) + C_(q,a,L))].

A bounded shell/component consumer with groups of at most K components
could approach

    classify_and_schedule
    + sum_q P_q [ceil(A_q/K) (G_q + H_q + R_F,q + R_L,q)
                 + sum_a (C_(q,a,F) + C_(q,a,L))],

plus union-closure, synchronization, storage, output and any duplicated
preparation at page boundaries. These symbols are work categories, not
interchangeable operation counts. Different masks retain separate membership;
this particular value source has matching full/LR predicates. No full-minus-LR
subtraction is required. #1842 only shares the two operators' preparation;
it does not implement the cross-component reuse described here.

The small 1.373 bound within current lane cohorts makes unrestricted
warp-local sharing a limited first target. A bounded shell-oriented component
tile is the stronger architectural candidate. Keep only a budgeted page of
descriptors plus per-group geometry/Hermite/radial scratch; do not materialize
all 407 million AO task records or an AO-fourth-power integral tensor. Scratch
is O(Q descriptor_bytes + resident_groups * source_workspace_bytes), with
explicit page capacity Q; actual bytes and occupancy depend on the selected
class and group size and remain unmeasured. Consume full and LR roots in
sequence where that shortens lifetime, retaining separate moments/outputs.

## Correctness and minimum qualification

Preserve canonical pair membership and all slot permutations, including equal
shell/AO IDs; primitive weights and Cartesian normalization; exact AO-level
screening; spherical density/output projection; separate J, full K and LR K;
and the existing bounded fallback. Shared data must outlive all consumers on
the owning stream without adding hidden global allocation. Force reuse also
needs repeated-atom merging, translation invariance and derivative-slot maps.
This value census cannot be used as its work count.

First select one hot class and instrument actual common preparations plus
component contractions in the old/new consumer. Require matching admitted
AO sets, independent full/LR matrices and forces, sanitizer success and budget
fallback coverage. Only then compare same-source/same-GPU complete original
and displaced cold plus five warm calls at 12 atoms and a larger admitted
point, with unchanged 1e-8 Eh / 1e-7 Eh/Bohr gates. Count new queue/union work
and inspect linked resources and dynamic traffic; theoretical reuse alone
does not establish lower endpoint cost.

The #1841 owner already implements single-screen full-range class pages. Its
first indexed 96-atom SVP source timing regresses 26.37%, with cost in consumers.
Do not duplicate that classifier or infer that classification alone suffices.
Its result is not a TZVPD calibration. Orders 7/8 scalar center derivatives
in #1845 are a separate experiment and cannot be counted as measured savings
from this census. Further VV10 experiments remain lower priority.

## Failed probe attempts retained

Slurm 5746 incorrectly used the host-vector J/K entry point, which dispatches
an independent public-AO kernel and never writes the canonical counter. Its
borrowed counter was uninitialized: the failure establishes no actual count.
Slurm 5747 reached the resident API but retained the HF helper's default
derivative_order=1; the value-only seam correctly rejected it. Slurm 5748
explicitly requests order zero and retains all original dimension/count gates.
These are diagnostic harness corrections, not production optimization gains.
