# Experiment: share scalar Coulomb roots across order-seven/eight force centers

Status: proposed; default-off native/compiler implementation, qualification pending
Date: 2026-10-04

## Mechanism and ownership

The generic full/LR derivative consumer in `direct_force_quartet.cuh` repeats
the Cartesian primitive recurrence with Dual3 seeds for U-1 distinct atoms.
The existing order-4/5/6 all-center Wick consumer does not cover orders 7/8.
Extending its exponential subset walk without measurement is not the chosen
route. This experiment instead reuses the retained Cartesian Hermite and
Coulomb recurrences, with one scalar auxiliary of order L+1 per primitive.

The compiler applies `d_A g_a = 2 alpha g_(a+1) - a g_(a-1)` through a view of
prepared Hermite rows. Nine contractions give the first three slot gradients;
translation invariance gives the fourth. Contraction coefficients and physical
AO normalization remain outside the primitive as before. The shell-class
canonicalizer tracks and reverses its slot permutation, including equal AO IDs.
The native consumer merges repeated atoms and retains its established force
weights, exact AO predicate, independent J/K channels, scatter and final-atom
reconstruction. No derivative tensor, source queue or extra screening pass is
introduced. PR #1841 continues to own the independent compact-class schedule.

`GENERATIVEQC_DIRECT_SCALAR_CENTER_GRADIENT=scalar` (alias `1`) is default off.
Admission is frozen in DeviceBatch and checkpoint/resource policy. Explicit
force selections of the reachable-recurrence or Hermite-convolution experiments
retain priority, declining this consumer; values-only selections do not conflict.
Only orders 7/8 in the full-range and individual LR consumers change. Direct SR,
orders outside 7/8, and the separate fused-RSH fallback keep their prior native
consumers. The molecular omega=0.3 facade uses the full plus LR consumers.
The compiler primitive itself supports independent Full/SR/LR moments; no
full-minus-long subtraction is introduced there. It uses strict FP64.

The force-control change does not accelerate each cold SCF iteration. PR #1842
separately addresses repeated full/LR value preparation. Neither improvement is
added to the other's timing or presumed to close the remaining endpoint gap.

## Work and storage model

For one admitted AO quartet with P primitive products and U unique atoms, the
old generic derivative performs P*(U-1) Dual3 source evaluations. The candidate
performs P scalar auxiliary preparations and P*9 raised/lowered contractions.
Each contraction's length depends on its actual x/y/z powers. Additional
coefficient preparation, derivative-row operations and contractions are not
zero. Molecular P/U/class counts, FLOPs and achieved rates remain null.

The old auxiliary has `choose(L+4,4)` Dual3 entries per center evaluation;
the new one has `choose(L+5,4)` doubles. At L=7 these are 10,560 versus 3,960
bytes of source-level auxiliary storage; at L=8, 15,840 versus 5,720 bytes.
These are neither total thread frames nor measured traffic/occupancy. For the
canonical shell angular tuple (A,B,C,D), the candidate's six Hermite workspaces
add `3*((A+2)*(B+2)*(A+B+4)+(C+2)*(D+1)*(C+D+3))*8` bytes. The fourth slot
needs no raised bound. Views, accumulators, caller frames and spills are extra.
A noinline contracted entry isolates the alternative recurrence frame, but
linked/device resource evidence is still required before making a storage claim.

## Validation and risks

A CPU-only precursor using the same retained contraction passed 384 independent
Gaussian Laplace/Wick primitive checks across the eight angular partitions,
Full/SR/LR, orientations, repeated/distinct centers, diffuse exponents and common
translations. Its largest normalized gradient discrepancy was 2.446e-16.
The receipt is `.artifacts/scalar-center-gradient-20261004/result.json`; this
precursor does not establish final-source or device qualification.

The integrated emitted-source suite subsequently passed **978 host tests** on
n2 using verified ccache, including 384 primitive, 24 multi-primitive/permutation,
9 equal-AO-slot, 80 policy/freezer and 481 adjacent source/ownership/allocation
cases. Final host receipts are retained under
`.artifacts/scalar-center-gradient-20261004/integrated-host/`. This establishes
arithmetic and host control flow, not CUDA behavior or performance. The native
borrowed census lease fences and restores its owner before storage destruction,
including exceptional exits.

Final-source gates compile emitted primitive and contracted wrappers, validate
all eight ERI permutations and multi-primitive signed coefficients, and exercise
the actual policy/freezer. Native `--scalar-center-forces-only` must observe
positive order7/full, order7/LR, order8/full and order8/LR contraction counts and
pass independent source forces in both spins, Cartesian/spherical bases and
repeated/distinct atoms. Diagnostic counters are borrowed, null in production,
and count contracted AO derivatives, not primitives or FLOPs.

Changing derivative arithmetic can alter cancellation and SCF/force roundoff.
Independent CPU oracles, GPU sanitizers, disabled-channel checks and original /
displaced complete E/F gates of 1e-8 Eh / 1e-7 Eh/Bohr remain required. Qualification
must compare same-binary off/on cold and five warm samples, keep other force
experiments off, and include a larger point before any default promotion.
Cold speedup, warm speedup and GPU resource improvement are currently unknown.

The frozen implementation has now passed the retained canonical-value and
range-force GPU regressions, the through-f response regression, and its
independent scalar-force gate on n1 RTX 5090 Slurm 5744. The actual alternative
consumer counts are order7/full = 37,008, order7/LR = 37,008, order8/full =
54,960 and order8/LR = 54,960. These count contracted derivatives in the
qualification fixtures, not the molecular workload or FLOPs. Compute Sanitizer
memcheck and initcheck repeat that gate with zero errors, and all eight selected
checkpoint policy cases pass. Complete off/on cold/warm/displaced measurements
are now admitted and running; endpoint results remain pending. Raw logs
and the Slurm, source and binary receipts are retained under
`.artifacts/scalar-center-gradient-20261004/qualification/5744/`.

## Linked resource observation before endpoint qualification

The n2 Release/sm_120 build completed with 453 ccache compiler commands. Its
library SHA-256 is
`1131222004b0736de9f8190a645fcd5e04dd7278f1869bdadc26d8b0c8b3e1a7`.
The frozen implementation is `eeaca5df3950c4c433c117b8d1ff6fbe19ea204b`;
this evidence-only follow-up changes no production source.

Compared with the #1842 parent, cuobjdump reports 255 registers and mixed
force-screened kernel STACK 90,696 -> 90,968 B. The order-7/8 diagnostic roots
also retain 255 registers and slightly larger stacks. The reduced scalar
auxiliary formula therefore did **not** become a smaller linked kernel frame:
the AD and higher-order fallback consumers remain reachable. Standalone
device-function rows with all-zero resource metadata are unavailable resource
evidence, not zero-cost functions. Static resources do not measure actual
spills, traffic, achieved occupancy or wall time. Receipts are under
`.artifacts/scalar-center-gradient-20261004/receipts/linked-force-comparison.json`.

Separately, #1841's first compact-class schedule is 26% slower on its indexed
96-atom SVP full-range source endpoint despite smaller class stacks; its
profile places cost in consumers rather than classifier/prefix/scatter. This
is different-source/basis evidence, not a TZVPD timing prediction. Do not
automatically compose that schedule, duplicate its classifier, or infer a win
from reduced stack. This candidate must first establish actual derivative
execution, independent GPU forces, sanitizers and full endpoint behavior.
