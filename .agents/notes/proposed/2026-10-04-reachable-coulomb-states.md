# Experiment: evaluate only consumer-reachable Coulomb recurrence states

Status: proposed (opt-in implementation; CUDA numerical gates pass, endpoint qualification pending)
Date: 2026-10-04

## Motivation and scope

The accepted full-TZVPD 12-atom diagnostic attributes 89.49% of warm device
duration to full/LR values and derivatives. Total orders 5--8 account for
76.04% of **value** duration, not a measured force-class distribution. The
[angular-pass experiment](../rejected/2026-10-04-tzvpd-angular-force-schedule.md) retains
254--255 linked registers/thread and already regresses the complete 3-atom
diagnostic by 29.15% and the 12-atom diagnostic by 10.063%. Splitting the
launch does not remove recurrence work.

`direct_cartesian_contraction_cuda.py::eri_cartesian_value` reads only
`R(0,t,u,v)` through the current component's total axis powers `(X,Y,Z)`.
`src/scf/cuda/coulomb_auxiliary.cuh` nevertheless initializes and fills the
whole `n+t+u+v <= L` simplex for each primitive AO quartet. The consumer's
roots do not need all these states. Both full and LR values use this owner;
the high-order force Dual3 fallback repeats it for participating atoms.
The specialized full/LR all-center order-4/5/6 force owner is separate.

The new `GENERATIVEQC_DIRECT_COULOMB_REACHABLE=1` control (also `reachable`)
passes exact Cartesian component powers from the compiler consumer to the
existing recurrence. It is independent of angular scheduling and stays off.
Native J/K and generated force owners freeze the selection in their borrowed
device-basis view; the control participates in resource/checkpoint provenance.
Public-AO fallback and specialized low-order workers remain available.

## Exact work, without a timing prediction

For `L=X+Y+Z`, current auxiliary storage has

    S(L) = binomial(L+4, 4)

scalar states, zeroed and then assigned by the recurrence. Following the
existing x-first dependency edges backwards gives exactly:

- `t > 0`: `0 <= n <= X-t`, `u <= Y`, `v <= Z`;
- `t = 0, u > 0`: `0 <= n <= X+Y-u`, `v <= Z`;
- `t = u = 0`: `0 <= n <= L-v`, `v <= Z`.

The number of reached states is therefore

    S_r = (Z+1)(Y+1) X(X+1)/2
        + (Z+1) [Y(X+Y+1) - Y(Y+1)/2]
        + (Z+1)(L+1) - Z(Z+1)/2.

An independent backwards traversal checks all 455 axis triples through order
12. For orders 5/6/7/8, complete simplex counts are 126/210/330/495; reached
counts range over 21--43 / 28--67 / 36--102 / 45--147 across component powers.
These are exact static state counts, **not FLOPs, screened molecular work,
kernel durations or endpoint speedups**. No GPU calibration is used.

For actually evaluated component/primitive instances q, recurrence assignments
change from `sum_q S(L_q)` to `sum_q S_r(X_q,Y_q,Z_q)`, and the explicit zeroing
loop is skipped for the selected dependency domain. The scalar arithmetic of
every retained state and its order remain identical. The shell/primitive/AO
admissions, radial moment ladder and Hermite/component contractions do not change.
Their actual counts remain null until an attributable runtime census is available. Cold
repeats value work over SCF iterations, so it can benefit from this source
change even though a force-only schedule cannot remove that cost.

## Memory and correctness boundaries

The initial candidate preserves the original packed simplex and index formula:
`8*S(L)` bytes for FP64 values and `32*S(L)` for Dual3, before other workspaces.
No new persistent GPU allocation is introduced. Fewer source-level assignments
do not prove smaller compiler stack, less measured traffic or better occupancy.
A compact layout is deliberately deferred to avoid combining recurrence and indexing
changes. Component bounds may also add branches/registers, so a loss is possible.

Every retained recurrence operand has an earlier assignment in the existing
z/y/x evaluation order. Unused storage is deliberately not read. The local
auxiliary's `double`, `Dual` and `Dual3` arrays have no implicit zero initialization,
so their unused cells remain uninitialized on the selected path. In contrast,
`MixedPrecisionFloat` default construction initializes every array element to
zero even when the explicit loop is skipped. Actual zero-store elimination,
stack traffic and generated CUDA instructions require compiler/device evidence;
the source-level assignment counts do not establish them.
Tests poison the auxiliary storage with NaNs and check exact writes as well as
every consumed root.
Malformed or unselected bounds retain the full simplex, preventing unsigned
loop underflow. SR uses its own moments, never subtraction of full/LR sources.
At fixed exponents/omega the Dual/Dual3 geometry seeds differentiate the same
radial recurrence; the AO powers themselves do not increase.

`tests/python/test_reachable_coulomb.py` compiles the real native shared
arithmetic with a verified ccache host compiler, checks bitwise equality to the
retained simplex, and compares values and all three jets to independent interval
quadrature plus analytic Cartesian derivatives. This is a host arithmetic test,
not CUDA execution. The first harness attempt missed the public include root;
that failure and a subsequent C++17/C++20 header mismatch are retained under
the n1 experiment directory. The corrected n1 host run passes all 40 cases:
455 component-power triples for each of full/LR/SR (1,365 comparisons of every
root and its three jets), plus invalid-domain fallback. Forty existing compiler
and promotion-inventory checks also pass. Neither run executes a GPU.

## Required qualification and excluded routes

Before promotion: compile the actual CUDA call graph with ccache; run unchanged
independent fixed-density full/LR, both-spin and repeated-center tests; run
memory/init sanitizers because unused workspace is now intentionally poisoned
in host tests; compare same-binary off/on complete E/F warm and cold, original
and displaced geometries, and a larger size. Check public-AO/resource fallback
and checkpoint identity. Keep energy/force gates 1e-8 Eh / 1e-7 Eh/Bohr.

Do not enable angular scheduling while isolating this candidate. Do not change
AO/density screening, drop diffuse/f basis functions, enable the rejected
through-f bounded value route, or repeat rejected VV10 variants. The public
12-atom LDA lifecycle worsens prepared cold by 15.12% and remains excluded.
Open Becke #1830, ordinary force AO-map #1833 and density-product #1798 work
have separate owners and scientific contracts; this change does not duplicate
them. No measured gain or default promotion is established by this note.

## CUDA build and numerical qualification

Production `fb6f6335c` has source identity
`026543856fc338d54ee73fc451a264b9951ef06ed52856c964db8d0d6a35a086`
(1,381 inputs), library SHA-256
`3b90323d90ccdca0c193567fa5dec135be178eafafa5bea9c5d8dad13a2e134c`,
and native-test executable SHA-256
`aadfd7b014d5cc70d3fc95a3ed1b36c64fcf43ccea4e15047979b1fdbb4a7d7d`.
The incremental CUDA rebuild exits zero, with all 453 compiler commands using
ccache. Shared-cache snapshots increase by 225 hits and 47 misses; other users
of that cache are not excluded. The first failed build remains retained: a
missing policy declaration was repaired through `direct_coulomb.hpp`'s shared
source-preparation adapter, without introducing an HF-policy header dependency
in the canonical provider. Its existing compiler children drained before any
source was patched.

n1 RTX 5090 Slurm **5736**, device visibility **1**, passes both
`--range-response-only` and `--through-f-response` with the recurrence selection
off and on. These include independent full/SR/LR values and derivatives, both
spins, Cartesian/spherical basis conventions, separate J/K source masks, and
indexed-domain/prefix-budget fallbacks. The executed native test source SHA-256
is `7200a4c8f2b386aa17f31b28d249a50de32bc36ebdc2efab59c9bf16443a859d`.
Unlike the older angular experiment's test source, this revision includes the
two-center f/f/s/s fixture: nonzero order-12 derivatives are independently
checked for both angular schedules. The general source-mask test uses the
prepared default angular schedule; the internal RSH and four-center fixtures
explicitly exercise both. Do not infer angular-enabled source-mask coverage
from this job's unset angular environment control.

The same job passes candidate memcheck and initcheck with **zero errors** and
all **five** selected checkpoint policy cases. These are five parameterized
host cases, including the new recurrence control, not six or GPU checkpoint
coverage. Native-donor 3-atom and 12-atom ABBA both complete; job 5736 exits zero.
The installed CUDA 12.9 Compute Sanitizer documents
initcheck as **global memory** initialization checking; it does not prove that
every per-thread local auxiliary cell was initialized. The dependency proof,
NaN-poisoned host execution and independent device value/derivative oracles
remain necessary for that contract. Sanitizer output is an additional gate,
not a substitute for them.

The pilot performs a genuine target priming solve after native density import;
only the five subsequent frozen-density calls must take one iteration. An
independent audit verifies every prime/replay E/F, allocation, source/library,
donor, runner, numerical controls and semantic SCF AO work. It reproduces the
previous angular negative control under Python `-O` and rejects seven corrupted
variants (force, missing repeat, GPU, donor, control, timing and iteration).
There is no newly paired reference timing in this pilot. Prepared cold,
displaced geometry, larger endpoints and actual molecular recurrence counts
remain outstanding; a static closure reduction is not a speedup claim.

The complete 3-atom ABBA has ten measured one-iteration calls per selection:
off/reachable warm medians are **1.925322 / 2.014610 s**, a **4.638% regression**.
All 24 prime/replay E/F calls pass independent rechecking (maximum errors
1.990e-13 Eh / 9.377e-12 Eh/Bohr). Derivative-stage medians are
**0.872968 / 1.019439 s**. This is evidence against enabling the joint value/force
selection for this size. SCF full/LR value durations are not separately observed
in this replay, so subtracting endpoint/stage medians cannot establish a value
kernel speedup. The control remains off; no speculative size guard or cold
gain is promoted.

The completed 12-atom ABBA has ten one-iteration warm calls per selection:
off/reachable medians are **63.010484 / 56.627230 s**, a **10.130% reduction**.
All 24 prime/replay E/F calls pass independent rechecking (maximum errors
5.002e-12 Eh / 6.776e-11 Eh/Bohr). Both off groups bracket the candidate:
group medians are 62.945372 / 56.527716 / 56.664583 / 63.038442 s.
Derivative-stage medians nonetheless increase **33.128862 / 35.868590 s**.
This is an observed complete warm E/F improvement for a native-seeded
diagnostic, not prepared cold, a paired GPU4PySCF timing or a README point.
The audit result is `.artifacts/reachable-coulomb-20261004/comparison12.json`.
Molecular force-source counts and FLOPs remain null.

The next implementation exposes independent value/force selection so an actual
ablation can retain the old derivative schedule. It also adds a separate
[Hermite axis-convolution experiment](2026-10-04-hermite-axis-convolution.md).
The first comparison must keep convolution off; neither benefit is assumed
additive, and both controls remain default off.
