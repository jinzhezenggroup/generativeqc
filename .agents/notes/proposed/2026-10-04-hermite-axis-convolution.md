# Experiment: convolve Cartesian pair coefficients before consuming roots

Status: proposed (default-off implementation; host arithmetic qualified)
Date: 2026-10-04

## Motivation and distinct scope

The accepted full-TZVPD 12-atom profile attributes 89.49% of warm time to
full/LR values and derivatives; total orders 5--8 dominate value duration.
These timings belong to the source/device identified in the
[integral diagnosis](2026-10-04-tzvpd-warm-integral-diagnosis.md).
The separate reachable-state experiment preserves the six-loop coefficient
contraction. It regresses three-atom E/F by 4.638%; completed 12-atom ABBA
reduces E/F from 63.010484 to 56.627230 s (10.130%), while derivative-stage
medians increase from 33.128862 to 35.868590 s. That does not qualify a size guard or establish
which SCF value kernel benefits by subtracting stage medians.

The next experiment addresses repeated coefficient products themselves.
`direct_cartesian_contraction_cuda.py::eri_cartesian_value` has six pair-index
loops, but its root depends only on the three sums of these indices. For each
axis, define pair powers A/B and

    H[k] = sum_(t+u=k) E_AB[t] (-1)^u E_CD[u].

The same real-valued integral is

    prefactor * sum_(x,y,z) H_x[x] H_y[y] H_z[z] R(0,x,y,z).

The compiler owns this lowering. Native preparation freezes the control behind
the existing shared source adapter, preserving canonical/HF ownership boundaries.
No reference or CPU scientific work is introduced into production.

## Controls and retained paths

`GENERATIVEQC_DIRECT_HERMITE_CONVOLUTION=values` selects generic FP64 values;
`forces` selects Dual/Dual3 derivatives; `1`/`all` select both. Absent, `0` or
unrecognized values retain the old contraction. Only generic total orders
five and above use the experiment. This is a diagnostic scope based on the
observed high-order hotspot, not a proven crossover or promoted policy.
Mixed precision is explicitly excluded from this reassociation. Public-AO
fallback and separate low-order/all-center force owners remain unchanged.

`GENERATIVEQC_DIRECT_COULOMB_REACHABLE` separately gains the same values/forces
choices; its original `1`/`reachable` aliases continue to select both. Its
existing mixed-value behavior is retained. Each choice is frozen and recorded
in the resource/checkpoint identity, including explicit admission of historical
checkpoints lacking the new control. Neither control becomes automatic.

Value/force selectors allow direct attribution experiments rather than assuming
that the two consumers should share the same schedule. Initial comparisons
must leave the other experiment off; composition is a separate qualification.

## Work and memory

For one actually evaluated primitive/AO instance with pair powers A_d/B_d,
the original contraction visits

    T6 = product_d((A_d+1)(B_d+1))

pair-index contributions. The candidate visits

    Tconv = sum_d((A_d+1)(B_d+1))
    Troot = product_d(A_d+B_d+1).

These term types have different arithmetic and are not interchangeable FLOPs.
For balanced total degree eight an example is 144 original terms versus
17 convolution plus 45 root terms. At balanced degree twelve it is 729 versus
27 plus 125. For A=(5,0,0), B=(0,0,0), both routes have six root terms and the
candidate adds preparation. Neither these examples nor total degree alone
establish an endpoint gain. Molecular invocation/shape counts remain null.

The packed three-axis weights need at most `(L+3)*sizeof(Scalar)` additional
local storage: 120 bytes FP64 or 480 bytes Dual3 at L=12. The much larger
Coulomb auxiliary is unchanged. There is no new persistent GPU allocation.
Compiler stack, spills, dynamic local traffic and achieved occupancy are not
inferred from the source array sizes. #1787's homogeneous probe calibration
does not qualify this heterogeneous high-register call graph.

## Correctness and minimum validation

This is exact algebra over real numbers but changes floating-point rounding,
including cancellation. The original recurrence proof's bitwise root equality
does not qualify the new contraction. Malformed powers exceeding the compiled
workspace bound fail closed; every used weight is initialized before reduction.
Full/LR/SR use their independent moment owners, never a new SR subtraction.

Host tests compile the emitted C++ and actual shared native arithmetic with
verified ccache. The independent oracle integrates the correlated Gaussian
Laplace measure using Wick moments; derivative oracles raise/lower primitive
polynomial powers. Through-f orders 0--12, distinct/repeated centers, diffuse
exponents and common translations check normalized values and all xyz jets.
Old/new recurrence and old/new contraction are crossed; mixed reassociation is
explicitly checked absent. CPU-only qualification on n2 passes all 439 selected
cases in 12.59 s: 157 new arithmetic/oracle cases, 10 actual native policy-parser
cases, 192 allocation/stack/frozen-policy cases, and 80 existing recurrence,
compiler and inventory cases. Normalized oracle gates are rtol=3e-10 and
atol=2e-11. Compiler cache version and pre/post snapshots, exact deployment and
exit-zero receipt are retained under
`.artifacts/hermite-convolution-20261004/`. All repository hooks pass.
These checks do not execute CUDA or establish an endpoint gain. No n2 GPU
calibration is used for the RTX 5090 measurements.

Before promotion: run unchanged independent molecular full/LR source gates,
both spin conventions, sanitizers, prepared-policy/fallback checks, then
same-binary complete cold/warm/displaced E/F and a larger endpoint. Keep the
full def2-TZVPD basis and 1e-8 Eh / 1e-7 Eh/Bohr gates. A successful warm pilot
alone is insufficient. Retain failed builds, numerical failures and slower
variants. Node3 remains excluded; no merge or README claim follows from the
implementation alone.
