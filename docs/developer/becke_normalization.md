# Becke Normalization Scheduling

The stationary Becke phase lowering uses the canonical grid-response ratio AD
and normalized-product helpers in `generativeqc_compiler.xc.grid_native`.
The cooperative normalization schedule changes execution, not the partition,
screening thresholds, derivative formula, or atom domain.

## Execution And Ordering

An admitted block has eight contiguous point lanes and sixteen atom lanes,
with 128 threads total. Atom lanes independently find partial log maxima,
evaluate scaled products and write product adjoints. One lane per point retains
the original atom-ordered denominator sum and invokes the same ratio AD used by
serial normalization. The ratio result is shared with the atom lanes.

Products and adjoints retain the existing global workspace layout. Exact-zero
products are not exponentiated, and zero numerator owners retain their ratio
adjoint. The maximum remains a frozen common scale, as in the canonical serial
path. Pair primal, reverse, atom gather and point-motion phases are unchanged;
the schedule does not reduce their pair visits or normalization atom entries.

All block barriers execute for partial point blocks and failed points. Invalid
owners, nonfinite seeds, absent finite maxima and previously set device errors
retain the failure/publication contract. Floating-point atomics are not used.

## Admission And Fallbacks

The existing optional phased Becke reservation remains the prerequisite. There
is no additional retained device allocation. Cooperative normalization supports
at most 128 atoms and requires actual device/kernel thread and static shared
memory limits to admit the block. Any miss retains the original serial
normalization kernel; losing optional pair-coefficient experiments remain
independent and default off. Generic geometry/AD fallbacks remain unchanged.

The private `_CudaSources` construction argument `becke_normalize` allows
qualification to select a serial or cooperative schedule before topology.
`None` retains the artifact's default, including legacy serial artifacts;
an explicit request requires the versioned configuration ABI. Neither a second
configuration nor a live-topology configuration is allowed.

`stationary_becke_normalize_metrics_v1` returns four cumulative owner words:
supported, selected, phased batches and phased points. Difference counters across
a complete force call; do not label cumulative replay totals as per-call work.

## Qualification

`tests/python/test_becke_normalize_cooperative.py` compares CUDA with the serial
AD path and an independent NumPy expression, including tails, exact zeros,
invalid input and the oversized-domain fallback. Real-device execution requires
a finite Slurm GPU allocation and assigned visibility.

The additional three-atom RKS/UKS analytic and reconverged finite-difference
gates test the deliberately retained small-domain fallback under both schedule
requests. Native phased reservation starts above 32 atoms; these are not
physical tests of admitted cooperative normalization. Complete 48/96-atom RKS
vectors independently qualify the admitted path.

`benchmarks/becke_normalize_pairs.py` uses the ordinary PBE0/def2-SVP reference
protocol and at least five interleaved complete energy-plus-force pairs. It
freezes the public native warm snapshot, replaces and primes whole gradient owners
outside timing, and retains every prime, setup call, force vector and actual SCF
history. Setup is not a matched cold/moved performance comparison.

`--profile --feasibility --repeats 1` produces a separate intrusive campaign
using the existing seven Becke intervals. Never add those intervals to clean
endpoint wall time or use feasibility-only observations to establish promotion.
