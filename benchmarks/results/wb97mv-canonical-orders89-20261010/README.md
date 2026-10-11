# WB97M-V automatic canonical orders eight/nine

Measurements start on 2026-10-10; qualification completes on 2026-10-11.
The matched timing base is fetched master
`ec04f1301e516d7f33031f43ae28d609b85fa0a3`. Three source-matched arms share
its force implementation: unmodified master (automatic order five), that
master plus the pending order-six/seven consumer, and the latter plus automatic
orders eight/nine. Their timings must not be relabeled as newer-master results.

The final integration base is fetched master
`b3a0eb2cfcb1e17b5e66936d97f0aaed527ad21f`, including its full-Coulomb
workspace overwrite change. Separate real-device integration gates protect
that source combination; they are not another matched performance experiment.

## Default contract and protocol

Eligible canonical FP64 sources automatically consume complete order-five
through-nine shell quartets from an existing immutable primitive-pair cache.
There is no new opt-in or method/basis/atom-count whitelist. For s/p/d/f, the
per-lane packet bounds are 1/2/3/6/9, covering 162/324/648/1296/2160 components.
Order-nine fddd, not ffdp, determines the nine-slot bound. Five row-prefix planes
cover disjoint angular blocks, including the order-eight equal-bucket triangle.
Cache absence, budget exhaustion and allocation denial retain bounded incumbent
consumers; MD-J retains priority. Component AO Schwarz tests, positive raw J/K,
primitive ordering, range identity and public projection keep their owners.
Other orders, fixed screening, compensation, resident values, combined
full-J/range-K, paired RSH, mixed precision and forces retain their consumers.

The fixture is the README **water proxy**, not an OMol25-distribution sample.
It uses complete spherical def2-TZVPD, exact Direct WB97M-V/RKS, strict FP64,
48 x 16 x 32 grid points per atom, three Becke iterations, E/density/screen
thresholds 1e-12/1e-10/1e-12, max100, VV10 density threshold 1e-8 and the existing
4 GiB incremental force budget. Independent gates are 1e-8 Eh / 1e-7 Eh/Bohr.

All matched timings and the independently seeded reference share finite Slurm
job 7230 on node1/RTX 5090. Device visibility is preserved. Timing includes
prepare, SCF, physical forces, synchronization and host publication; imports,
context and Calculator construction precede the clock, while teardown and
serialization follow it. Compiler/artifact caches are preserved, with verified
ccache 4.5.1 and recorded compiler commands/statistics. This is fresh-owner
cold execution, not empty-cache compilation. First-use allocations are retained
separately and excluded from the three alternating 3-atom repeat medians.

## Complete E+F

| Metric | Frozen master | Automatic 5–7 | Automatic 5–9 |
| --- | ---: | ---: | ---: |
| 3-atom median seconds, three repeats | 12.700011 | 11.980636 | 11.821136 |
| 3-atom iterations / Focks, every repeat | 15 / 15 | 15 / 15 | 15 / 15 |
| 12-atom seconds, one matched triplet | 529.333544 | 344.358152 | 248.930230 |
| 12-atom prepare seconds | 0.998427 | 0.975334 | 0.986216 |
| 12-atom SCF/publication seconds | 492.185078 | 307.167360 | 212.418862 |
| 12-atom physical force seconds | 36.150039 | 36.215458 | 35.525153 |
| 12-atom iterations / Focks | 21 / 21 | 21 / 21 | 21 / 21 |

Orders eight/nine reduce the 3-atom median by **1.33%** and the 12-atom complete
endpoint by **27.71%** relative to 5–7. The cumulative 5–9 reduction against
frozen master is **52.97%**, or **2.13x**. The 12-atom result is one triplet,
not a repeated median. Candidate errors are 5.230e-12 Eh / 2.818e-11 Eh/Bohr.
The improvement is in SCF, not fewer Focks or a changed force domain. AO work
(excluding discovery timing), force counters and force traffic match across arms.
Unavailable post-screen integral counts remain unavailable, not inferred from
logical capacities.

### Independent GPU4PySCF reference

Installed GPU4PySCF 1.8.1 / PySCF 2.14.0 / CuPy 13.6.0 supplies the independent
reference. Its explicitly **full-Fock rebuild** diagnostic converges at the
unchanged gates/grid/max100 with 45 iterations / 46 Focks and complete E+F
**182.736137 seconds**. No native/reference density seed is shared.

Native remains **1.362x slower** than this reference. This is not a stock-default
GPU4PySCF ratio: the earlier stock incremental 12-atom run did not converge in
100 cycles. The reference diagnostic records `reference_full_fock=true`
separately; its inherited protocol `reference_fock_policy=incremental` field is
not its actual solver policy.

## Source diagnostics and resource bounds

All 56 first-density 12-atom source actions total 20.819443 / 12.021961 /
7.511317 GPU seconds for master / 5–7 / 5–9. Orders eight/nine fall from
3.414476/1.695408 to 0.402108/0.202062 seconds. These are source actions,
not complete E+F timings. Representative restricted order-eight/nine kernel
attributes change from 176/190 registers and 9064/10824 local bytes to
146/132 registers, 768/896 local bytes and 10264/12024 shared bytes. Local
memory is bounded, not eliminated; resources for the newer integration are
recorded separately.

The 96-atom capture selects only first-density actions 14/19/42/47: full and
LR order-eight/nine blocks. All other source classes are deliberately omitted;
execution exits before publishing E/F.

| Selected 96-atom action | Automatic 5–7 seconds | Automatic 5–9 seconds |
| --- | ---: | ---: |
| Order eight, full J/K | 89.547852 | 22.938420 |
| Order nine, full J/K | 64.888852 | 17.972199 |
| Order eight, LR K | 86.776836 | 22.736551 |
| Order nine, LR K | 62.872410 | 17.860695 |
| These four actions only | 304.085949 | 81.507865 |

These selected sources improve **3.73x**. The six requested matrix channels
differ by at most **4.253e-14** (1e-8 absolute consistency gate). Unrequested
legacy ABI J buffers are not compared. Baseline outer counts denote **AO
quartets** and candidate counts denote **shell quartets**; their ratio is not
a same-unit semantic work reduction. Complete 96-atom E+F timing and an
independent physical 96-atom gate remain **null**.

## Qualification and retained evidence

Both frozen and newer integration builds pass four native suites, eight
independent Libcint cases (Cartesian/spherical, RHF/UHF, full J/K and SR/LR
K-only), complete/tail/empty/coincident/signed primitive-work gates through
order twelve, and all four persistent-CTA sanitizers with zero errors/hazards.
The newer-master 3-atom first-use endpoint passes the independent physical
gate in 27.977653 seconds; it is qualification only, not a matched speed claim.
Host Direct tests report 832 passes; compiler structure checks 511 modules
with zero errors. Source/binary hashes and pre-commit results are retained.

[evidence.json.gz](evidence.json.gz) owns compact acceptance, per-arm identities,
resource attributes, work counters, source reconstruction and the raw archive's
SHA-256 inventory. Raw journals, profiles, matrix dumps and qualification
binaries remain in the stream-verified local archive under
`/data/jzzeng/wb97m-canonical-order89-20261010-ec04f1301`, not a Release or Git
history payload. The initial test-construction preflight failure and corrected
full rerun are retained separately rather than discarded.

The three receipts for automatic orders five, six/seven and eight/nine use
lossless deterministic gzip to remain within the tracked evidence budget;
decompression reproduces their exact JSON bytes. For this receipt:

```bash
gzip -dc benchmarks/results/wb97mv-canonical-orders89-20261010/evidence.json.gz | python -m json.tool
```

Orders three/four/five now account for about 55.5% of this 12-atom source
capture; ten through twelve account for about 6.7%. Dedicated lower-order
scheduling is the next evidence-led target, not blindly promoting either the
previously regressing lower-order CTA or unqualified higher-slot consumers.
