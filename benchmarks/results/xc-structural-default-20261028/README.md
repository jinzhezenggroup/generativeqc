# Default structural XC sharing qualification

The ordinary General entry now uses an immutable grid-constant parameter row.
Restricted ordinary and both batched entries retain their existing storage ABIs;
the [paired-input RKS repair](../xc-shared-point-abi-20261028/README.md) is retained.
Every compatible group still instantiates four kernels. Selection uses complete
mathematical and execution contracts, with no functional-name exception.

`discover` is the default in CMake, the standalone registry generator and the
compiler package. This applies inside enabled automatic XC CUDA AOT; its master
switch remains opt-in. `qualified` restores the legacy prepared-product selection,
and sharing OFF restores literal specialization. Unsupported methods, precision
or derivative products do not acquire admission through parameter sharing.

## Artifacts and scientific scope

The final selected package contains 73 registrations in 12 groups: **48 versus
292 point kernels**, and **7,863,216 versus 41,532,592 object bytes**. These are
selected XC objects, not the full native library. Cache-aware compilation receipts
are retained but are not a controlled cache-miss compile-time comparison.

Both packages, the complete native library and the CUDA numerical driver use the
same frozen producer and integrated native sources. CMake's actual default shared
registry and CPU registry are byte-identical to the independently generated audit
products. Only the selected CUDA XC objects differ between complete endpoint arms.
`receipts.json.xz` retains object/source/binary identities, complete link commands,
toolchain/cache receipts, launch scripts, raw timing samples and scientific work.
Exact observed event values are interned in a sorted table and each sample stores
an integer offset from its process base index. This is lossless storage, not
rounding: the reducer restores every floating value and verifies each original
population's canonical SHA-256 before applying any gate.

The source base is published master `a3ec1f8fd`; exact overlays reconstruct the
measured `99ffc690d` integration plus the uncommitted candidate. The optional
reconstruction authenticates the native source identity and all 34 files in each
CUDA package plus 10 CPU files. Generated products and disassembly remain transient.

## Timing protocol and gates

Hardware: Slurm node2 PRO6000, SM120, NVCC 12.9.86, GCC 11.4 and verified sccache
0.16.0. All GPU execution uses finite `srun` allocations and preserves assigned
device visibility. CPU compilation reused the existing compiler cache.

The initial point campaign covers all 73 shared registrations, both spins,
16,384/262,144 points and ordinary/tile-127 traversal: **584 cases and 3,504 process
measurements** in three randomized paired rounds. Each process records 50 CUDA
events after 10 warm launches. Compare the median of process medians; retain every
observation. During this pass, the complete native library was compiling on the
same host; no other GPU job ran concurrently.

Before that pass finished, a confirmation rule was fixed: after compilation ends,
run five additional randomized paired rounds for every initial ratio above 1.01
and the 80 original pilot control cases. Keep initial and confirmation populations
separate and also report their pooled medians. Four initial cases exceed 1.01;
only ordinary UKS APBE 16K remains above it after confirmation, at 1.011221.
This direct-launch population does not pass a universal 1% call-span gate.

Those event spans include a CPU submission interval after the start event; the
10-launch warmup also does not control duration across small and large kernels.
A separate diagnostic therefore captures exactly one point kernel in a CUDA Graph,
warms it for at least 100 ms and records the same 50 events. This measures GPU
execution with a common graph submission path. It changes no production kernel,
parameter row or traversal. All 32 pilot cases pass 1.01; APBE ordinary UKS 16K is
0.999643. Moving to 64 threads does not establish a consistent improvement, and
256 threads substantially regresses several cases; retain 128 threads.

Final GPU qualification applies the graph protocol to **every ordinary case**,
not just initial failures: 292 cases and 1,752 process measurements. All 292
batched cases retain their original direct-launch protocol and gate. The boundary
remains 1.01 in each declared GPU case. GGA_C_MGGAC UKS 262K initially measures
1.010087. A fixed five-round confirmation covers every member of its discovered
contract plus APBE/KTBM_0/OW controls, both spins and sizes (56 cases, 560
measurements); retain the initial values and use all eight rounds for those cases.
Report both protocols and all failed direct
call spans; do not pool their timings or claim a universal host-launch speedup.
Complete endpoints below include real preparation and submission costs.

Complete energy endpoints use every group representative, the APBE/PBE_SOL/XPBE
family, TH/BKL repairs and KTBM controls, plus four-water APBE RKS. Each cohort
has five paired processes. Each process includes configuration/preparation in the
cold timer, three warm calls with frozen seed updates, and a displaced geometry.
The acceptance boundary is 1.02 for each cold/warm/moved cohort, equal semantic
operator work, iterations and Fock builds, and maximum paired energy error 3e-9 Eh.
Oracle work runs separately from these timed production calls.

Final acceptance covers **34 converged endpoint cohorts and 1,700 complete
calls**. Semantic work, iterations and Fock counts match in every pair. Maximum
paired energy error is **2.27e-13 Eh**; the largest cold/warm/moved median ratio is
**1.008677**, below the 1.02 gate. Across the declared ordinary Graph and batched
direct-event point cases, the maximum ratio is **1.008197**, below 1.01. Ordinary
Graph cases alone peak at 1.007130. The selected XC objects are **81.07% smaller**.
These are qualification boundaries, not a claim of universal endpoint speedup.

## Independent numerical qualification

The retained host campaign has 296 compiler/launcher/CMake passes plus 23
source-identity/consumer-contract passes and 10 subtests. Generated host launchers
execute against independent Libxc, including both selectors and partial tiles.
The real-device campaign covers every shared registration's points, invalid-input
errors, selector rollback, fixed-density Fock, replay and directional derivatives.
Public PySCF KS oracles cover representatives in cold/warm/moved phases, with an
additional exact-grid four-water RKS oracle. There are **681 passes**, one existing
MVSB UKS skip, and one retained larger UKS failure. The raw JUnit and per-case receipts retain
all numerical tolerances, binaries, scientific identities and work ledgers.

MVSB UKS H4 retains its pre-existing literal/shared convergence limitation as one
explicit skipped complete-endpoint case. Its point and fixed-density checks remain
required. The original failure receipts remain in the preceding campaign; this
change neither repairs convergence nor certifies arbitrary geometries. The added
four-water UKS case also fails to converge in 400 iterations in both frozen arms;
its failed energy, residual histories and incomplete work ledgers are retained,
not treated as converged energy agreement. It is excluded from positive timing
qualification. The four-water RKS cold/warm/moved oracle passes, with maximum
energy error 1.14e-12 Eh and Fock error below 1e-13.

## Pilot alternatives

All 1,200 initial pilot measurements are retained: 80 cases, three rounds and five arms
(literal, previous shared, sign-column merging, ordinary grid-constant storage,
and their combination). Ordinary storage passes the pilot's 1.01 boundary versus
both literal and previous shared. APBE ordinary UKS 16K changes from 1.01056 to
1.00721 versus literal. Sign merging alone does not repair APBE and regresses
GGA_C_APBE RKS ordinary 16K to 1.0160 versus literal and 1.0274 versus previous
shared; it was rejected. The combined candidate passed the pilot but was not
selected because it adds that unnecessary transformation.

The two additional scheduling pilots each retain 384 measurements, including
all 64/128/256-thread results. The original Slurm 3663 numerical run was cancelled
after the confirmed direct-span failure, before completing its numerical suite;
its partial log is retained and is not counted as a pass. Final numerical and
endpoint qualification uses the same frozen production artifacts in a new job.

The old all-entry grid-constant failure remains in the
[preceding receipts](../xc-shared-point-abi-20261028/README.md). Static compiler
resources are diagnostic evidence, not measured dynamic stalls. These results
qualify the recorded compiler/device domain; explicit fallbacks remain available
when qualifying another architecture.

## Replay

Normal replay requires no compiler, GPU, network or historical Git objects:

```bash
python benchmarks/results/xc-structural-default-20261028/reduce.py
```

The optional source audit needs the published base locally and normal compiler
Python dependencies, but no native compilation or GPU:

```bash
PYTHONPATH=python:. python benchmarks/results/xc-structural-default-20261028/reproduce_sources.py \
  --output .cache/xc-structural-default-audit
```

See the [decision note](../../../.agents/notes/implemented/performance/2026-10-28-xc-structural-default.md)
for the storage-ABI rationale, bounded fallbacks and revisit conditions.
