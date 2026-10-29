# Required native progressive-HF policies — CPU qualification

Production/current native build: `e2157668e` (full revision in the capsule).
Baseline frontend: `131144549f91223cf751a27874c365834cc7c443`.
Both frontends use the current library, SHA-256
`fa9450537f4018bfb472503a8de41b90ac69abc5e321cdd83e1b4e65e3a51b8a`.
Native/build/compiler inputs are unchanged; this removes missing-protocol Python
policy selection without changing scientific equations, ABI or backend admission.

The unchanged complete progressive-HF protocol runs RHF/UHF × Direct/DF,
alternating two genuinely different H2/H2+ geometries. Each ABBA and BAAB block
performs three warmups then 21 measured samples, for **672 timed endpoints**.
Each endpoint constructs source/target calculators and the target snapshot/plan,
executes source SCF, projection, target energy/forces, fixed-density physical
audit and owner cleanup. Imports are excluded; no density persists across samples.
OMP/OpenBLAS/MKL use one thread and paired arms share CPU 40. No exclusive-host,
speedup, new numerical domain or CUDA claim is made.

All eight median-ratio gates pass the unchanged 1.05 limit; maximum ratio is
**1.0016859405931222**. Every corresponding measured sample has exactly equal
source/target energies, full target force and density arrays, physical residual,
iteration/Fock counts, projection/restart and final disposition. Establishing the
target does not manufacture independent observed accuracy: the final status may
remain `unverified` without an independent accuracy assessment.

The earlier shared-image campaign also completed 672 endpoints, maximum ratio
1.0041193828828907. Its image was built at `fafe44f53`. The current-source rebuild
produced a different binary, so the complete protocol was repeated above; both
populations retain their own image identities and outcomes.

The cached current-source Release build/install passes all 127 CTests and 95
focused Python checks, including independent PySCF physical audits, exact/below
capacity, integer/cost oracle comparisons, required-protocol refusal, all final
gates and progressive execution. Both planning and final-verification SDK probes
compile against installed public headers and run with `PATH=/nonexistent`.
Verified sccache 0.16.0 wraps the compiler; shared cache counters are retained as
provenance, not isolated compilation performance.

`evidence.json.xz` contains every timed sample in both campaigns, exact replay
source/command, native/frontend/binary identities, SDK receipts, test summaries,
cache counters and ignored-log hashes. Decode it with
`json.loads(lzma.decompress(path.read_bytes()))`. Reuse the retained runner with
separate baseline/candidate checkouts and the qualified candidate image; preserve
all four cases, both orders, three warmups, 21 samples and the 1.05 gate.
