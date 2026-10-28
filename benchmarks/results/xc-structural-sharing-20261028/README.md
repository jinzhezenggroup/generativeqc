# Structural XC sharing qualification

The planner scans all selected automatic XC registrations after literal
specialization and differentiation. It lifts arithmetic constants, preserves
predicates/fixed powers/zero terms/CSE, and compares complete paired/general ABIs.
It restores group-common coefficients and deduplicates identical launch columns.

The 197-registration census finds 12 groups covering 73 registrations. The
qualified default retains the original 26 KTBM registrations; `discover` is an
explicit opt-in for the remaining candidates. The full catalog has 788 literal
point kernels, 688 with the qualified policy, and 544 with discovery. This census
does not expand public functional admission or claim that all Libxc functionals
can share a single worker.

The selected 73-registration package compares complete literal specialization
with all 12 shared groups. NVCC 12.9.1 targets SM120 on Slurm node2 with a PRO6000
allocation. Builds use verified sccache 0.16.0 and retain its statistics; caches
are never cleared. Build times are cache-aware observations, not a controlled
cache-miss compiler-speed claim. Exact producer reconstruction, generated-source
identities, commands, object and binary hashes accompany the measurements.

The selected package's 33 objects shrink from **41,532,592 to 7,997,872 bytes**
(**80.74%**); its point kernels shrink from **292 to 48**. These are package
figures, not an 80.74% reduction of the complete native library.

The first randomized paired pilot covers the first and last registration in
every group, both spin layouts, 16,384/262,144 points and ordinary/tile-127
traversal: 192 cases, with 50 event samples after 10 warm launches per arm.
Most new families are near neutral, but some small paired batched TH and BKL
cases regress by approximately 2–3%. A separate final-binary confirmation retains
both repetitions for those families and KTBM controls. No slow sample is excluded.
The confirmation contains 48 cases and 96 paired measurements. Its median ratio
is **0.999880**, but the maximum is **1.029765**; four cases exceed 1% regression.
The existing per-case point gate therefore does not support blanket promotion.

Independent Libxc checks cover all 73 rows, ordinary/batched layouts, both spins,
domain errors, explicit equal-spin disable, fixed-density AO/XC/Fock, replay and
directional derivatives. Public PySCF KS checks cover one member of each of the
12 groups in RKS and UKS. Complete endpoint timings use alternating fresh-process
pairs with cold preparation, three frozen-seed warm calls and displaced geometry,
plus a larger four-water RKS control. Raw work/precision ledgers and all energy
comparisons are retained rather than treating point timing as an endpoint result.

There are **657 passing real-GPU point/Fock checks**, **23 passing independent
public-KS checks**, and one unqualified MVSB UKS cohort that also fails to converge
with literal specialization. The 150 endpoint processes retain all six failed
attempts (three per arm for that cohort). The other 24 cohorts provide **720
complete calls** with identical iteration/Fock counts and precision/work ledgers;
maximum paired energy difference is **2.274e-13 Eh**. Cold/warm/moved phase median
ratios span **0.980498–1.010619**. The isolated larger control is four-water APBE RKS.
These endpoint results do not override the failed point-case gate.

Initial complete-library attempts correctly failed the CPU/CUDA scientific ABI
gate: the cached native CPU registry and newly emitted CUDA descriptors belonged
to different compiler-source identities. Both registries were regenerated from
one frozen producer snapshot and relinked against the same remaining native
objects. Failed attempts remain in the receipt. This is why compiler-byte changes,
including changes without new arithmetic, must regenerate both registry owners.
The exported library source metadata describes the reused native build; exact
binary hashes and the recorded replacement-object links bind these experiments.

The publication changes only formatting of the measured package planner; a
one-off identical-AST check records that distinction. Both publication registries
were regenerated together, compiled and linked again. Three final RKS smoke
cohorts (APBE, BR78 and KTBM) pass all five cold/warm/moved calls. Separate
publication source bytes, generated-file identities and binary hashes are retained.
The default-policy comparison uses the prior package planner with the same
canonical producer inputs; it does not reuse stale descriptor identities.

The `structural_discovery` record in the
[consolidated receipt](../xc-shared-point-abi-20261028/receipts.json.xz) contains
the complete populations, independent-oracle receipts, initial failures, exact
producer sources, generated-source identities, compiler/cache provenance and
experiment drivers. Every numerical record is unchanged by consolidation with
the subsequent repair campaign. Generated products, binaries, objects and raw
disassembly remain ignored local artifacts. The reducer checks recorded
source/object bindings and paired numerical/work gates, and reports every timing
case, including failed gates:

```bash
python benchmarks/results/xc-structural-sharing-20261028/reduce.py
```

The [source audit](../xc-shared-point-abi-20261028/reproduce_sources.py) can
rebuild both CPU/CUDA source packages using `--snapshot structural` or
`--snapshot publication`. It uses a published ancestor plus retained producer
overlays and does not depend on the unmerged discovery PR head.

See the [decision note](../../../.agents/notes/implemented/architecture/2026-10-28-xc-structural-parameter-discovery.md)
and [current guide](../../../docs/developer/xc_native_cuda.md). This evidence
supports explicit discovery and retained specialization, not a new global
performance default or qualification on other GPU architectures.
