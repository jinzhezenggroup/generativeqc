# Retained CUDA AOT default-policy input

`production-shell-classes.json` is the exact stock manifest from
`44f34575f0c91b846c7265e5c388f068de09826c` (the P1b policy-dimension repair).
Its identity is retained in
`manifests/maintenance/cuda_aot_default_decisions.json`.

`effective-defaults.json.xz` retains the corresponding semantic resolution for
the audited stock targets: schedules, recurrences, consumer/route masks,
preferences, packaging and target facts. The gate compares current resolution
with this immutable data rather than re-resolving the old manifest under the
new compiler, which could otherwise hide implicit policy changes. This is a
policy contract, not a generated-source or function-body fingerprint.

This is an immutable historical **input**, not a newly qualified result, new
benchmark campaign, or authority to promote additional defaults. It preserves
the existing tuned/portable decisions while the checker evaluates future
semantic changes against source-matched cold/warm/moved campaigns. Formatting
of the current manifest is not compared byte-for-byte with this input.

The preceding P1b migration compared generated CUDA, registry and derivative
outputs against `8d2ad48175b79f2a3cb5749d3552413cdf5b7d4e`: stock auto bundles
for SM80/SM90/SM120 and forced-portable SM120 were byte-identical. That was a
one-off migration observation, not a permanent generated-source fingerprint.
No GPU numerical or timing result is asserted by this retained manifest.

Existing scientific and performance records keep their original source and
device identity. In particular, retained provisional f-shell selections are
not certified merely because they are present in this manifest.

## Source-matched compile-only acquisition

`sm80-sm90-compile.json.xz` retains input identities, compiler logs, CUOBJDump resource
text, toolchain versions and acquisition index for 88 translation units built
from the P1b source above, native source identity
`56fde0d6534cbfd3c003df9ca385c08e50f46c4ae0d3586d45eee31f76288eef`.
All 44 stock SM120 primary selections were retargeted, audit-only, to SM80 and
SM90. Compilation succeeded for all 88 units. The generated primary full-range
Fock/force and declared mixed/streaming wrappers were included; optional
competitors and exact SR/LR derivative packages were not.

The acquisition used CUDA 12.9.86, GCC 11.4, C++20, the compiler adapter's release
optimization and sccache 0.16.0 on n2, through finite Slurm job 5297
(`main`, `gpu:pro6000:1`, one hour requested, visibility `0`). This node's actual
device is RTX PRO 6000 Blackwell/SM120: this was **not A100 or H100 execution**.
No candidate was enabled in production or executed on a GPU. Resources are
observations, not a resource-admission/nonregression result. The initial GCC12
attempt failed because its host `cc1plus` was unavailable; that infrastructure
failure remains separately retained locally and is not a CUDA correctness
failure. Completed-job controller accounting is unavailable on this cluster;
the bundle distinguishes the submitted request from an accounting receipt.

Large objects and generated input sources remain in ignored acquisition storage;
their hashes, source/emission contract and raw resource output are retained here.
`python tools/check_cuda_aot_policy.py --verify-compile-inputs` optionally
regenerates inputs from matching source and checks their acquisition hashes.
This is not a permanent required-CI generated-source fingerprint. Shared sccache
statistics do not observe the adapters' owned per-invocation servers, so no cache
hit rate is claimed. The bundle protects integrity, not acquisition authenticity.

`tools/check_cuda_aot_policy.py` verifies retained log/resource hashes
and parses per-kernel resource observations before attaching only the covered
source-matched feature cells. Exact-package, optional-competitor, device-numerical
and endpoint cells remain not-run. After native source changes, this acquisition
becomes historical automatically rather than imposing a generated-source
fingerprint on future refactors. It never promotes a default.

## Source-matched SM120 endpoint diagnostics

`sm120-endpoint-smoke.json.xz` retains both Release build contracts, native
profile/ABI receipts, CUOBJDump text, acquisition scripts, independent PySCF
2.14.0 Cartesian reference results and all raw stock RHF E+analytic-force
process records. Both binaries use the same P1b source identity above. Only the
AOT profile differs. The native profile test passes for both auto (`sm_120`)
and forced portable (`portable_cuda`), with a matched nonempty registry.
The common bounded build disables Hessian AOT leaves and CPU stationary-force
AOT; it is not a claim about a release distribution's full package/startup cost.

Finite Slurm job 17464 (`main`, `gpu:5090:1`, 30 minutes) runs four fresh
processes per arm for each basis, balanced in order, on the same physical
RTX 5090 UUID. Each process executes cold, warm, moved and moved-warm endpoints.
Cold includes import, profile probe, construction, preparation and force host
export; the other phases include synchronous execution/export. This is
application-fresh timing, not an OS/driver cache-cold experiment. The 16 process
records contain actual SCF/Fock counts and explicit invalid quartet counters;
unavailable counters must not be invented or relabeled as zero work.

| Stock case | Auto maximum force error | Portable maximum force error | Outcome |
| --- | --- | --- | --- |
| Water/def2-SVP, all four phases | 5.611e-8 | 5.611e-8 | All 32 phase observations pass |
| Water/def2-TZVP, cold and warm | 6.759e-7 | 6.759e-7 | All 16 observations fail the unchanged 1e-7 force gate |
| Water/def2-TZVP, moved and moved-warm | 4.485e-8 | 4.485e-8 | All 16 observations pass |

All energy and translation gates pass. The failed TZVP stock results remain
failed; they do not establish a portable-only defect. A separate, explicitly
nondefault diagnostic in finite job 17467 tightens SCF convergence identically
in both arms (`energy_tolerance=1e-12`, `density_tolerance=1e-10`). Initial SCF
iterations increase from 9 to 16 and maximum force errors fall to about 1.12e-10.
This supports a convergence-limited stock force discrepancy rather than an
AOT-profile-specific mathematical discrepancy. Those diagnostic records are
kept separate and never replace stock acceptance, change production tolerances,
or enter a performance ranking.

The checker replays raw reference/run hashes, process ownership, source,
physical-device and library identities, complete phases and numerical outcomes.
It reports the failed arm/phase groups without failing mere evidence-integrity
validation or declaring a default qualified. No per-shell execution coverage,
A100/H100 numerics, RTX PRO 6000 endpoint equivalence, full #3084 cohort rerun,
or global-default performance win is claimed. Small-water timings cannot settle
the production workload's portable-profile slowdown.
